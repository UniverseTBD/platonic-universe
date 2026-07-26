#!/usr/bin/env python3
"""Step 1 — distributed extract + linear probe.

For each (model, size, modality) tuple in MODEL_GRID × MODALITIES, this
worker:

  1. Extracts a (n_use, d) embedding matrix on the COSMOS-Web sample,
     caching the .npy locally.
  2. Optionally uploads the .npy to ``$PU_EMB_REPO`` so other clusters
     (and downstream re-analysis) can avoid re-extracting.
  3. Runs a linear probe against every physical label in CATALOG_COLUMNS
     plus a derived ``g-r`` colour. The probe applies the published
     preprocessing recipe (``z>0`` filter and ``[0,4]`` clip for
     redshift; per-property 1–99 percent quantile clip; StandardScaler
     fit on the train fold; 10 random 80/20 splits; mean ± std R²).
  4. Builds a cosine kNN graph and a 2-D UMAP for the embedding.
  5. Uploads ``probe.parquet``, ``neighbours.parquet``, ``umap.parquet``
     to ``$PU_REGRESS_RESULTS_REPO`` under ``done/<tag>/`` in a single
     ``upload_folder`` commit.

Workers atomically claim tuples via ``running/<tag>.running`` markers,
so any number of workers across any number of clusters can divide the
work with no other coordination. Idempotent and restart-safe; stale
claims (>1 h) are auto-released.

Required environment
--------------------
    HF_TOKEN                    HF token with write access to the repos below
    PU_REGRESS_RESULTS_REPO     HF dataset id for results, e.g. <owner>/pu-regress-results
    PU_REGRESS_OUT              persistent local output dir
    PU_REGRESS_LOCKS            persistent local lock dir

Optional environment
--------------------
    PU_EMB_REPO                 if set, also upload .npy embeddings here
    PU_REGRESS_DATASET          default Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2
    PU_REGRESS_DLCACHE          default /tmp/pu_regress_dl/
    PU_REGRESS_N_USE            default 45000
    PU_REGRESS_BATCH_SIZE       default 16
    PU_REGRESS_N_RUNS           default 10; random 80/20 splits per probe
    PU_REGRESS_TEST_SIZE        default 2000; held-out galaxies per split
    PU_REGRESS_KNN_K            default 10; k for the per-tuple kNN graph
    PU_REGRESS_STALE_S          default 3600 (1 h)
    PU_REGRESS_TARGET           e.g. "hsc/vit_base" — single-tuple mode
"""
from __future__ import annotations

import gc
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError
from torch.utils.data import DataLoader

from pu.models import get_adapter
from pu.pu_datasets.cosmosweb import CATALOG_COLUMNS

# ---------------------------------------------------------------------------
# Config from environment (no defaults that point at a specific account / repo)
# ---------------------------------------------------------------------------
HF_TOKEN  = os.environ["HF_TOKEN"]
COORD_REPO = os.environ["PU_REGRESS_RESULTS_REPO"]
OUT_DIR    = Path(os.environ["PU_REGRESS_OUT"])
LOCK_DIR   = Path(os.environ["PU_REGRESS_LOCKS"])
DATASET    = os.environ.get("PU_REGRESS_DATASET",
                            "Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2")
DLCACHE    = Path(os.environ.get("PU_REGRESS_DLCACHE", "/tmp/pu_regress_dl"))
N_USE      = int(os.environ.get("PU_REGRESS_N_USE", "45000"))
BATCH_SIZE = int(os.environ.get("PU_REGRESS_BATCH_SIZE", "16"))
N_RUNS     = int(os.environ.get("PU_REGRESS_N_RUNS", "10"))
TEST_SIZE  = int(os.environ.get("PU_REGRESS_TEST_SIZE", "2000"))
KNN_K      = int(os.environ.get("PU_REGRESS_KNN_K", "10"))
EMB_REPO   = os.environ.get("PU_EMB_REPO", "")  # optional; if set, .npy is uploaded here
STALE_S    = int(os.environ.get("PU_REGRESS_STALE_S", "3600"))
TARGET     = os.environ.get("PU_REGRESS_TARGET", "")  # single-tuple mode

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOCK_DIR.mkdir(parents=True, exist_ok=True)
DLCACHE.mkdir(parents=True, exist_ok=True)

api = HfApi(token=HF_TOKEN)


def log(*a, **kw):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}]", *a, flush=True, **kw)


# ---------------------------------------------------------------------------
# Model grid: (alias, size, hf_model_name, n_params_M).
# Same set as Ashod's run_intramodal.py; kept here so this script is
# self-contained and the grid can be filtered via CLI / env without
# touching upstream data.
# ---------------------------------------------------------------------------
MODEL_GRID = [
    # (alias, size, hf_id, params_M)
    ("vit",        "base",       "google/vit-base-patch16-224-in21k",     86),
    ("vit",        "large",      "google/vit-large-patch16-224-in21k",    307),
    ("vit",        "huge",       "google/vit-huge-patch14-224-in21k",     632),
    ("clip",       "base",       "openai/clip-vit-base-patch16",          86),
    ("clip",       "large",      "openai/clip-vit-large-patch14",         307),
    ("dinov3",     "vits16",     "facebook/dinov3-vits16-pretrain-lvd1689m",   22),
    ("dinov3",     "vits16plus", "facebook/dinov3-vits16plus-pretrain-lvd1689m", 26),
    ("dinov3",     "vitb16",     "facebook/dinov3-vitb16-pretrain-lvd1689m",    86),
    ("dinov3",     "vitl16",     "facebook/dinov3-vitl16-pretrain-lvd1689m",    307),
    ("dinov3",     "vith16plus", "facebook/dinov3-vith16plus-pretrain-lvd1689m", 650),
    ("dinov3",     "vit7b16",    "facebook/dinov3-vit7b16-pretrain-lvd1689m",   7000),
    ("convnext",   "nano",       "facebook/convnextv2-nano-22k-224",      16),
    ("convnext",   "tiny",       "facebook/convnextv2-tiny-22k-224",      29),
    ("convnext",   "base",       "facebook/convnextv2-base-22k-224",      89),
    ("convnext",   "large",      "facebook/convnextv2-large-22k-224",     198),
    ("ijepa",      "huge",       "facebook/ijepa_vith14_22k",             632),
    ("ijepa",      "giant",      "facebook/ijepa_vitg16_22k",             1011),
    ("vjepa",      "large",      "facebook/vjepa2-vitl-fpc64-256",         307),
    ("vjepa",      "huge",       "facebook/vjepa2-vith-fpc64-256",         632),
    ("vjepa",      "giant",      "facebook/vjepa2-vitg-fpc64-256",         1011),
    ("astropt",    "015M",       "Smith42/astroPT_v2.0",                  15),
    ("astropt",    "095M",       "Smith42/astroPT_v2.0",                  95),
    ("astropt",    "850M",       "Smith42/astroPT_v2.0",                  850),
    ("vit-mae",    "base",       "facebook/vit-mae-base",                 86),
    ("vit-mae",    "large",      "facebook/vit-mae-large",                307),
    ("vit-mae",    "huge",       "facebook/vit-mae-huge",                 632),
    ("paligemma",  "3b",         "google/paligemma2-3b-mix-224",          3000),
    ("paligemma",  "10b",        "google/paligemma2-10b-mix-224",         10000),
    ("paligemma",  "28b",        "google/paligemma2-28b-mix-224",         28000),
    ("llava_15",   "7b",         "llava-hf/llava-1.5-7b-hf",              7000),
    ("llava_15",   "13b",        "llava-hf/llava-1.5-13b-hf",             13000),
]
MODALITIES = ("hsc", "jwst")


def tag_for(modality: str, alias: str, size: str) -> str:
    return f"{modality}__{alias}_{size}"


# ---------------------------------------------------------------------------
# HF claim / release coordination
# ---------------------------------------------------------------------------
def hf_list_state() -> tuple[set[str], dict[str, datetime]]:
    """Return (done_tags, running_tag -> last_modified).

    A tag is 'done' iff both done/<tag>/probe.parquet and
    done/<tag>/neighbours.parquet are present on the dataset (legacy single-
    file done/<tag>.parquet entries are also accepted for back-compat with
    the pre-pairs layout).
    """
    files = api.list_repo_files(COORD_REPO, repo_type="dataset")
    done_legacy = {f.replace("done/", "").replace(".parquet", "")
                   for f in files
                   if f.startswith("done/") and f.endswith(".parquet")
                   and "/" not in f.replace("done/", "")}
    # New folder layout: done/<tag>/{probe,neighbours}.parquet
    by_tag: dict[str, set[str]] = {}
    for f in files:
        if f.startswith("done/") and "/" in f[len("done/"):] and f.endswith(".parquet"):
            tag = f[len("done/"):].split("/", 1)[0]
            leaf = f[len("done/"):].split("/", 1)[1]
            by_tag.setdefault(tag, set()).add(leaf)
    done_new = {tag for tag, leaves in by_tag.items()
                if {"probe.parquet", "neighbours.parquet"}.issubset(leaves)}
    done = done_legacy | done_new
    running_tags = {f.replace("running/", "").replace(".running", "")
                    for f in files
                    if f.startswith("running/") and f.endswith(".running")}
    # Estimate ages from commit history.
    ages: dict[str, datetime] = {}
    try:
        commits = api.list_repo_commits(COORD_REPO, repo_type="dataset")
        for c in commits:
            msg = (c.message or "")
            if msg.startswith("claim ") or msg.startswith("release ") or msg.startswith("reap "):
                parts = msg.split(" ", 1)
                if len(parts) == 2:
                    t = parts[1].strip()
                    if t in running_tags and t not in ages:
                        ages[t] = c.created_at
    except HfHubHTTPError:
        pass
    return done, ages


def reap_stale(running_ages: dict[str, datetime]) -> None:
    now = datetime.now(timezone.utc)
    for tag, last in running_ages.items():
        age_s = (now - last).total_seconds()
        if age_s > STALE_S:
            try:
                api.delete_file(
                    path_in_repo=f"running/{tag}.running",
                    repo_id=COORD_REPO, repo_type="dataset",
                    commit_message=f"reap {tag}",
                )
                log(f"reaped stale {tag} (age={age_s/60:.1f}m)")
            except (EntryNotFoundError, HfHubHTTPError):
                pass


def try_claim(tag: str) -> bool:
    """Atomic-ish claim via local O_EXCL lockfile + HF marker upload."""
    local_lock = LOCK_DIR / f"{tag}.lock"
    try:
        fd = os.open(local_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, f"{os.uname().nodename} pid={os.getpid()}\n".encode())
        os.close(fd)
    except FileExistsError:
        return False
    try:
        api.upload_file(
            path_or_fileobj=f"{os.uname().nodename} pid={os.getpid()}\n".encode(),
            path_in_repo=f"running/{tag}.running",
            repo_id=COORD_REPO, repo_type="dataset",
            commit_message=f"claim {tag}",
        )
        return True
    except Exception as e:
        log(f"  [{tag}] HF claim failed: {type(e).__name__}: {e}")
        try:
            local_lock.unlink()
        except FileNotFoundError:
            pass
        return False


def release_claim(tag: str) -> None:
    try:
        api.delete_file(path_in_repo=f"running/{tag}.running",
                        repo_id=COORD_REPO, repo_type="dataset",
                        commit_message=f"release {tag}")
    except (EntryNotFoundError, HfHubHTTPError):
        pass
    try:
        (LOCK_DIR / f"{tag}.lock").unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Catalog pass: collect physics labels once, in row order.
# ---------------------------------------------------------------------------
def collect_catalog(n_use: int) -> dict[str, np.ndarray]:
    log(f"catalog pass over {DATASET} (limit {n_use})")
    ds = load_dataset(DATASET, split="train", streaming=True)
    cols = list(CATALOG_COLUMNS.values())
    raw = {c: [] for c in cols}
    for i, row in enumerate(ds):
        for c in cols:
            raw[c].append(row[c])
        if i + 1 >= n_use:
            break
    out = {param: np.array(raw[col], dtype=np.float32)
           for param, col in CATALOG_COLUMNS.items()}
    out["g-r"] = out["mag_g"] - out["mag_r"]
    log(f"catalog: {len(out['redshift'])} rows, {len(out)} fields")
    return out


# ---------------------------------------------------------------------------
# Embedding extraction (one (model, size, modality) at a time)
# ---------------------------------------------------------------------------
def make_preprocessor(adapter, modality: str):
    """Same logic as Ashod's run_intramodal.py preprocessor builder, condensed."""
    from PIL import Image
    image_col = f"{modality}_images"
    target = (224, 224)

    def upsample(img):
        return img.resize(target, Image.Resampling.BILINEAR)

    if hasattr(adapter, "processor") and adapter.processor is not None:
        proc = adapter.processor

        def hf_wrapper(example):
            img = upsample(example[image_col])
            if adapter.alias == "clip":
                p = proc(images=img, return_tensors="pt")
            elif hasattr(adapter, "_PROMPTS"):
                prompt = adapter._PROMPTS.get(adapter.alias, "<image> ")
                p = proc(text=prompt, images=img, return_tensors="pt")
            else:
                p = proc(img, return_tensors="pt")
            if "pixel_values" in p:
                return {modality: p["pixel_values"].squeeze(0)}
            if "pixel_values_videos" in p:
                return {modality: p["pixel_values_videos"]
                        .repeat(1, 16, 1, 1, 1).squeeze(0)}
            raise KeyError("processor output missing pixel_values")

        return hf_wrapper

    if adapter.alias == "sam2":
        sam2_transforms = adapter.predictor._transforms

        def sam2_wrapper(example):
            img = upsample(example[image_col])
            arr = np.asarray(img)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            return {modality: sam2_transforms(arr)}

        return sam2_wrapper

    if adapter.alias == "astropt":
        from astropt.local_datasets import GalaxyImageDataset
        from torchvision import transforms

        def normalise(x):
            std, mean = torch.std_mean(x, dim=1, keepdim=True)
            return (x - mean) / (std + 1e-8)

        galproc = GalaxyImageDataset(
            None, spiral=True,
            transform={"images": transforms.Compose([transforms.Lambda(normalise)])},
            modality_registry=adapter.model.modality_registry,
        )

        def astropt_wrapper(example):
            img = upsample(example[image_col])
            arr = np.asarray(img, dtype=np.float32)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            arr = arr.swapaxes(0, 2)
            t = torch.from_numpy(arr).to(torch.float)
            im = galproc.process_galaxy(t).to(torch.float)
            return {f"{modality}_images": im,
                    f"{modality}_positions": torch.arange(0, len(im), dtype=torch.long)}

        return astropt_wrapper

    raise ValueError(
        f"no preprocessor for adapter '{adapter.alias}' "
        "(needs .processor, or be sam2 / astropt)"
    )


def extract_embeddings(alias: str, size: str, hf_id: str,
                       modality: str, n_use: int) -> np.ndarray:
    """Extract a (n, d) embedding matrix for n_use galaxies.

    Streams cosmosweb directly without going through the dataset adapter,
    because the adapter's prepare() signature doesn't accept the extra
    flags we need (telescope, n_galaxies, remove_image_col) and its blanket
    `.remove_columns(image_cols)` would silently delete AstroPT's
    in-place-overwritten output column.
    """
    cache = DLCACHE / f"emb_{modality}_{alias}_{size}_{n_use}.npy"
    if cache.exists():
        log(f"  using cached {cache}")
        return np.load(cache)

    adapter_cls = get_adapter(alias)
    adapter = adapter_cls(hf_id, size, alias=alias)
    adapter.load()

    proc_fn = make_preprocessor(adapter, modality)
    image_col = f"{modality}_images"

    # Stream cosmosweb. select_columns first so we don't pay decoder cost on
    # the other side's image column. filter is a no-op (kept for parity with
    # the adapter API). map applies the model's preprocessor.
    ds = (
        load_dataset(DATASET, split="train", streaming=True)
        .select_columns([image_col])
        .map(proc_fn)
    )

    # AstroPT's preprocessor *overwrites* `hsc_images` in-place with a
    # tensor and adds `hsc_positions`. For all other adapters, the
    # preprocessor outputs a clean `<modality>` key (no `_images` suffix).
    # Remove any leftover columns that aren't part of the embed_for_mode
    # input contract.
    if alias == "astropt":
        ds = ds.select_columns([f"{modality}_images", f"{modality}_positions"])
    else:
        ds = ds.remove_columns([image_col]).select_columns([modality])

    if hasattr(ds, "with_format"):
        ds = ds.with_format("torch")

    dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0)
    chunks: list[np.ndarray] = []
    for batch in dl:
        emb = adapter.embed_for_mode(batch, modality).float().cpu().numpy()
        chunks.append(emb)
        if sum(len(c) for c in chunks) >= n_use:
            break
    Z = np.concatenate(chunks, axis=0)[:n_use]
    np.save(cache, Z)
    log(f"  extracted {Z.shape}, cached -> {cache}")

    del adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return Z


# ---------------------------------------------------------------------------
# Per-tuple regression
# ---------------------------------------------------------------------------
def _prep_property(name: str, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the published per-property preprocessing.

    Returns (mask, y_clipped). The mask selects catalog rows that survive
    the property-specific filter (used to also subset embeddings). y_clipped
    is then trimmed to its own 1–99% quantiles to suppress outliers that
    would otherwise blow up unregularised OLS.
    """
    valid = np.isfinite(y)
    if name == "redshift":
        valid &= (y > 0)
    if not valid.any():
        return valid, y
    y_v = y[valid].astype(np.float32, copy=True)
    if name == "redshift":
        y_v = np.clip(y_v, 0, 4)
    lo, hi = np.quantile(y_v, [0.01, 0.99])
    y_v = np.clip(y_v, lo, hi)
    out = y.astype(np.float32, copy=True)
    out[valid] = y_v
    return valid, out


def _probe_published(Z: np.ndarray, y: np.ndarray, mask: np.ndarray
                     ) -> tuple[float, float]:
    """Linear probe with the published recipe: StandardScaler embeddings on
    each train fold, fit OLS, score on test fold. Repeats N_RUNS random
    80/20 splits and returns (mean R², std R²).

    Implementation: torch.linalg.lstsq on GPU when CUDA is available, falls
    back to sklearn LinearRegression on CPU otherwise.
    """
    Zv = Z[mask].astype(np.float32, copy=False)
    yv = y[mask].astype(np.float32, copy=False)
    if len(Zv) < TEST_SIZE + 100:
        return float("nan"), float("nan")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    scores = []
    for seed in range(N_RUNS):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(Zv))
        test_idx, train_idx = order[:TEST_SIZE], order[TEST_SIZE:]
        Xtr_np = Zv[train_idx]; ytr_np = yv[train_idx]
        Xte_np = Zv[test_idx];  yte_np = yv[test_idx]

        mu = Xtr_np.mean(axis=0)
        sd = Xtr_np.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        Xtr_np = (Xtr_np - mu) / sd
        Xte_np = (Xte_np - mu) / sd

        if use_cuda:
            Xtr = torch.from_numpy(Xtr_np).to(device)
            Xte = torch.from_numpy(Xte_np).to(device)
            ytr = torch.from_numpy(ytr_np).to(device)
            yte = torch.from_numpy(yte_np).to(device)
            ones_tr = torch.ones(Xtr.shape[0], 1, device=device)
            ones_te = torch.ones(Xte.shape[0], 1, device=device)
            Xtr_b = torch.cat([Xtr, ones_tr], dim=1)
            Xte_b = torch.cat([Xte, ones_te], dim=1)
            sol = torch.linalg.lstsq(Xtr_b, ytr.unsqueeze(1)).solution.squeeze(1)
            pred = Xte_b @ sol
            ss_res = float(torch.sum((yte - pred) ** 2))
            ss_tot = float(torch.sum((yte - yte.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        else:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            m = LinearRegression()
            m.fit(Xtr_np, ytr_np)
            r2 = r2_score(yte_np, m.predict(Xte_np))
        scores.append(float(r2))

    return float(np.mean(scores)), float(np.std(scores))


def _build_knn(Z: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-row kNN indices + cosine distances (excluding self).
    Returns (idx, dist) where idx is (n, k) int32 and dist is (n, k) float32.
    Uses faiss if available, else sklearn."""
    Zc = np.ascontiguousarray(Z, dtype=np.float32)
    Zc /= (np.linalg.norm(Zc, axis=1, keepdims=True) + 1e-12)
    try:
        import faiss
        d = Zc.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(Zc)
        sim, idx = index.search(Zc, k + 1)
        # cosine distance = 1 - inner-product similarity (since vectors are unit-norm)
        dist = (1.0 - sim).astype(np.float32)
        return idx[:, 1:].astype(np.int32), dist[:, 1:]
    except Exception:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Zc)
        dist, idx = nn.kneighbors(return_distance=True)
        return idx[:, 1:].astype(np.int32), dist[:, 1:].astype(np.float32)


def _umap_2d(Z: np.ndarray, idx: np.ndarray, dist: np.ndarray,
             k: int) -> np.ndarray | None:
    """Run UMAP using a precomputed kNN graph. Returns (n, 2) float32 or
    None if umap-learn isn't installed (in which case the worker still
    succeeds, just no umap.parquet is uploaded)."""
    try:
        import umap
    except Exception:
        return None
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    emb2d = umap.UMAP(
        n_neighbors=k, metric="cosine",
        precomputed_knn=(idx, dist, None),
        random_state=0, n_epochs=200,
    ).fit_transform(Zn)
    return emb2d.astype(np.float32)


def regress_tuple(alias: str, size: str, hf_id: str, params_M: int,
                  modality: str, catalog: dict[str, np.ndarray]
                  ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None]:
    """Returns (probe_df, neighbors_df, umap_df).

    probe_df has one row per physics property: R² mean + std + permutation p-value.
    neighbors_df has one row per galaxy: the (k,) integer indices of its k
    nearest neighbours in this tuple's embedding space.
    umap_df has one row per galaxy: 2-D UMAP coords (or None if umap-learn
    isn't available).
    """
    Z = extract_embeddings(alias, size, hf_id, modality, N_USE)

    log(f"  building cosine kNN graph (k={KNN_K})")
    nn_idx, nn_dist = _build_knn(Z, KNN_K)
    nn_df = pl.DataFrame({
        "galaxy_idx":  np.arange(len(Z), dtype=np.int32),
        "neighbours":  pl.Series(
            "neighbours",
            nn_idx,
            dtype=pl.Array(pl.Int32, KNN_K),
        ),
    })

    log("  computing UMAP-2D (precomputed kNN)")
    emb2d = _umap_2d(Z, nn_idx, nn_dist, KNN_K)
    umap_df = None
    if emb2d is not None:
        umap_df = pl.DataFrame({
            "galaxy_idx": np.arange(len(Z), dtype=np.int32),
            "umap_x":     emb2d[:, 0],
            "umap_y":     emb2d[:, 1],
        })

    rows = []
    for prop, y_raw in catalog.items():
        n = min(len(Z), len(y_raw))
        mask, y = _prep_property(prop, y_raw[:n])
        try:
            r2_mean, r2_std = _probe_published(Z[:n], y, mask)
        except Exception as e:
            log(f"    {prop}: probe failed: {type(e).__name__}: {e}")
            r2_mean, r2_std = float("nan"), float("nan")
        log(f"    R²({prop}) = {r2_mean:.4f} ± {r2_std:.4f}  "
            f"(n_valid={int(mask.sum())})")
        rows.append({
            "modality": modality,
            "model_alias": alias,
            "model_size": size,
            "model_hf_id": hf_id,
            "params_M": int(params_M),
            "property": prop,
            "r2_mean": r2_mean,
            "r2_std":  r2_std,
            "n_valid": int(mask.sum()),
            "n": int(n),
            "d": int(Z.shape[1]),
            "n_runs": N_RUNS,
            "test_size": TEST_SIZE,
        })
    return pl.DataFrame(rows), nn_df, umap_df


# ---------------------------------------------------------------------------
# Top-level loop
# ---------------------------------------------------------------------------
def main() -> int:
    log(f"worker on {os.uname().nodename} pid={os.getpid()}")
    log(f"COORD_REPO={COORD_REPO}  DATASET={DATASET}  N_USE={N_USE}")

    # Build the (modality, alias, size) work list.
    work: list[tuple[str, str, str, str, int]] = []
    for modality in MODALITIES:
        for alias, size, hf_id, params_M in MODEL_GRID:
            work.append((modality, alias, size, hf_id, params_M))

    if TARGET:
        # Single-tuple mode for partition / re-runs.
        try:
            t_modality, t_alias_size = TARGET.split("/", 1)
            t_alias, t_size = t_alias_size.rsplit("_", 1)
        except ValueError:
            log(f"[fatal] PU_REGRESS_TARGET must be 'modality/alias_size', got {TARGET!r}")
            return 1
        work = [w for w in work if w[0] == t_modality and w[1] == t_alias and w[2] == t_size]
        if not work:
            log(f"[fatal] no MODEL_GRID entry matches {TARGET}")
            return 1

    rng = random.Random((os.getpid() << 16) ^ int(time.time()))
    rng.shuffle(work)
    log(f"work queue: {len(work)} (modality, model, size) tuples")

    done_tags, running_ages = hf_list_state()
    reap_stale(running_ages)

    catalog = None  # lazily — only if we actually claim something

    for modality, alias, size, hf_id, params_M in work:
        tag = tag_for(modality, alias, size)
        if tag in done_tags:
            continue
        if tag in running_ages:
            continue
        if not try_claim(tag):
            continue

        log(f"[claim] {tag} ({hf_id}, params_M={params_M})")
        try:
            if catalog is None:
                catalog = collect_catalog(N_USE)
            probe_df, nn_df, umap_df = regress_tuple(
                alias, size, hf_id, params_M, modality, catalog,
            )
            # Stage parquets into a per-tag folder, then upload as one
            # commit via upload_folder (cuts HF commit volume vs uploading
            # files separately).
            tag_dir = OUT_DIR / tag
            tag_dir.mkdir(exist_ok=True)
            probe_path = tag_dir / "probe.parquet"
            nn_path    = tag_dir / "neighbours.parquet"
            probe_df.write_parquet(probe_path, compression="zstd")
            nn_df.write_parquet(nn_path, compression="zstd")
            if umap_df is not None:
                umap_path = tag_dir / "umap.parquet"
                umap_df.write_parquet(umap_path, compression="zstd")
            import shutil
            staging = OUT_DIR / f".upload_{tag}_{os.getpid()}"
            (staging / f"done/{tag}").mkdir(parents=True, exist_ok=True)
            shutil.copy2(probe_path, staging / f"done/{tag}/probe.parquet")
            shutil.copy2(nn_path,    staging / f"done/{tag}/neighbours.parquet")
            if umap_df is not None:
                shutil.copy2(umap_path, staging / f"done/{tag}/umap.parquet")
            try:
                api.upload_folder(
                    folder_path=str(staging),
                    repo_id=COORD_REPO, repo_type="dataset",
                    commit_message=f"done {tag}",
                )
            finally:
                shutil.rmtree(staging, ignore_errors=True)
            log(f"[done] {tag} -> done/{tag}/"
                f"{{probe,neighbours{'' if umap_df is None else ',umap'}}}.parquet")

            # Optional: ship the cached .npy to the embeddings repo so
            # later analyses (e.g. re-probing with a different recipe)
            # don't need a GPU. Each .npy is ~250–600 MB.
            if EMB_REPO:
                emb_npy = DLCACHE / f"emb_{modality}_{alias}_{size}_{N_USE}.npy"
                if emb_npy.exists():
                    try:
                        api.upload_file(
                            path_or_fileobj=str(emb_npy),
                            path_in_repo=emb_npy.name,
                            repo_id=EMB_REPO,
                            repo_type="dataset",
                            commit_message=f"emb {tag}",
                        )
                        log(f"[emb] {emb_npy.name} -> {EMB_REPO}")
                    except Exception as e:
                        log(f"[emb] upload failed for {emb_npy.name}: "
                            f"{type(e).__name__}: {e}")
        except KeyboardInterrupt:
            release_claim(tag)
            raise
        except Exception:
            log(f"[ERROR] {tag}\n{traceback.format_exc()}")
        finally:
            release_claim(tag)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    log("queue exhausted, exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
