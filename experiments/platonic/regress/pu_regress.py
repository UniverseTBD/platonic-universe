#!/usr/bin/env python3
"""Cooperative physics-regression worker.

For each (model, size, modality) tuple, extract embeddings on the
COSMOS-Web sample and run a 5-fold linear probe against every physical
label in CATALOG_COLUMNS, plus a derived `g-r` colour. Output is one
small parquet per tuple uploaded to a coordination dataset on Hugging
Face. Workers atomically claim tuples via `running/<tag>.running`
markers exactly as pu_solve.py does, so any number of workers on any
number of clusters can divide the work without speaking to each other.

Idempotent. Restart-safe. Stale claims (>1h) auto-released.

Required environment:
    HF_TOKEN                    no default; required
    PU_REGRESS_RESULTS_REPO     no default; HF dataset id, e.g. <owner>/pu-regress-results
    PU_REGRESS_OUT              no default; persistent local output dir
    PU_REGRESS_LOCKS            no default; persistent local lock dir

Optional environment:
    PU_REGRESS_DATASET          default Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2
    PU_REGRESS_DLCACHE          default /tmp/pu_regress_dl/
    PU_REGRESS_N_USE            default 45000
    PU_REGRESS_BATCH_SIZE       default 16
    PU_REGRESS_CV_FOLDS         default 5
    PU_REGRESS_PCA_COMPONENTS   default 0 (disabled)
    PU_REGRESS_STALE_S          default 3600 (1 h)
    PU_REGRESS_TARGET           e.g. "hsc/dino_giant" — single-tuple mode
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

from pu.metrics.physics import linear_probe
from pu.models import get_adapter
from pu.pu_datasets import get_dataset_adapter
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
CV_FOLDS   = int(os.environ.get("PU_REGRESS_CV_FOLDS", "5"))
PCA_K      = int(os.environ.get("PU_REGRESS_PCA_COMPONENTS", "0")) or None
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
    """Return (done_tags, running_tag -> last_modified)."""
    files = api.list_repo_files(COORD_REPO, repo_type="dataset")
    done = {f.replace("done/", "").replace(".parquet", "")
            for f in files if f.startswith("done/") and f.endswith(".parquet")}
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
    """Extract a (n, d) embedding matrix for n_use galaxies. Cached on local
    disk; identical-key second calls are instant."""
    cache = DLCACHE / f"emb_{modality}_{alias}_{size}_{n_use}.npy"
    if cache.exists():
        log(f"  using cached {cache}")
        return np.load(cache)

    adapter_cls = get_adapter(alias)
    adapter = adapter_cls(hf_id, size, alias=alias)
    adapter.load()

    proc_fn = make_preprocessor(adapter, modality)
    ds_adapter = get_dataset_adapter("cosmosweb")(DATASET, modality)
    ds_adapter.load()
    ds = ds_adapter.prepare(
        processor=proc_fn,
        modes=[modality],
        filterfun=lambda row: True,
        telescope=modality,
        n_galaxies=n_use,
        remove_image_col=(alias != "astropt"),
    )
    if alias == "astropt":
        ds = ds.select_columns([f"{modality}_images", f"{modality}_positions"])
    else:
        ds = ds.select_columns([modality])

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
def regress_tuple(alias: str, size: str, hf_id: str, params_M: int,
                  modality: str, catalog: dict[str, np.ndarray]) -> pl.DataFrame:
    Z = extract_embeddings(alias, size, hf_id, modality, N_USE)
    rows = []
    for prop, y in catalog.items():
        n = min(len(Z), len(y))
        try:
            r2 = linear_probe(Z[:n], y[:n], cv=CV_FOLDS, pca_components=PCA_K)
        except Exception as e:
            log(f"    {prop}: probe failed: {type(e).__name__}: {e}")
            r2 = float("nan")
        log(f"    R²({prop}) = {r2:.4f}")
        rows.append({
            "modality": modality,
            "model_alias": alias,
            "model_size": size,
            "model_hf_id": hf_id,
            "params_M": int(params_M),
            "property": prop,
            "r2_mean": float(r2),
            "n": int(n),
            "d": int(Z.shape[1]),
            "cv_folds": CV_FOLDS,
            "pca_k": PCA_K or 0,
        })
    return pl.DataFrame(rows)


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
            df = regress_tuple(alias, size, hf_id, params_M, modality, catalog)
            tmp = OUT_DIR / f"{tag}.parquet.tmp.{os.getpid()}"
            final = OUT_DIR / f"{tag}.parquet"
            df.write_parquet(tmp, compression="zstd")
            os.rename(tmp, final)
            api.upload_file(
                path_or_fileobj=str(final),
                path_in_repo=f"done/{tag}.parquet",
                repo_id=COORD_REPO, repo_type="dataset",
                commit_message=f"done {tag}",
            )
            log(f"[done] {tag} -> done/{tag}.parquet")
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
