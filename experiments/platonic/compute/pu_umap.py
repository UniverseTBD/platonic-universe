#!/usr/bin/env python3
"""Cross-cluster cooperative worker for UMAP + kNN-purity on candidate
(model, survey, block) triples produced by pu_solve.py.

Reads candidates.parquet + labels_*.parquet from static/ on the
PU_UMAP_RESULTS_REPO HF dataset. For each (model, survey) tuple it
atomically claims, downloads the matching embedding parquet from
<owner>/platonic-embeddings, projects each candidate column, computes
faiss-cpu kNN, kNN-purity vs the survey's physical label, UMAP via
the precomputed kNN graph, and uploads one tiny umap_*.parquet per
candidate block to done/<tag>/ on the coordination dataset.

Idempotent. Restart-safe. Stale claims (>1h) auto-released.
"""
from __future__ import annotations

import gc
import os
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone

import faiss
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import umap
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HF_TOKEN          = os.environ["HF_TOKEN"]
COORD_REPO        = os.environ["PU_UMAP_RESULTS_REPO"]      # <owner>/pu-umap-results
EMBED_REPO        = os.environ["PU_UMAP_EMBED_REPO"]
OUT_DIR           = os.environ["PU_UMAP_OUT"]               # persistent
LOCK_DIR          = os.environ["PU_UMAP_LOCKS"]             # persistent
DLCACHE           = os.environ.get("PU_UMAP_DLCACHE", "/tmp/pu_umap_dl")
STALE_SECONDS     = int(os.environ.get("PU_UMAP_STALE_SECONDS", str(60 * 60)))
KNN_K             = int(os.environ.get("PU_UMAP_KNN_K", "50"))
UMAP_EPOCHS       = int(os.environ.get("PU_UMAP_EPOCHS",  "200"))
HNSW_M            = int(os.environ.get("PU_UMAP_HNSW_M",  "32"))
HNSW_EF_SEARCH    = int(os.environ.get("PU_UMAP_HNSW_EF", "64"))
HNSW_EF_CONSTRUCT = int(os.environ.get("PU_UMAP_HNSW_EF_CON", "200"))

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOCK_DIR, exist_ok=True)
os.makedirs(DLCACHE, exist_ok=True)

api = HfApi(token=HF_TOKEN)


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Static inputs (candidates + labels) — pulled once at startup
# ---------------------------------------------------------------------------
def fetch_static() -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    log("downloading static/candidates.parquet")
    cand_path = hf_hub_download(COORD_REPO, "static/candidates.parquet",
                                repo_type="dataset", token=HF_TOKEN,
                                local_dir=DLCACHE)
    cand = pl.read_parquet(cand_path)

    labels: dict[str, pl.DataFrame] = {}
    for survey in ["desi", "jwst", "legacysurvey", "sdss"]:
        log(f"downloading static/labels_{survey}.parquet")
        p = hf_hub_download(COORD_REPO, f"static/labels_{survey}.parquet",
                            repo_type="dataset", token=HF_TOKEN,
                            local_dir=DLCACHE)
        labels[survey] = pl.read_parquet(p)
    return cand, labels


def survey_label(labels: dict[str, pl.DataFrame], survey: str) -> tuple[np.ndarray, str, np.ndarray]:
    """Return (label_array, label_name, hsc_object_id) for a survey, dropping
    rows with bad redshift flags where applicable. Length is the survey's row
    count aligned with the embedding parquet (row-order alignment from upstream)."""
    df = labels[survey]
    obj_id = df["hsc_object_id"].to_numpy()
    if survey == "desi":
        z = df["Z"].to_numpy()
        bad = df["ZWARN"].to_numpy() if "ZWARN" in df.columns else np.zeros(len(df), bool)
        z = z.astype(np.float32)
        z[bad.astype(bool)] = np.nan
        return z, "z_spec", obj_id
    if survey == "sdss":
        z = df["Z"].to_numpy().astype(np.float32)
        bad = df["ZWARNING"].to_numpy() if "ZWARNING" in df.columns else np.zeros(len(df), bool)
        z[bad.astype(bool)] = np.nan
        return z, "z_spec", obj_id
    if survey == "jwst":
        return df["i_cmodel_mag"].to_numpy().astype(np.float32), "i_cmodel_mag", obj_id
    if survey == "legacysurvey":
        # g - r color via fluxes
        g = df["FLUX_G"].to_numpy().astype(np.float32)
        r = df["FLUX_R"].to_numpy().astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            color = -2.5 * np.log10(np.where((g > 0) & (r > 0), g / r, np.nan))
        return color, "g_minus_r", obj_id
    raise ValueError(f"unknown survey {survey}")


# ---------------------------------------------------------------------------
# Claim / release coordination
# ---------------------------------------------------------------------------
def claim_dir_url(tag: str) -> str:
    return f"running/{tag}.running"


def done_prefix(tag: str) -> str:
    return f"done/{tag}/"


def already_done(tag: str, n_expected: int) -> bool:
    """Return True if done/<tag>/ already has ≥ n_expected umap_*.parquet files."""
    try:
        files = api.list_repo_files(COORD_REPO, repo_type="dataset")
    except HfHubHTTPError:
        return False
    n = sum(1 for f in files if f.startswith(done_prefix(tag)) and f.endswith(".parquet"))
    return n >= n_expected


def list_running() -> set[str]:
    files = api.list_repo_files(COORD_REPO, repo_type="dataset")
    return {f.split("/", 1)[1].rsplit(".running", 1)[0]
            for f in files if f.startswith("running/") and f.endswith(".running")}


def claim_age_seconds(tag: str) -> float | None:
    try:
        api.dataset_info(COORD_REPO, files_metadata=False)
    except HfHubHTTPError:
        return None
    # cheap-ish: get last modified by listing commit history of the marker
    try:
        commits = api.list_repo_commits(COORD_REPO, repo_type="dataset")
    except HfHubHTTPError:
        return None
    target = claim_dir_url(tag)
    for c in commits:
        if target in (c.message or ""):
            return (datetime.now(timezone.utc) - c.created_at).total_seconds()
    return None


def try_claim(tag: str) -> bool:
    """Atomic-ish claim via a local O_EXCL lockfile + HF marker upload.
    The local lockfile prevents two workers on the same node racing;
    the HF marker is the cross-cluster signal."""
    local_lock = os.path.join(LOCK_DIR, f"{tag}.lock")
    try:
        fd = os.open(local_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, f"{os.getpid()} {datetime.now(timezone.utc).isoformat()}\n".encode())
        os.close(fd)
    except FileExistsError:
        return False

    try:
        api.upload_file(
            path_or_fileobj=f"{os.uname().nodename} pid={os.getpid()}\n".encode(),
            path_in_repo=claim_dir_url(tag),
            repo_id=COORD_REPO,
            repo_type="dataset",
            commit_message=f"claim {tag}",
        )
        return True
    except Exception as e:
        log(f"  [{tag}] HF claim failed: {type(e).__name__}: {e}")
        try:
            os.remove(local_lock)
        except FileNotFoundError:
            pass
        return False


def release_claim(tag: str) -> None:
    try:
        api.delete_file(claim_dir_url(tag), repo_id=COORD_REPO, repo_type="dataset",
                        commit_message=f"release {tag}")
    except (EntryNotFoundError, HfHubHTTPError):
        pass
    try:
        os.remove(os.path.join(LOCK_DIR, f"{tag}.lock"))
    except FileNotFoundError:
        pass


def reap_stale_claims() -> None:
    """Delete running markers older than STALE_SECONDS."""
    try:
        commits = api.list_repo_commits(COORD_REPO, repo_type="dataset")
    except HfHubHTTPError:
        return
    age_by_tag: dict[str, float] = {}
    now = datetime.now(timezone.utc)
    for c in commits:
        msg = (c.message or "")
        if msg.startswith("claim ") or msg.startswith("release "):
            tag = msg.split(" ", 1)[1].strip()
            age_by_tag.setdefault(tag, (now - c.created_at).total_seconds())
    for tag in list_running():
        age = age_by_tag.get(tag, 0)
        if age > STALE_SECONDS:
            log(f"reaping stale claim {tag} (age={age/60:.1f}min)")
            try:
                api.delete_file(claim_dir_url(tag), repo_id=COORD_REPO, repo_type="dataset",
                                commit_message=f"reap stale {tag}")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Per-block compute
# ---------------------------------------------------------------------------
def faiss_knn(Z: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (idx, dist) for cosine kNN on L2-normalized Z. Index 0 is self."""
    Zc = np.ascontiguousarray(Z, dtype=np.float32)
    Zc /= (np.linalg.norm(Zc, axis=1, keepdims=True) + 1e-12)
    d = Zc.shape[1]
    index = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCT
    index.hnsw.efSearch = HNSW_EF_SEARCH
    index.add(Zc)
    sim, idx = index.search(Zc, k + 1)
    dist = 1.0 - sim
    return idx, dist


def knn_purity(idx: np.ndarray, label: np.ndarray, *, n_bins: int = 20
              ) -> tuple[float, np.ndarray]:
    """Return (global purity, per-row purity) for kNN-bin-matching.

    Per-row purity = fraction of this row's k neighbors that share its label
    bin. Global purity = mean of per-row purities (excluding rows with no
    valid neighbors). NaN where the row's own label is missing.
    """
    valid = ~np.isnan(label)
    if valid.sum() < n_bins * 5:
        return float("nan"), np.full(len(label), np.nan, dtype=np.float32)
    edges = np.nanquantile(label, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    binned = np.digitize(label, edges) - 1
    binned[~valid] = -1
    own = binned[:, None]
    nbrs = binned[idx[:, 1:]]
    mask = (nbrs >= 0) & (own >= 0)
    match = (nbrs == own) & mask
    denom = mask.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_row = np.where(denom > 0, match.sum(axis=1) / np.maximum(denom, 1), np.nan)
    per_row = per_row.astype(np.float32)
    global_purity = float(np.nanmean(per_row))
    return global_purity, per_row


def local_twoNN_dim(dist: np.ndarray) -> np.ndarray:
    """Per-point local intrinsic dim via Facco et al. 2017 TwoNN: for each
    point, the ratio mu = r2/r1 of the 2nd to 1st neighbor distance gives
    a single-sample MLE of intrinsic dim d via d = log(2)/log(mu).
    Per-point version (no aggregation) — color UMAP by this to see locally
    high-d vs low-d regions. Returns NaN where r1==0 or r2<=r1."""
    # dist[:, 0] is self (=0); dist[:, 1] is r1; dist[:, 2] is r2
    if dist.shape[1] < 3:
        return np.full(dist.shape[0], np.nan, dtype=np.float32)
    r1 = dist[:, 1].astype(np.float64)
    r2 = dist[:, 2].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.where(r1 > 0, r2 / r1, np.nan)
        d_local = np.where(mu > 1.0, np.log(2.0) / np.log(mu), np.nan)
    return d_local.astype(np.float32)


def process_block(model: str, survey: str, block_name: str,
                  emb_pq_path: str,
                  label: np.ndarray, label_name: str,
                  obj_id: np.ndarray) -> pl.DataFrame:
    log(f"  [{model}/{survey}/{block_name}] reading column")
    Z = np.stack(
        pq.read_table(emb_pq_path, columns=[block_name])
          .column(0).to_numpy(zero_copy_only=False)
    ).astype(np.float32)
    n, d = Z.shape
    log(f"    Z shape={Z.shape} ({Z.nbytes/1e9:.2f} GB)")

    # Row-order alignment between embedding parquet and labels parquet.
    # We take the first min(n_embed, n_labels) rows. This is correct when
    # the embedding extraction preserved row order from the source crossmatch
    # AND any dropped rows are at the tail. Hard-fail if the row delta is
    # >20% to avoid silently processing badly-misaligned data.
    n_lab = len(label)
    delta = abs(n - n_lab) / max(n, n_lab)
    if delta > 0.20:
        raise RuntimeError(
            f"row-count mismatch too large to trust row-order alignment: "
            f"embed n={n} vs label n={n_lab} (delta={delta:.0%}). "
            f"Refusing to silently misalign labels with embeddings."
        )
    if n != n_lab:
        log(f"    [WARN] row-count mismatch: embed n={n}, label n={n_lab} "
            f"(delta={delta:.1%}). Truncating to first {min(n, n_lab)} rows.")
    n_eff = min(n, n_lab, len(obj_id))
    Z = Z[:n_eff]
    label_eff = label[:n_eff]
    obj_id_eff = obj_id[:n_eff]

    log(f"    faiss kNN (k={KNN_K}, HNSW M={HNSW_M})")
    idx, dist = faiss_knn(Z, KNN_K)

    purity, purity_per_row = knn_purity(idx, label_eff)
    log(f"    kNN-purity vs {label_name}: {purity:.3f}")

    # Per-point local intrinsic dim (TwoNN, Facco et al. 2017).
    # Free given we already have the kNN distances.
    local_dim = local_twoNN_dim(dist)
    log(f"    local TwoNN dim: median={np.nanmedian(local_dim):.2f}  "
        f"p16={np.nanpercentile(local_dim, 16):.2f}  "
        f"p84={np.nanpercentile(local_dim, 84):.2f}")

    # PCA-50 reduction — kept on disk so any future low-dim method
    # (t-SNE, kPCA, 3D UMAP, diffusion maps, ...) is fast: ~50 MB per
    # candidate vs ~5 GB raw, and 50D preserves > 99% of structure for
    # a manifold whose intrinsic dim is < 20.
    log("    PCA-50")
    Zc = Z - Z.mean(axis=0, keepdims=True)
    from sklearn.decomposition import TruncatedSVD
    pca50 = TruncatedSVD(n_components=min(50, d - 1), random_state=0).fit_transform(Zc)
    if pca50.shape[1] < 50:
        pca50 = np.pad(pca50, ((0, 0), (0, 50 - pca50.shape[1])))

    log(f"    UMAP (epochs={UMAP_EPOCHS})")
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    emb2d = umap.UMAP(
        n_neighbors=KNN_K, metric="cosine",
        precomputed_knn=(idx, dist, None),
        n_epochs=UMAP_EPOCHS, random_state=0,
    ).fit_transform(Zn)

    out = pl.DataFrame({
        "hsc_object_id":     obj_id_eff,
        "umap_x":            emb2d[:, 0].astype(np.float32),
        "umap_y":            emb2d[:, 1].astype(np.float32),
        "label":             label_eff.astype(np.float32),
        "label_name":        [label_name] * n_eff,
        "model":             [model] * n_eff,
        "survey":             [survey] * n_eff,
        "block":             [block_name] * n_eff,
        "knn_purity_local":  purity_per_row,
        "twoNN_dim_local":   local_dim,
        # PCA-50 as fixed-width float32 list (avoids polars' f64 promotion).
        "pca50":             pl.Series(
                                "pca50",
                                pca50.astype(np.float32),
                                dtype=pl.Array(pl.Float32, 50),
                             ),
    })
    out = out.with_columns(pl.lit(purity).alias("knn_purity"))

    del Z, Zc, Zn, idx, dist, emb2d, pca50, purity_per_row, local_dim
    gc.collect()
    return out


# ---------------------------------------------------------------------------
# Top-level loop
# ---------------------------------------------------------------------------
def safe_tag(model: str, survey: str) -> str:
    return f"{survey}__{model}".replace("/", "_")


def upload_done(tag: str, files: list[str]) -> None:
    """Upload all of a tuple's outputs in ONE commit via upload_folder.
    Cuts HF commit volume from O(blocks_per_tuple) to O(1)."""
    if not files:
        return
    # Stage files into a temp dir mirroring the desired path-in-repo layout.
    staging = tempfile.mkdtemp(prefix="pu_umap_upload_")
    try:
        target = os.path.join(staging, "done", tag)
        os.makedirs(target, exist_ok=True)
        for f in files:
            shutil.copy2(f, os.path.join(target, os.path.basename(f)))
        api.upload_folder(
            folder_path=staging,
            repo_id=COORD_REPO,
            repo_type="dataset",
            commit_message=f"done {tag} ({len(files)} blocks)",
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def embed_filename(model: str, survey: str) -> str:
    # Actual layout on <owner>/platonic-embeddings: <survey>/<model>_blocks_layerwise.parquet
    return f"{survey}/{model}_blocks_layerwise.parquet"


def main() -> int:
    log(f"worker on {os.uname().nodename} pid={os.getpid()}")
    log(f"COORD_REPO={COORD_REPO}  EMBED_REPO={EMBED_REPO}")

    cand, labels = fetch_static()
    n_per_tag = (
        cand.group_by(["survey", "model"]).len()
            .rename({"len": "n_blocks"})
    )
    work = (cand.select(["survey", "model"]).unique()
                .join(n_per_tag, on=["survey", "model"]))
    log(f"work queue: {len(work)} (model, survey) tuples, "
        f"{cand.height} candidate blocks total")

    # randomize the order so multiple workers diverge naturally
    work = work.sample(fraction=1.0, shuffle=True, seed=os.getpid())

    reap_stale_claims()

    for row in work.iter_rows(named=True):
        survey, model, n_blocks = row["survey"], row["model"], row["n_blocks"]
        tag = safe_tag(model, survey)

        if already_done(tag, n_blocks):
            log(f"[skip] {tag}: already in done/")
            continue
        if tag in list_running():
            log(f"[skip] {tag}: claimed by another worker")
            continue
        if not try_claim(tag):
            log(f"[skip] {tag}: claim race lost")
            continue

        local = None
        try:
            emb_fname = embed_filename(model, survey)
            log(f"[claim] {tag} -> downloading {emb_fname}")
            t0 = time.time()
            local = hf_hub_download(EMBED_REPO, emb_fname, repo_type="dataset",
                                    token=HF_TOKEN, local_dir=DLCACHE)
            log(f"  downloaded in {time.time()-t0:.1f}s "
                f"({os.path.getsize(local)/1e9:.1f} GB)")

            blocks = (cand.filter((pl.col("survey") == survey) & (pl.col("model") == model))
                          ["block_a_name"].to_list())

            try:
                label_arr, label_name, obj_id = survey_label(labels, survey)
            except Exception:
                log(f"  [{tag}] survey_label FAILED:\n{traceback.format_exc()}")
                continue   # release_claim runs in finally, then move on
            written: list[str] = []
            for block_name in blocks:
                try:
                    out = process_block(model, survey, block_name, local,
                                        label_arr, label_name, obj_id)
                    safe_block = block_name.replace("/", "_").replace(".", "_")
                    fname = f"umap_{tag}__{safe_block}.parquet"
                    fpath = os.path.join(OUT_DIR, fname)
                    out.write_parquet(fpath)
                    written.append(fpath)
                    log(f"    wrote {fpath} ({os.path.getsize(fpath)/1e6:.1f} MB)")
                except Exception:
                    log(f"  [{block_name}] FAILED:\n{traceback.format_exc()}")

            log(f"  uploading {len(written)} outputs to done/{tag}/")
            try:
                upload_done(tag, written)
            except Exception:
                log(f"  [{tag}] upload_done FAILED (worker continues):\n{traceback.format_exc()}")

        except Exception:
            log(f"  [{tag}] tuple FAILED (worker continues):\n{traceback.format_exc()}")
        finally:
            if local:
                try:
                    os.remove(local)
                except Exception:
                    pass
            release_claim(tag)
            log(f"[release] {tag}")

    log("queue exhausted, exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
