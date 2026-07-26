#!/usr/bin/env python3
"""ONE-SHOT layerwise crossmodal calibration.

  sbatch pu_solve.py         # queue one worker
  sbatch pu_solve.py         # queue another (atomically claims separate work)
  sbatch pu_solve.py         # ...as many as your fair-share will allow

Each invocation is one worker. Workers cooperate via lockfiles and existing-output
checks on /work/nvme — there is no central scheduler. Re-running is a no-op for
already-completed tasks. Crashed jobs leave stale locks; pass --reset-stale to
clean them up.

WHAT IT COMPUTES — per (survey, model) parquet on <owner>/platonic-embeddings:
  - Per-block-pair calibrated CKA  (HSC-side block i × second-side block j)
  - Per-block-pair calibrated MKNN
  - 1000-permutation Gröger calibration on GPU using cached centered Grams
  - Output: <out>/<survey>__<model>.parquet, long-form, polars-readable.

WHAT IT NEEDS:
  - HF_TOKEN env var (the only secret you give it; dodges anonymous rate limits).
  - Anything pip/conda gave you that already runs the analysis stack:
      torch (CUDA), polars, pyarrow, numpy, huggingface_hub.

WHY THIS DOESN'T HANG LIKE THE LAST RUN:
  Old `wiring.stream_column` used `pf.iter_batches(...)` over `HfFileSystem`,
  which makes one HTTP range-request per row-group. For parquets with ~10k tiny
  row-groups (LLaVA-OneVision and friends) that's hours of pure HTTP latency
  per file. Here we `hf_hub_download` the whole parquet once per task, then read
  columns from local disk — single bulk download, no per-row-group round-trips.
"""

# ---- SLURM directives ------------------------------------------------------
#SBATCH --job-name=pu_solve
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --partition=ghx4
#SBATCH --account=<your-slurm-account>
#SBATCH --output=pu_solve_%j.log
#SBATCH --error=pu_solve_%j.err
# ---------------------------------------------------------------------------

import argparse
import errno
import gc
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

HF_REPO = os.environ.get("PU_SOLVE_EMBED_REPO", "")  # set to e.g. "<owner>/platonic-embeddings"
SURVEYS = ("desi", "jwst", "legacysurvey", "sdss")

# Paths — override via env if you want them somewhere else.
ROOT = Path(os.environ.get("PU_SOLVE_ROOT", str(Path.home() / "pu_runs")))
OUT_DIR = Path(os.environ.get("PU_SOLVE_OUT", str(ROOT / "solve_out")))
LOCK_DIR = Path(os.environ.get("PU_SOLVE_LOCKS", str(ROOT / "solve_locks")))
DL_CACHE = Path(os.environ.get("PU_SOLVE_DLCACHE", str(ROOT / ".solve_dl")))

# Knobs — sane defaults; override via env.
# N_SUBSAMPLE=0 means "use full N". With feature-space CKA and sparse MKNN
# (see calibrate_pair_gpu below), memory scales as O(N·d), not O(N²), so
# we can do full-N permutation calibration on legacysurvey (N≈100k) without
# OOM. The MAX_N_GPU guard exists only for absurdly large hypothetical
# datasets; raise if you actually have them.
N_SUBSAMPLE = int(os.environ.get("PU_SOLVE_N_SUBSAMPLE", "0"))
MAX_N_GPU = int(os.environ.get("PU_SOLVE_MAX_N_GPU", "250000"))
N_PERM = int(os.environ.get("PU_SOLVE_N_PERM", "1000"))
K_MKNN = int(os.environ.get("PU_SOLVE_K", "10"))
SEED = int(os.environ.get("PU_SOLVE_SEED", "42"))
STALE_LOCK_S = int(os.environ.get("PU_SOLVE_STALE_LOCK_S", "3600"))
# kNN-build chunk size (smaller = less peak memory, more launch overhead).
KNN_CHUNK = int(os.environ.get("PU_SOLVE_KNN_CHUNK", "4096"))

# Cross-cluster coordination (optional). Setting PU_SOLVE_RESULTS_REPO to
# a HF dataset id (e.g. "<owner>/pu-solve-results") lets multiple SLURM
# clusters cooperate: each task is claim-marked on HF before processing,
# completed parquets are uploaded, claim markers deleted. Workers from
# either cluster see each other's state via dataset listings.
RESULTS_REPO = os.environ.get("PU_SOLVE_RESULTS_REPO") or None
RESULTS_REPO_TYPE = "dataset"
HF_STALE_RUNNING_S = int(os.environ.get("PU_SOLVE_HF_STALE_S", "3600"))

# Pathological-column filter. Vocab-projection layers like paligemma's
# `lm_head_*` (d ≈ 256k) cannot be calibrated — Z.T @ Z would need
# d² × 4 bytes ≈ 246 GiB on the GPU, AND pyarrow's int32 fixed-size-list
# offsets overflow at N=101k × d=256k = 2.6e10 elements. Columns with
# list_size above MAX_BLOCK_DIM are dropped at discovery time. Real
# transformer hidden layers max out around d=8192, so 16384 is generous.
MAX_BLOCK_DIM = int(os.environ.get("PU_SOLVE_MAX_BLOCK_DIM", "16384"))
EXCLUDE_NAME_SUBSTRINGS = tuple(
    s.strip() for s in os.environ.get(
        "PU_SOLVE_EXCLUDE_NAMES", "lm_head"
    ).split(",") if s.strip()
)

# ---- single-task partition mode --------------------------------------------
# When PU_SOLVE_TARGET is set, the worker bypasses the HF claim queue and
# processes exactly one (survey, model) tuple. Combined with PU_SOLVE_PAIR_RANGE,
# this lets multiple workers split the (block_a × block_b) grid for a single
# heavy task. Each partition writes <tag>.partial.<start>-<end>.parquet which
# merge_partials.py concatenates into the final <tag>.parquet.
TARGET = os.environ.get("PU_SOLVE_TARGET", "")           # e.g. "legacysurvey/paligemma_28b_28b"
PAIR_RANGE_STR = os.environ.get("PU_SOLVE_PAIR_RANGE", "")  # e.g. "2738:5476"
PAIR_START, PAIR_END = (None, None)
if PAIR_RANGE_STR:
    _s, _e = PAIR_RANGE_STR.split(":")
    PAIR_START, PAIR_END = int(_s), int(_e)


# ---- tiny logger -----------------------------------------------------------
def log(*a, **kw):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True, **kw)


# ---- HF discovery ----------------------------------------------------------
def list_models(survey: str, hf_token: str) -> list[str]:
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=hf_token)
    try:
        files = fs.ls(f"datasets/{HF_REPO}/{survey}", detail=False)
    except Exception as e:
        log(f"[warn] list {survey}: {type(e).__name__}: {e}")
        return []
    out = []
    for f in files:
        name = Path(f).name
        if name.endswith("_blocks_layerwise.parquet"):
            out.append(name[: -len("_blocks_layerwise.parquet")])
    return sorted(out)


def list_all_tasks(hf_token: str) -> list[tuple[str, str]]:
    tasks: list[tuple[str, str]] = []
    for s in SURVEYS:
        for m in list_models(s, hf_token):
            tasks.append((s, m))
    return tasks


# ---- HF cross-cluster coordination -----------------------------------------
#
# Two clusters cooperating via a shared HF dataset. Layout:
#   datasets/<RESULTS_REPO>/done/<survey>__<model>.parquet     # completed work
#   datasets/<RESULTS_REPO>/running/<survey>__<model>.running  # active claim
#
# Per task, a worker:
#   1. Lists dataset files (one API call).
#   2. Skips if `done/<task>.parquet` exists.
#   3. Skips if `running/<task>.running` exists AND younger than
#      HF_STALE_RUNNING_S — older claims belong to crashed workers, take over.
#   4. PUTs `running/<task>.running` containing worker_id + timestamp.
#      (HF doesn't have atomic create-or-fail; race window from list-to-PUT
#      is ~100 ms; both racing workers will run; last `done` upload wins.
#      Wasted compute but never corrupt.)
#   5. Processes locally.
#   6. PUTs `done/<task>.parquet`, then deletes `running/<task>.running`.
def _hf_clients(hf_token: str):
    """Lazy import + construct HF api/fs once. Returns (api, fs, worker_id)."""
    import socket

    from huggingface_hub import HfApi, HfFileSystem
    api = HfApi(token=hf_token)
    fs = HfFileSystem(token=hf_token)
    worker_id = f"{socket.gethostname()}-{os.getpid()}-{int(time.time())}"
    return api, fs, worker_id


def hf_list_state(api, repo: str) -> tuple[set[str], dict[str, float]]:
    """Returns (done_keys, running_keys_with_age_seconds).
    `done_keys`: set of "<survey>__<model>" strings.
    `running_keys_with_age_seconds`: dict mapping the same key to age in s.
    """
    done: set[str] = set()
    running: dict[str, float] = {}
    try:
        files = api.list_repo_files(repo, repo_type=RESULTS_REPO_TYPE)
    except Exception as e:
        log(f"[hf-coord] list_repo_files failed: {type(e).__name__}: {e}")
        return done, running
    # We need ages for running files — list_repo_tree gives last_commit info.
    try:
        siblings = api.dataset_info(repo).siblings or []
        commit_times: dict[str, float] = {}
        for s in siblings:
            lc = getattr(s, "lastCommit", None)
            if lc and getattr(lc, "date", None):
                commit_times[s.rfilename] = lc.date.timestamp()
    except Exception:
        commit_times = {}
    now = time.time()
    for f in files:
        if f.startswith("done/") and f.endswith(".parquet"):
            done.add(f[len("done/"):-len(".parquet")])
        elif f.startswith("running/") and f.endswith(".running"):
            key = f[len("running/"):-len(".running")]
            ct = commit_times.get(f, now)  # if no info, treat as just-now
            running[key] = max(0.0, now - ct)
    return done, running


def hf_try_claim(api, repo: str, survey: str, model: str,
                 worker_id: str) -> bool:
    """Upload a claim marker. Returns True iff upload succeeded."""
    task_key = f"{survey}__{model}"
    body = f"{worker_id}@{int(time.time())}\n".encode()
    try:
        api.upload_file(
            path_or_fileobj=body,
            path_in_repo=f"running/{task_key}.running",
            repo_id=repo,
            repo_type=RESULTS_REPO_TYPE,
            commit_message=f"claim {task_key} by {worker_id}",
        )
        return True
    except Exception as e:
        log(f"[hf-coord] claim {task_key}: {type(e).__name__}: {e}")
        return False


def hf_upload_done(api, repo: str, survey: str, model: str,
                   local_parquet: Path) -> bool:
    """Upload completed parquet and try to delete the running marker."""
    task_key = f"{survey}__{model}"
    try:
        api.upload_file(
            path_or_fileobj=str(local_parquet),
            path_in_repo=f"done/{task_key}.parquet",
            repo_id=repo,
            repo_type=RESULTS_REPO_TYPE,
            commit_message=f"done {task_key}",
        )
    except Exception as e:
        log(f"[hf-coord] upload {task_key}: {type(e).__name__}: {e}")
        return False
    try:
        api.delete_file(
            path_in_repo=f"running/{task_key}.running",
            repo_id=repo,
            repo_type=RESULTS_REPO_TYPE,
            commit_message=f"finished {task_key}",
        )
    except Exception:
        # Stale running marker isn't fatal — it just gets reaped after
        # HF_STALE_RUNNING_S elapsed.
        pass
    return True


def hf_drop_running(api, repo: str, survey: str, model: str) -> None:
    """Best-effort: drop running marker on failure path."""
    try:
        api.delete_file(
            path_in_repo=f"running/{survey}__{model}.running",
            repo_id=repo,
            repo_type=RESULTS_REPO_TYPE,
            commit_message=f"drop claim {survey}__{model}",
        )
    except Exception:
        pass


# ---- atomic claim via O_EXCL lockfile --------------------------------------
def try_claim(lock_path: Path) -> bool:
    """Returns True iff this worker now owns the lock."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()}@{int(time.time())}\n".encode())
        os.close(fd)
        return True
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    # Lock exists. Maybe stale.
    try:
        age = time.time() - lock_path.stat().st_mtime
    except FileNotFoundError:
        return False  # raced; another worker just released it
    if age <= STALE_LOCK_S:
        return False
    try:
        lock_path.unlink()
    except OSError:
        return False
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()}@{int(time.time())}\n".encode())
        os.close(fd)
        return True
    except OSError:
        return False


def release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except OSError:
        pass


def touch_lock(lock_path) -> None:
    if lock_path is None:
        return
    try:
        os.utime(lock_path, None)
    except OSError:
        pass


# ---- HF download (bulk, not range-streamed) --------------------------------
def download_parquet(survey: str, model: str, hf_token: str) -> str:
    from huggingface_hub import hf_hub_download
    DL_CACHE.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        filename=f"{survey}/{model}_blocks_layerwise.parquet",
        token=hf_token,
        cache_dir=str(DL_CACHE),
    )


# ---- block-column discovery + loading --------------------------------------
def list_blocks(local_path: str, survey: str) -> tuple[list[str], list[str], list[str]]:
    """Discover candidate column names. Filters out:
      - any column whose name contains a substring in EXCLUDE_NAME_SUBSTRINGS
        (default: 'lm_head' — paligemma/llava vocab projections at d ≈ 256k
        that OOM Z.T@Z and overflow pyarrow int32 offsets at large N)
      - any fixed_size_list column with list_size > MAX_BLOCK_DIM (real
        hidden layers cap at d≈8192; 16384 is a safe ceiling)
    """
    pf = pq.ParquetFile(local_path)
    schema = pf.schema_arrow
    accepted: list[str] = []
    skipped: list[tuple[str, str]] = []
    for f in schema:
        name = f.name
        # Name-pattern filter (lm_head etc.)
        bad_name = next(
            (sub for sub in EXCLUDE_NAME_SUBSTRINGS if sub in name), None,
        )
        if bad_name:
            skipped.append((name, f"name contains {bad_name!r}"))
            continue
        # Dimension filter — only meaningful for fixed_size_list types.
        d = None
        try:
            d = int(f.type.list_size)
        except (AttributeError, ValueError, TypeError):
            pass
        if d is not None and d > MAX_BLOCK_DIM:
            skipped.append((name, f"d={d} > MAX_BLOCK_DIM={MAX_BLOCK_DIM}"))
            continue
        accepted.append(name)
    if skipped:
        log(f"  [{survey}] dropped {len(skipped)} pathological columns:")
        for name, reason in skipped[:6]:
            log(f"    - {name}  ({reason})")
        if len(skipped) > 6:
            log(f"    ... and {len(skipped) - 6} more")
    hsc = [n for n in accepted if n.endswith("_hsc")]
    side = [n for n in accepted if n.endswith(f"_{survey}")]
    paired = [n for n in accepted if n == f"{survey}_embedding"]
    return hsc, side, paired


def read_column(local_path: str, column: str) -> np.ndarray:
    """Bulk-read one fixed-size-list column → (N, d) float32."""
    pf = pq.ParquetFile(local_path)
    table = pf.read(columns=[column])
    arr = table.column(0).combine_chunks()
    flat = np.asarray(arr.values, dtype=np.float32)
    d = arr.type.list_size
    return flat.reshape(-1, d)


# ---- Gröger calibration ----------------------------------------------------
def gröger(observed: float, nulls: np.ndarray,
           alpha: float = 0.05, smax: float = 1.0) -> dict:
    K = len(nulls)
    combined = np.concatenate([
        np.array([observed], dtype=np.float64),
        np.asarray(nulls, dtype=np.float64),
    ])
    combined.sort()
    idx = max(0, min(int(np.ceil((1.0 - alpha) * (K + 1))) - 1, K))
    tau = float(combined[idx])
    cal = max(0.0, min(1.0, (float(observed) - tau) / max(smax - tau, 1e-30)))
    p = float((1 + (nulls >= observed).sum()) / (K + 1))
    return {
        "score": float(observed),
        "calibrated_score": cal,
        "tau": tau,
        "p_value": p,
        "null_mean": float(nulls.mean()),
        "null_std": float(nulls.std() + 1e-30),
    }


# ---- GPU permutation calibrators ------------------------------------------
#
# Memory scaling:
#   CKA  via FEATURE-SPACE — uses ||Z1_c.T @ Z2_c[π]||_F² ; never builds an
#                            N×N Gram. Peak memory: O(N·d) for embeddings +
#                            O(d²) for working matrices.
#   MKNN via SPARSE kNN   — represents the kNN graph as (N, k) int tensors
#                            instead of an N×N membership matrix. Peak
#                            memory: O(N·k) + chunk·N for the kNN build.
#
# At N=101k, d=4096, k=10 these together peak around 5 GB on the GPU,
# vs 120 GB for the dense Gram-and-membership formulation we used before.
# That's why this version can do full-N legacysurvey permutation calibration.

def _enable_tf32() -> None:
    """Hopper/Ampere matmul gets ~2× faster on TF32; precision is sufficient
    for permutation testing (10-bit mantissa, fp32 accumulator)."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _build_knn_chunked(Zn, k: int, chunk: int):
    """Build a (N, k) tensor of top-k cosine neighbours, chunked to avoid
    materializing the full N×N similarity. Self-similarity is masked.
    `Zn` must already be L2-row-normalized."""
    import torch
    n = Zn.shape[0]
    knn = torch.empty((n, k), dtype=torch.long, device=Zn.device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sim_chunk = Zn[start:end] @ Zn.T  # (chunk, N)
        # Mask self-similarity (per-row diagonal)
        local_rows = torch.arange(end - start, device=Zn.device)
        global_cols = torch.arange(start, end, device=Zn.device)
        sim_chunk[local_rows, global_cols] = -float("inf")
        knn[start:end] = sim_chunk.topk(k, dim=1).indices
        del sim_chunk
    return knn


def _mknn_overlap_sparse(knn1, knn2, k: int):
    """Mean per-row |kNN_1 ∩ kNN_2| / k for two (N, k) sparse kNN tensors.
    Uses an (N, k, k) bool intermediate; at k=10 that's 100 N bytes."""
    matches = (knn1.unsqueeze(2) == knn2.unsqueeze(1))  # (N, k, k)
    overlap_count = matches.any(dim=2).sum(dim=1)        # (N,)
    return overlap_count.float().mean() / float(k)


def calibrate_pair_gpu(Z1: np.ndarray, Z2: np.ndarray, k: int,
                       n_perm: int, seed: int, device,
                       knn_chunk: int = KNN_CHUNK) -> dict:
    """Returns {'cka': (obs, nulls), 'mknn': (obs, nulls)} at full N.

    CKA uses feature-space identity:
        ||Z1_c^T Z2_c[π]||_F² / (||Z1_c^T Z1_c||_F · ||Z2_c^T Z2_c||_F)
    Per perm: one (d1, N)×(N, d2) matmul; no N×N intermediate.

    MKNN uses sparse kNN graphs (not dense N×N membership). The
    permutation null re-pairs object labels via perm/perm_inv lookups
    and counts overlap on (N, k, k) tensors.
    """
    import torch
    _enable_tf32()
    g_cka = torch.Generator(device=device).manual_seed(int(seed))
    g_mk = torch.Generator(device=device).manual_seed(int(seed) + 1)

    Z1t = torch.from_numpy(Z1).to(device).float()
    Z2t = torch.from_numpy(Z2).to(device).float()
    n = Z1.shape[0]

    # ===== CKA — feature-space, no Gram =====
    Z1c = Z1t - Z1t.mean(dim=0, keepdim=True)
    Z2c = Z2t - Z2t.mean(dim=0, keepdim=True)

    # Denominator (permutation-invariant): compute once.
    XtX = Z1c.T @ Z1c                                    # (d1, d1)
    YtY = Z2c.T @ Z2c                                    # (d2, d2)
    nrm1 = torch.sqrt((XtX * XtX).sum() + 1e-30)
    nrm2 = torch.sqrt((YtY * YtY).sum() + 1e-30)
    denom = nrm1 * nrm2
    del XtX, YtY

    # Observed
    M = Z1c.T @ Z2c                                      # (d1, d2)
    cka_obs = ((M * M).sum() / denom).item()
    del M

    # Permutation null: per perm, one matmul.
    cka_nulls = torch.empty(n_perm, device=device)
    for i in range(n_perm):
        perm = torch.randperm(n, generator=g_cka, device=device)
        Mp = Z1c.T @ Z2c[perm]                           # (d1, d2)
        cka_nulls[i] = (Mp * Mp).sum() / denom
        del Mp
    cka_nulls_np = cka_nulls.cpu().numpy()
    del Z1c, Z2c, cka_nulls

    # ===== MKNN — sparse kNN, no membership matrix =====
    Z1n = Z1t / (Z1t.norm(dim=1, keepdim=True) + 1e-30)
    Z2n = Z2t / (Z2t.norm(dim=1, keepdim=True) + 1e-30)
    del Z1t, Z2t

    knn1 = _build_knn_chunked(Z1n, k, knn_chunk)         # (N, k) long
    knn2 = _build_knn_chunked(Z2n, k, knn_chunk)
    del Z1n, Z2n

    mknn_obs = _mknn_overlap_sparse(knn1, knn2, k).item()

    # Permutation null. permuted_knn_2[i] = π_inv(knn_2[π(i)]) — repair the
    # object-label space without rebuilding any kNN graph.
    mknn_nulls = torch.empty(n_perm, device=device)
    arange_n = torch.arange(n, device=device)
    for i in range(n_perm):
        perm = torch.randperm(n, generator=g_mk, device=device)
        perm_inv = torch.empty_like(perm)
        perm_inv[perm] = arange_n
        knn2_perm = perm_inv[knn2[perm]]                  # (N, k)
        mknn_nulls[i] = _mknn_overlap_sparse(knn1, knn2_perm, k)
        del knn2_perm, perm_inv, perm
    mknn_nulls_np = mknn_nulls.cpu().numpy()
    del knn1, knn2, mknn_nulls, arange_n

    if hasattr(device, "type") and device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "cka": (cka_obs, cka_nulls_np),
        "mknn": (mknn_obs, mknn_nulls_np),
    }


# ---- Per-block manifold-shape statistics ----------------------------------
#
# Computed once per column at first cache-load.  All operations run on the
# same GPU as the calibration step.  Memory profile per column:
#   - feature copy:            N · d · 4 bytes (already paid by calibration)
#   - column covariance:       d · d · 4 bytes (transient, e.g. 85 MB at d=4608)
#   - eigvalsh of covariance:  ~d · d · 4 bytes scratch (LAPACK xSYEVD)
#   - chunked kNN:             chunk · N · 4 bytes (1.6 GB at chunk=4096, N=101k)
# Total transient peak is well below 4 GB on the GPU at the largest model
# we'll see (paligemma_28b at d=4608, N=101k).  Each column adds ~1–4 s
# of GPU work on H100; ~70 columns → a few extra minutes per task.
def compute_shape_stats_gpu(Z: np.ndarray, device, seed: int = 0,
                             knn_k: int = 10,
                             knn_chunk: int = 4096,
                             aniso_pairs: int = 50_000) -> dict[str, float]:
    """Per-block manifold descriptors.  Returns dict of stat_name → scalar.

    Stats:
      anisotropy           — mean pairwise cosine over `aniso_pairs` random
                              row pairs.  ~1 = cone-collapsed, ~0 = isotropic.
      mean_l2_norm         — mean ||Z[i]|| across rows; the activation scale.
      participation_ratio  — (Σλ)² / Σλ² of the column-cov spectrum;
                              "effective dimensionality" by 2nd moment.
      top_eigval_ratio     — λ_max / Σλ; how concentrated the cov is.
      effective_rank       — exp(spectral entropy of cov); information-theoretic
                              effective dimensionality.
      knn_dist_p50         — median k-th cosine-NN distance (k = knn_k).
      intrinsic_dim_twoNN  — Facco et al. 2017 TwoNN intrinsic-dim estimator.
    """
    import torch
    g = torch.Generator(device=device).manual_seed(int(seed))
    Zt = torch.from_numpy(Z).to(device, non_blocking=True).float()
    n, d = Zt.shape
    stats: dict[str, float] = {}

    # ---- L2 norms + cosine-normalised view ----
    norms = Zt.norm(dim=1)
    stats["mean_l2_norm"] = float(norms.mean())
    Zn = Zt / (norms.unsqueeze(1) + 1e-30)

    # ---- Anisotropy: mean cosine over random row pairs ----
    if n > 1:
        n_pairs = min(aniso_pairs, n * 20)
        i_idx = torch.randint(0, n, (n_pairs,), generator=g, device=device)
        j_idx = torch.randint(0, n, (n_pairs,), generator=g, device=device)
        mask = i_idx != j_idx
        if int(mask.sum()) > 0:
            cos = (Zn[i_idx[mask]] * Zn[j_idx[mask]]).sum(dim=1)
            stats["anisotropy"] = float(cos.mean())
        else:
            stats["anisotropy"] = float("nan")
        del i_idx, j_idx, mask
    else:
        stats["anisotropy"] = float("nan")

    # ---- Spectral stats from column covariance ----
    Zc = Zt - Zt.mean(dim=0, keepdim=True)
    cov = (Zc.T @ Zc) / float(max(n - 1, 1))
    try:
        eigvals = torch.linalg.eigvalsh(cov)
    except Exception:
        # Fall back to f64 if f32 eigh fails on edge cases.
        eigvals = torch.linalg.eigvalsh(cov.double()).float()
    del cov, Zc
    eigvals = eigvals.clamp(min=0.0)
    s_eig = float(eigvals.sum())
    sq_eig = float((eigvals * eigvals).sum())
    stats["participation_ratio"] = (s_eig * s_eig) / (sq_eig + 1e-30)
    stats["top_eigval_ratio"] = (
        float(eigvals.max()) / (s_eig + 1e-30) if eigvals.numel() else float("nan")
    )
    if s_eig > 0:
        p = eigvals / s_eig
        p_pos = p[p > 1e-30]
        H = -(p_pos * torch.log(p_pos)).sum() if p_pos.numel() > 0 else torch.tensor(0.0, device=device)
        stats["effective_rank"] = float(torch.exp(H))
        del p, p_pos
    else:
        stats["effective_rank"] = float("nan")
    del eigvals

    # ---- Chunked kNN distances (cosine) → median + TwoNN ----
    if n >= 3 and knn_k >= 2:
        kk = min(knn_k, n - 1)
        first = torch.empty(n, device=device)
        second = torch.empty(n, device=device)
        kth = torch.empty(n, device=device)
        chunk = min(knn_chunk, n)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            sim = Zn[start:end] @ Zn.T  # (chunk, N) cosine
            local = torch.arange(end - start, device=device)
            glob = torch.arange(start, end, device=device)
            sim[local, glob] = -float("inf")  # exclude self
            top = sim.topk(kk, dim=1).values
            dist = 1.0 - top  # cosine distance from cosine sim
            first[start:end] = dist[:, 0]
            second[start:end] = dist[:, 1]
            kth[start:end] = dist[:, kk - 1]
            del sim, top, dist
        stats["knn_dist_p50"] = float(kth.median())
        keep = (first > 1e-12) & (second > first)
        n_keep = int(keep.sum())
        if n_keep > 10:
            log_mu = torch.log(second[keep] / first[keep])
            denom = float(log_mu.sum())
            stats["intrinsic_dim_twoNN"] = float(n_keep) / denom if denom > 0 else float("nan")
        else:
            stats["intrinsic_dim_twoNN"] = float("nan")
        del first, second, kth
    else:
        stats["knn_dist_p50"] = float("nan")
        stats["intrinsic_dim_twoNN"] = float("nan")

    del Zt, Zn, norms
    if hasattr(device, "type") and device.type == "cuda":
        torch.cuda.empty_cache()
    return stats


# ---- task processor --------------------------------------------------------
ROW_SCHEMA = {
    "survey": pl.String, "model": pl.String,
    "block_a_idx": pl.Int64, "block_a_name": pl.String,
    "block_b_idx": pl.Int64, "block_b_name": pl.String,
    "metric": pl.String,
    "n_used": pl.Int64, "n_full": pl.Int64,
    "d_a": pl.Int64, "d_b": pl.Int64,
    "score": pl.Float64, "calibrated_score": pl.Float64,
    "tau": pl.Float64, "p_value": pl.Float64,
    "null_mean": pl.Float64, "null_std": pl.Float64,
}


def process_task(survey: str, model: str, hf_token: str,
                 device, lock_path: Path,
                 cleanup_after: bool = True) -> list[dict]:
    log(f"[{survey}/{model}] downloading...")
    t0 = time.time()
    local = download_parquet(survey, model, hf_token)
    sz_gb = os.path.getsize(local) / 1e9
    log(f"[{survey}/{model}] downloaded {sz_gb:.2f} GB in {time.time()-t0:.0f}s")
    touch_lock(lock_path)

    try:
        hsc_cols, side_cols, paired = list_blocks(local, survey)
        if not hsc_cols:
            log(f"[{survey}/{model}] no _hsc block columns, skipping")
            return []
        b_cols = side_cols if side_cols else paired
        if not b_cols:
            log(f"[{survey}/{model}] no second-side columns, skipping")
            return []

        # Resolve effective N: by default use full N, capping at MAX_N_GPU
        # to stay within the GPU's memory budget for centered Grams + perms.
        # Override PU_SOLVE_N_SUBSAMPLE > 0 to force a smaller sample regardless.
        n_full = pq.ParquetFile(local).metadata.num_rows
        target = N_SUBSAMPLE if N_SUBSAMPLE > 0 else n_full
        target = min(target, MAX_N_GPU)

        if n_full > target:
            rng = np.random.default_rng(SEED)
            idx = rng.choice(n_full, size=target, replace=False)
            idx.sort()
            n_used = target
            if N_SUBSAMPLE == 0:
                log(f"[{survey}/{model}] note: capped at MAX_N_GPU={MAX_N_GPU} "
                    f"(N_full={n_full} exceeds GPU-safe limit)")
        else:
            idx = np.arange(n_full, dtype=np.int64)
            n_used = n_full

        log(f"[{survey}/{model}] n_full={n_full} n_used={n_used} "
            f"hsc={len(hsc_cols)} side={len(b_cols)}")

        # LRU column cache.  Models like paligemma_3b on legacysurvey have
        # ~110 columns × 5900 dims × 100k rows = 264 GB total — eagerly
        # loading all columns upfront OOMs even on 128 GB nodes.  We cap
        # cache memory at COL_CACHE_GB and re-read evicted columns from the
        # local /tmp parquet (cheap — sequential NVMe reads at GB/s).
        from collections import OrderedDict
        col_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        cache_budget_bytes = int(float(os.environ.get("PU_SOLVE_COL_CACHE_GB", "64")) * 1e9)

        # Per-block shape stats: computed once per column on first cache-miss,
        # idempotent via shape_done set.  Recorded into shape_rows alongside
        # the pair-calibration rows; written to the same output parquet.
        shape_done: set[str] = set()
        shape_rows: list[dict] = []

        def record_shape(col: str, side_label: str, side_block_idx: int,
                         arr: np.ndarray) -> None:
            if col in shape_done:
                return
            shape_done.add(col)  # mark before compute so a failure doesn't retry
            try:
                stats = compute_shape_stats_gpu(arr, device, seed=SEED)
            except Exception as e:
                log(f"[{survey}/{model}] shape '{col}' failed: "
                    f"{type(e).__name__}: {e}")
                return
            for stat_name, val in stats.items():
                shape_rows.append({
                    "survey": survey, "model": model,
                    "block_a_idx": int(side_block_idx),
                    "block_a_name": col,
                    "block_b_idx": -1,
                    "block_b_name": side_label,
                    "metric": f"shape_{stat_name}",
                    "n_used": int(n_used), "n_full": int(n_full),
                    "d_a": int(arr.shape[1]), "d_b": 0,
                    "score": float(val), "calibrated_score": float(val),
                    "tau": float("nan"), "p_value": float("nan"),
                    "null_mean": float("nan"), "null_std": float("nan"),
                })

        def get_col(col: str) -> np.ndarray:
            if col in col_cache:
                col_cache.move_to_end(col)
                return col_cache[col]
            arr = read_column(local, col)[idx]
            col_cache[col] = arr
            cur_bytes = sum(a.nbytes for a in col_cache.values())
            # Evict oldest until under budget; always keep at least 2
            # (we need both ca and cb to compute a pair).
            while cur_bytes > cache_budget_bytes and len(col_cache) > 2:
                _, evicted = col_cache.popitem(last=False)
                cur_bytes -= evicted.nbytes
            return arr

        rows: list[dict] = []
        n_pairs_full = len(hsc_cols) * len(b_cols)
        partition_active = (PAIR_START is not None and PAIR_END is not None)
        if partition_active:
            log(f"[{survey}/{model}] PAIR_RANGE={PAIR_START}:{PAIR_END}  "
                f"(of {n_pairs_full} total pairs)")
        n_pairs = (PAIR_END - PAIR_START) if partition_active else n_pairs_full
        pair_count = 0
        t2 = time.time()
        nb = len(b_cols)
        for i, ca in enumerate(hsc_cols):
            # Skip whole row if no pair in this row falls in our partition.
            if partition_active:
                row_lo, row_hi = i * nb, (i + 1) * nb
                if row_hi <= PAIR_START or row_lo >= PAIR_END:
                    continue
            Za = get_col(ca)
            record_shape(ca, "hsc", i, Za)
            for j, cb in enumerate(b_cols):
                flat_idx = i * nb + j
                if partition_active and not (PAIR_START <= flat_idx < PAIR_END):
                    continue
                pair_count += 1
                t_pair = time.time()
                Zb = get_col(cb)
                record_shape(cb, "side", j, Zb)
                # Per-pair seed: decorrelates null distributions across cells
                # so cross-cell statistical comparisons (e.g. "is HSC L3↔SUR L5
                # significantly stronger than HSC L2↔SUR L6?") aren't biased by
                # shared permutation sequences. SEED is the deterministic anchor;
                # i,j shifts give each cell its own independent null sample.
                pair_seed = SEED + i * 10_000 + j
                out = calibrate_pair_gpu(
                    Za, Zb, K_MKNN, N_PERM, pair_seed, device,
                )
                for metric, (obs, nulls) in out.items():
                    cal = gröger(obs, nulls)
                    rows.append({
                        "survey": survey, "model": model,
                        "block_a_idx": int(i), "block_a_name": ca,
                        "block_b_idx": int(j), "block_b_name": cb,
                        "metric": metric,
                        "n_used": int(n_used), "n_full": int(n_full),
                        "d_a": int(Za.shape[1]),
                        "d_b": int(Zb.shape[1]),
                        **cal,
                    })
                if pair_count % 10 == 0 or pair_count == n_pairs:
                    cur_gb = sum(a.nbytes for a in col_cache.values()) / 1e9
                    log(f"[{survey}/{model}] pair {pair_count}/{n_pairs}  "
                        f"({time.time()-t_pair:.1f}s/pair, "
                        f"avg {(time.time()-t2)/pair_count:.1f}s, "
                        f"cache={len(col_cache)}cols/{cur_gb:.1f}GB)")
                    touch_lock(lock_path)

        # Append per-block shape stats (one row per (column, stat)) to the
        # same long-form output.  Plotting code filters by metric prefix:
        #   pair-alignment rows: metric ∈ {"cka", "mknn"}
        #   shape rows:          metric.startswith("shape_")
        rows.extend(shape_rows)

        col_cache.clear()
        gc.collect()
        log(f"[{survey}/{model}] {len(rows)} rows "
            f"(pair: {len(rows) - len(shape_rows)}, shape: {len(shape_rows)}) "
            f"in {time.time()-t0:.0f}s total")
        return rows
    finally:
        if cleanup_after:
            try:
                p = Path(local).resolve()
                if str(p).startswith(str(DL_CACHE.resolve())):
                    p.unlink()
                    log(f"[{survey}/{model}] cleaned local download")
            except OSError as e:
                log(f"[{survey}/{model}] cleanup warning: {e}")


def write_output_atomic(survey: str, model: str, rows: list[dict]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if PAIR_START is not None and PAIR_END is not None:
        suffix = f".partial.{PAIR_START}-{PAIR_END}"
    final = OUT_DIR / f"{survey}__{model}{suffix}.parquet"
    tmp   = OUT_DIR / f"{survey}__{model}{suffix}.parquet.tmp.{os.getpid()}"
    pl.DataFrame(rows, schema=ROW_SCHEMA).write_parquet(tmp, compression="zstd")
    os.rename(tmp, final)
    return final


# ---- main ------------------------------------------------------------------
def cmd_status(tasks: list[tuple[str, str]]) -> None:
    done = locked = 0
    pending: list[tuple[str, str]] = []
    for s, m in tasks:
        if (OUT_DIR / f"{s}__{m}.parquet").exists():
            done += 1
        elif (LOCK_DIR / f"{s}__{m}.lock").exists():
            locked += 1
        else:
            pending.append((s, m))
    log(f"  total:   {len(tasks)}")
    log(f"  done:    {done}")
    log(f"  locked:  {locked}")
    log(f"  pending: {len(pending)}")
    if pending and len(pending) <= 30:
        log("  pending list:")
        for s, m in pending:
            log(f"    {s}/{m}")


def cmd_reset_stale() -> None:
    n = 0
    for lock in LOCK_DIR.glob("*.lock"):
        try:
            age = time.time() - lock.stat().st_mtime
            if age > STALE_LOCK_S:
                lock.unlink()
                n += 1
        except OSError:
            pass
    log(f"Cleaned {n} stale locks (>{STALE_LOCK_S}s old).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true",
                    help="Print done/locked/pending counts and exit.")
    ap.add_argument("--reset-stale", action="store_true",
                    help="Delete locks older than $PU_SOLVE_STALE_LOCK_S and exit.")
    ap.add_argument("--no-cleanup", action="store_true",
                    help="Don't delete the downloaded parquet after each task.")
    ap.add_argument("--max-tasks", type=int, default=None,
                    help="Stop this worker after N tasks (default: until empty).")
    args = ap.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        log("[fatal] HF_TOKEN env var not set.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    DL_CACHE.mkdir(parents=True, exist_ok=True)

    # Clean orphaned <task>.parquet.tmp.<DEAD_PID> files from prior crashed
    # workers (older than 1h). Doesn't touch anything younger — those might
    # belong to a sibling worker mid-write.
    for stale in OUT_DIR.glob("*.parquet.tmp.*"):
        try:
            if time.time() - stale.stat().st_mtime > 3600:
                stale.unlink()
                log(f"cleaned orphan tmp: {stale.name}")
        except OSError:
            pass

    # OOM guard: with feature-space CKA + sparse MKNN, peak memory scales
    # as O(N·d), so MAX_N_GPU is very generous (250k by default). The guard
    # only catches absurd manual overrides. Real surveys top out at ~100k.
    if N_SUBSAMPLE > MAX_N_GPU:
        log(f"[fatal] PU_SOLVE_N_SUBSAMPLE={N_SUBSAMPLE} > MAX_N_GPU={MAX_N_GPU}: "
            f"increase MAX_N_GPU after sizing the GPU memory budget for "
            f"O(N·d) embeddings and O(chunk·N) kNN-build scratch.")
        sys.exit(1)

    if args.reset_stale:
        cmd_reset_stale()
        return

    # Stagger startup so multiple workers don't hit HF list_dir simultaneously.
    time.sleep(random.uniform(0.0, 5.0))

    # ---- partition mode: bypass the queue, do exactly one task --------------
    if TARGET:
        try:
            t_survey, t_model = TARGET.split("/", 1)
        except ValueError:
            log(f"[fatal] PU_SOLVE_TARGET must be 'survey/model', got {TARGET!r}")
            sys.exit(1)
        log(f"PARTITION MODE: target={t_survey}/{t_model} "
            f"range={PAIR_RANGE_STR or 'full'}")
        import torch
        device = (torch.device("cuda") if torch.cuda.is_available()
                  else torch.device("cpu"))
        if device.type == "cuda":
            log(f"GPU: {torch.cuda.get_device_name(0)} "
                f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
        else:
            log("WARNING: CUDA not available, running on CPU.")
        # No HF claim, no local lock — partition workers are managed externally.
        # CRITICAL: never clean up the embedding parquet in partition mode.
        # Sibling workers share the same file via DL_CACHE; the first finisher
        # would unlink it out from under the others.
        rows = process_task(
            t_survey, t_model, hf_token, device, lock_path=None,
            cleanup_after=False,
        )
        if rows:
            final = write_output_atomic(t_survey, t_model, rows)
            log(f"=== partition done {t_survey}/{t_model} → {final} ({len(rows)} rows) ===")
        else:
            log("=== partition produced no rows ===")
        return

    log("Listing tasks from HF...")
    tasks = list_all_tasks(hf_token)
    log(f"Total tasks: {len(tasks)}")

    if args.status:
        cmd_status(tasks)
        return

    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log(f"GPU: {torch.cuda.get_device_name(0)} "
            f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        device = torch.device("cpu")
        log("WARNING: CUDA not available, running on CPU. Will be slow.")

    # ---- HF cross-cluster coordination setup ----
    hf_api = hf_worker_id = None
    if RESULTS_REPO:
        try:
            hf_api, _hf_fs, hf_worker_id = _hf_clients(hf_token)
            # Auto-bootstrap: ensure the dataset exists. If we lack write
            # permission to create it (e.g. friend's worker, owner's namespace),
            # exist_ok handles it; if the dataset is missing AND we can't make
            # it, we'll fail on list_repo_files below and fall back to local.
            try:
                from huggingface_hub import create_repo
                create_repo(RESULTS_REPO, repo_type=RESULTS_REPO_TYPE,
                            private=True, exist_ok=True, token=hf_token)
            except Exception as e:
                # Either the repo already exists with us not the owner (fine),
                # or we don't have create permission (also fine if owner made
                # it already). Either way, list_repo_files below will tell us.
                log(f"[hf-coord] create_repo (likely benign): "
                    f"{type(e).__name__}: {e}")
            done_keys, running_ages = hf_list_state(hf_api, RESULTS_REPO)
            before = len(tasks)
            tasks = [
                (s, m) for (s, m) in tasks
                if f"{s}__{m}" not in done_keys
                and (
                    f"{s}__{m}" not in running_ages
                    or running_ages[f"{s}__{m}"] > HF_STALE_RUNNING_S
                )
            ]
            log(f"[hf-coord] repo={RESULTS_REPO} worker={hf_worker_id} "
                f"done={len(done_keys)} running={len(running_ages)} "
                f"remaining={len(tasks)}/{before}")
        except Exception as e:
            log(f"[hf-coord] disabled, list failed: {type(e).__name__}: {e}")
            hf_api = None

    # Distribute work: shuffle so multiple workers don't hammer the same task.
    rng = random.Random((os.getpid() << 16) ^ int(time.time()))
    rng.shuffle(tasks)

    n_done = 0
    t_worker = time.time()
    for survey, model in tasks:
        if args.max_tasks is not None and n_done >= args.max_tasks:
            log(f"hit --max-tasks={args.max_tasks}, stopping.")
            break

        out_path = OUT_DIR / f"{survey}__{model}.parquet"
        lock_path = LOCK_DIR / f"{survey}__{model}.lock"
        if out_path.exists():
            continue
        if not try_claim(lock_path):
            continue

        # If HF coord is enabled, also claim on HF before doing real work.
        # (Local lock prevented two workers on the SAME cluster; HF claim
        # blocks workers on the OTHER cluster.)
        hf_claimed = False
        if hf_api is not None and RESULTS_REPO:
            hf_claimed = hf_try_claim(hf_api, RESULTS_REPO, survey, model, hf_worker_id)
            if not hf_claimed:
                # Couldn't take HF claim — release local and move on.
                release_lock(lock_path)
                continue

        log(f"=== claim {survey}/{model} ===")
        try:
            rows = process_task(
                survey, model, hf_token, device, lock_path,
                cleanup_after=not args.no_cleanup,
            )
            if rows:
                final = write_output_atomic(survey, model, rows)
                log(f"=== done  {survey}/{model} → {final} ({len(rows)} rows) ===")
                if hf_claimed:
                    if hf_upload_done(hf_api, RESULTS_REPO, survey, model, final):
                        log(f"[hf-coord] uploaded {survey}__{model}.parquet")
                n_done += 1
            else:
                log(f"=== empty {survey}/{model} (no rows produced) ===")
                if hf_claimed:
                    hf_drop_running(hf_api, RESULTS_REPO, survey, model)
        except KeyboardInterrupt:
            log(f"interrupted on {survey}/{model}")
            release_lock(lock_path)
            if hf_claimed:
                hf_drop_running(hf_api, RESULTS_REPO, survey, model)
            raise
        except Exception as e:
            log(f"=== ERROR {survey}/{model}: {type(e).__name__}: {e}")
            traceback.print_exc(file=sys.stdout)
            if hf_claimed:
                hf_drop_running(hf_api, RESULTS_REPO, survey, model)
        finally:
            release_lock(lock_path)

    log(f"\nWorker exiting. Did {n_done} tasks in {time.time() - t_worker:.0f}s.")


if __name__ == "__main__":
    main()
