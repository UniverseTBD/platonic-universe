# Compute tier

Reads layer-wise embeddings from a public HF dataset; produces calibrated CKA,
MKNN, per-block shape statistics, and per-candidate UMAP outputs.

## Required environment

```bash
export HF_TOKEN=<your-hf-token>                         # private datasets / write access
export PU_SOLVE_RESULTS_REPO=<owner>/pu-solve-results   # phase-1 outputs land here
export PU_SOLVE_EMBED_REPO=<owner>/platonic-embeddings  # source of layerwise embeddings
export PU_UMAP_RESULTS_REPO=<owner>/pu-umap-results     # phase-2 outputs land here
export PU_UMAP_EMBED_REPO=<owner>/platonic-embeddings   # same as above
```

No defaults. Scripts fail loudly if any of these are unset.

## One-time setup

```bash
python setup_solve_hf.py  "$PU_SOLVE_RESULTS_REPO"
python setup_umap_hf.py   "$PU_UMAP_RESULTS_REPO"
```

Each creates the dataset (private) with `done/`, `running/`, `static/`
placeholder folders.

## Running

### Phase 1 — solve

Submit many workers; they coordinate via the HF dataset:

```bash
# SLURM (SGE-style)
for i in {1..16}; do sbatch slurm/pu_solve.sh; done

# vast.ai 8-GPU partition mode (only needed for the heavy tail tuple)
bash vast/vast_paligemma.sh
```

### Phase 2 — UMAP

After Phase 1 finishes for the 144 (model, survey) tuples:

```bash
python select_candidates.py     # writes derived/candidates.parquet
# upload candidates + the four labels parquets to PU_UMAP_RESULTS_REPO/static/
# (see setup_umap_hf.upload_candidates_and_labels)
for i in {1..16}; do sbatch slurm/pu_umap.sh; done
```

## Knobs

Every script reads its config from environment variables. The most important:

| Variable | Default | What it does |
|---|---|---|
| `PU_SOLVE_OUT` | `~/pu_runs/solve_out` | persistent local output dir |
| `PU_SOLVE_LOCKS` | `~/pu_runs/solve_locks` | persistent local lock dir |
| `PU_SOLVE_DLCACHE` | (per-job `/tmp`) | embedding parquet cache |
| `PU_SOLVE_COL_CACHE_GB` | `64` | column cache budget per worker |
| `PU_SOLVE_MAX_BLOCK_DIM` | `16384` | drops layers wider than this (e.g. paligemma's vocab projection) |
| `PU_SOLVE_PAIR_RANGE` | unset | enables partition mode for one task: `start:end` |
| `PU_SOLVE_TARGET` | unset | with `PAIR_RANGE`, restricts to one (survey, model) tuple |
| `PU_SOLVE_N_PERM` | `1000` | permutations for Gröger calibration |
| `PU_UMAP_KNN_K` | `50` | $k$ for kNN-purity / UMAP |

## Cluster-specific config

The SLURM and vast.ai wrappers have a `PER_CLUSTER` block at the top —
edit those four lines (venv-activate command, output dir, lock dir,
module-load string) for your environment. Account/partition lines in
the `#SBATCH` directives need the same treatment.
