#!/usr/bin/env python3
"""Pairwise MKNN + Wasserstein on the per-tuple kNN graphs uploaded by
pu_regress.py.

Two pair families are computed (matching Ashod's pipeline):

  Intramodal — adjacent model sizes within the same family on the same
               modality. Tests whether scaling preserves representation
               structure.
                 e.g. (hsc, vit/base)   vs (hsc, vit/large)
                      (hsc, vit/large)  vs (hsc, vit/huge)

  Crossmodal — same model size, both modalities. Tests whether the
               model's HSC and JWST embeddings agree on which galaxies
               are nearest neighbours.
                 e.g. (hsc, dino_giant) vs (jwst, dino_giant)

For each pair we compute:
  - MKNN overlap (mean fraction of shared neighbours)
  - Wasserstein-1 distance between the physics-label distributions
    sampled from each side's nearest-neighbour set, per property,
    averaged over galaxies

Output: one summary parquet with one row per (pair_kind, group, ...) entry.

Usage:
    python pu_regress_pairs.py \\
        --pull-from <owner>/pu-regress-results \\
        --out-dir   /path/to/derived
        [--upload-to <owner>/pu-regress-results]   # writes summary.pairs.parquet under done/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Adjacent-size pairs per family. Has to match pu_regress.MODEL_GRID order.
# Hardcoded here to keep this script self-contained; a deviation would be
# obvious as a missing pair in the output.
# ---------------------------------------------------------------------------
FAMILY_SIZES = {
    "vit":       ["base", "large", "huge"],
    "clip":      ["base", "large"],
    "dinov3":    ["vits16", "vits16plus", "vitb16", "vitl16", "vith16plus", "vit7b16"],
    "convnext":  ["nano", "tiny", "base", "large"],
    "ijepa":     ["huge", "giant"],
    "vjepa":     ["large", "huge", "giant"],
    "astropt":   ["015M", "095M", "850M"],
    "vit-mae":   ["base", "large", "huge"],
    "paligemma": ["3b", "10b", "28b"],
    "llava_15":  ["7b", "13b"],
}
MODALITIES = ("hsc", "jwst")

# Set of (modality, alias, size) tuples whose probe files we'll read.
ALL_TUPLES = [
    (m, alias, size)
    for m in MODALITIES
    for alias, sizes in FAMILY_SIZES.items()
    for size in sizes
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--pull-from", default=os.environ.get("PU_REGRESS_RESULTS_REPO"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--upload-to", default=None,
                   help="If set, upload the summary parquet here under done/summary.pairs.parquet.")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Where to keep pulled per-tuple parquets (default: $out_dir/_regress_cache).")
    return p.parse_args()


def fetch_all(repo: str, dest: Path, token: str | None) -> Path:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)
    files = [f for f in api.list_repo_files(repo, repo_type="dataset")
             if f.startswith("done/") and f.endswith(".parquet")]
    if not files:
        raise SystemExit(f"[fatal] no done/*.parquet on {repo}")
    print(f"pulling {len(files)} files from {repo}")
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        hf_hub_download(repo, f, repo_type="dataset",
                        local_dir=str(dest), token=token)
    return dest / "done"


def load_neighbours(done_dir: Path, tag: str) -> np.ndarray | None:
    p = done_dir / tag / "neighbours.parquet"
    if not p.exists():
        return None
    df = pl.read_parquet(p)
    arr = np.array(df["neighbours"].to_list(), dtype=np.int32)
    return arr


def load_probe(done_dir: Path, tag: str) -> pl.DataFrame | None:
    p = done_dir / tag / "probe.parquet"
    if not p.exists():
        return None
    return pl.read_parquet(p)


def reconstruct_catalog_from_probes(done_dir: Path) -> dict[str, np.ndarray] | None:
    """Pulls one probe parquet to learn the catalog row count and properties,
    then reconstructs the per-galaxy parameter arrays by streaming the source
    dataset (this is unavoidable — physics labels themselves aren't uploaded
    per tuple, only the kNN indices into the catalog).
    """
    # pick the first probe parquet we can find
    for tag_dir in done_dir.iterdir():
        if (tag_dir / "probe.parquet").exists():
            df = pl.read_parquet(tag_dir / "probe.parquet")
            n = int(df["n"][0])
            break
    else:
        return None

    # Stream the source dataset to recover labels (same logic as pu_regress).
    from datasets import load_dataset

    from pu.pu_datasets.cosmosweb import CATALOG_COLUMNS
    DATASET = os.environ.get(
        "PU_REGRESS_DATASET",
        "Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2",
    )
    print(f"reconstructing catalog (N={n}) from {DATASET}")
    ds = load_dataset(DATASET, split="train", streaming=True)
    cols = list(CATALOG_COLUMNS.values())
    raw = {c: [] for c in cols}
    for i, row in enumerate(ds):
        for c in cols:
            raw[c].append(row[c])
        if i + 1 >= n:
            break
    out = {param: np.asarray(raw[col], dtype=np.float32)
           for param, col in CATALOG_COLUMNS.items()}
    out["g-r"] = out["mag_g"] - out["mag_r"]
    return out


def mknn_overlap(nn1: np.ndarray, nn2: np.ndarray) -> float:
    """Mean per-row fraction |N1 ∩ N2| / k. Same metric as
    pu.metrics.neighbors.mknn_neighbor_input."""
    k = nn1.shape[1]
    overlap = [len(set(a).intersection(b)) for a, b in zip(nn1, nn2)]
    return float(np.mean(overlap) / k)


def wasserstein_per_property(nn1: np.ndarray, nn2: np.ndarray,
                             y: np.ndarray) -> float:
    """Per-row W1 between {y[N1(i)]} and {y[N2(i)]}, averaged. Same as
    pu.metrics.physics.wass_distance."""
    from scipy.stats import wasserstein_distance as w_d
    n = nn1.shape[0]
    finite = np.isfinite(y)
    if not finite.any():
        return float("nan")
    y = np.where(finite, y, np.nan)
    vals = []
    for i in range(n):
        s1 = y[nn1[i]]
        s2 = y[nn2[i]]
        s1 = s1[np.isfinite(s1)]
        s2 = s2[np.isfinite(s2)]
        if len(s1) == 0 or len(s2) == 0:
            continue
        vals.append(w_d(s1, s2))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    args = parse_args()
    if not args.pull_from:
        print("[fatal] --pull-from required (or set $PU_REGRESS_RESULTS_REPO)",
              file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or (args.out_dir / "_regress_cache")
    done_dir = fetch_all(args.pull_from, cache, os.environ.get("HF_TOKEN"))

    catalog = reconstruct_catalog_from_probes(done_dir)
    if catalog is None:
        print("[fatal] no probe parquet found in cache", file=sys.stderr)
        return 1

    # Cache loaded neighbour matrices once.
    nn_cache: dict[tuple[str, str, str], np.ndarray] = {}
    for modality, alias, size in ALL_TUPLES:
        tag = f"{modality}__{alias}_{size}"
        n = load_neighbours(done_dir, tag)
        if n is not None:
            nn_cache[(modality, alias, size)] = n

    print(f"loaded {len(nn_cache)}/{len(ALL_TUPLES)} neighbour matrices")
    rows = []

    # ---- Intramodal: adjacent sizes within a family ----
    for modality in MODALITIES:
        for alias, sizes in FAMILY_SIZES.items():
            for s1, s2 in zip(sizes, sizes[1:]):
                t1 = (modality, alias, s1)
                t2 = (modality, alias, s2)
                if t1 not in nn_cache or t2 not in nn_cache:
                    continue
                nn1 = nn_cache[t1]
                nn2 = nn_cache[t2]
                nrow = min(len(nn1), len(nn2))
                nn1c, nn2c = nn1[:nrow], nn2[:nrow]
                row = {
                    "pair_kind":   "intramodal",
                    "modality_a":  modality,
                    "modality_b":  modality,
                    "model_alias": alias,
                    "size_a":      s1,
                    "size_b":      s2,
                    "n":           int(nrow),
                    "k":           int(nn1.shape[1]),
                    "mknn":        mknn_overlap(nn1c, nn2c),
                }
                for prop, y in catalog.items():
                    row[f"wass_{prop}"] = wasserstein_per_property(
                        nn1c, nn2c, y[:nrow])
                rows.append(row)

    # ---- Crossmodal: same model size, both modalities ----
    for alias, sizes in FAMILY_SIZES.items():
        for size in sizes:
            t_h = ("hsc",  alias, size)
            t_j = ("jwst", alias, size)
            if t_h not in nn_cache or t_j not in nn_cache:
                continue
            nn1 = nn_cache[t_h]
            nn2 = nn_cache[t_j]
            nrow = min(len(nn1), len(nn2))
            nn1c, nn2c = nn1[:nrow], nn2[:nrow]
            row = {
                "pair_kind":   "crossmodal",
                "modality_a":  "hsc",
                "modality_b":  "jwst",
                "model_alias": alias,
                "size_a":      size,
                "size_b":      size,
                "n":           int(nrow),
                "k":           int(nn1.shape[1]),
                "mknn":        mknn_overlap(nn1c, nn2c),
            }
            for prop, y in catalog.items():
                row[f"wass_{prop}"] = wasserstein_per_property(
                    nn1c, nn2c, y[:nrow])
            rows.append(row)

    if not rows:
        print("[fatal] no pairs computed (no neighbour matrices loaded?)",
              file=sys.stderr)
        return 1

    df = pl.DataFrame(rows)
    out_pq = args.out_dir / "regress_pairs.parquet"
    df.write_parquet(out_pq, compression="zstd")
    print(f"wrote {out_pq}  ({len(df)} pairs)")
    print()
    summary = (df.group_by("pair_kind")
                  .agg([pl.col("mknn").mean().alias("mknn_mean"),
                        pl.col("mknn").min().alias("mknn_min"),
                        pl.col("mknn").max().alias("mknn_max"),
                        pl.len().alias("n_pairs")]))
    print(summary)

    if args.upload_to:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.upload_file(
            path_or_fileobj=str(out_pq),
            path_in_repo="done/summary.pairs.parquet",
            repo_id=args.upload_to,
            repo_type="dataset",
            commit_message="pu_regress_pairs: refresh pairwise summary",
        )
        print(f"uploaded -> {args.upload_to}/done/summary.pairs.parquet")

    return 0


if __name__ == "__main__":
    sys.exit(main())
