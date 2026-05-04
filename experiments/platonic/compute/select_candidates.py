"""Aggregate Phase-1 solve outputs and pick the top-K (model, survey, block)
triples worth UMAPing in Phase 2.

Reads per-(survey, model) parquets either from a local directory or by
pulling `done/*.parquet` from a Hugging Face coordination dataset. Writes
two derived parquets locally and optionally uploads them to a Phase-2
coordination dataset under `static/`.

Usage:
    # local-only (you've already downloaded done/*.parquet somewhere):
    python select_candidates.py \\
        --solve-dir /path/to/solve_out/done \\
        --out-dir /path/to/derived

    # pull from a coordination dataset on HF and aggregate:
    python select_candidates.py \\
        --pull-from <owner>/pu-solve-results \\
        --out-dir /path/to/derived

    # ... and upload candidates.parquet to the Phase-2 dataset:
    python select_candidates.py \\
        --pull-from <owner>/pu-solve-results \\
        --out-dir /path/to/derived \\
        --upload-to <owner>/pu-umap-results

Env:
    HF_TOKEN                  required if any HF interaction
    PU_SOLVE_RESULTS_REPO     default for --pull-from
    PU_UMAP_RESULTS_REPO      default for --upload-to
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--solve-dir", type=Path, default=None,
                   help="Local dir containing done/*.parquet "
                        "(if omitted, --pull-from must be given).")
    p.add_argument("--pull-from", default=os.environ.get("PU_SOLVE_RESULTS_REPO"),
                   help="HF dataset id to pull done/*.parquet from. "
                        "Defaults to $PU_SOLVE_RESULTS_REPO.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Local dir for derived/{candidates,per_block}.parquet.")
    p.add_argument("--upload-to", default=None,
                   help="If set, upload candidates.parquet to this HF dataset's "
                        "static/ folder. Defaults to $PU_UMAP_RESULTS_REPO if "
                        "the env var is set and this flag is absent.")
    p.add_argument("--top-k", type=int, default=3,
                   help="Top-K HSC-side blocks per (survey, model) to keep.")
    return p.parse_args()


def fetch_solve_dir(repo_id: str, dest: Path) -> Path:
    """Pull every done/*.parquet from an HF dataset to `dest`. Returns dest."""
    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    files = [f for f in api.list_repo_files(repo_id, repo_type="dataset")
             if f.startswith("done/") and f.endswith(".parquet")]
    if not files:
        raise SystemExit(f"[fatal] no done/*.parquet found in {repo_id}")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"pulling {len(files)} parquets from {repo_id} -> {dest}")
    for f in files:
        hf_hub_download(repo_id, f, repo_type="dataset",
                        local_dir=str(dest), token=token)
    return dest / "done"


def main() -> int:
    args = parse_args()

    if args.solve_dir is None and args.pull_from is None:
        print("[fatal] specify --solve-dir or --pull-from "
              "(or set $PU_SOLVE_RESULTS_REPO).", file=sys.stderr)
        return 2

    solve_dir: Path
    if args.solve_dir is not None:
        solve_dir = args.solve_dir
    else:
        cache = args.out_dir / "_solve_cache"
        solve_dir = fetch_solve_dir(args.pull_from, cache)

    paths = sorted(glob.glob(str(solve_dir / "*.parquet")))
    if not paths:
        print(f"[fatal] no parquets at {solve_dir}", file=sys.stderr)
        return 1
    print(f"{len(paths)} solve parquets")

    df = pl.concat([pl.read_parquet(p) for p in paths])
    print(f"total rows: {len(df):,}")
    print(f"surveys: {df['survey'].unique().to_list()}")
    print(f"models : {sorted(df['model'].unique().to_list())}")

    shape = (df.filter(pl.col("metric").str.starts_with("shape_"))
               .with_columns(side=pl.col("block_b_name"))
               .select(["survey", "model", "side", "block_a_idx",
                        "metric", "score"])
               .pivot(values="score",
                      index=["survey", "model", "side", "block_a_idx"],
                      on="metric"))

    align = df.filter(pl.col("metric") == "cka")

    plat = (align.group_by(["survey", "model", "block_a_idx", "block_a_name"])
                  .agg(mean_cka=pl.col("calibrated_score").mean(),
                       max_cka=pl.col("calibrated_score").max(),
                       n_pairs=pl.len()))

    hsc_shape = shape.filter(pl.col("side") == "hsc").drop("side")
    joined = plat.join(hsc_shape,
                       on=["survey", "model", "block_a_idx"],
                       how="left")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_block = args.out_dir / "per_block.parquet"
    candidates = args.out_dir / "candidates.parquet"

    joined.write_parquet(per_block)
    print(f"\n[per_block.parquet] {len(joined):,} rows -> {per_block}")

    cand = (joined.sort("mean_cka", descending=True)
                  .group_by(["survey", "model"], maintain_order=True)
                  .head(args.top_k))
    cand.write_parquet(candidates)
    print(f"[candidates.parquet] {len(cand):,} rows "
          f"(top-{args.top_k} per (survey, model)) -> {candidates}")

    upload_to = args.upload_to or os.environ.get("PU_UMAP_RESULTS_REPO")
    if upload_to:
        from huggingface_hub import HfApi
        token = os.environ.get("HF_TOKEN")
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(candidates),
            path_in_repo="static/candidates.parquet",
            repo_id=upload_to,
            repo_type="dataset",
            commit_message="select_candidates.py: refresh static/candidates.parquet",
        )
        print(f"\nuploaded -> {upload_to}/static/candidates.parquet")

    return 0


if __name__ == "__main__":
    sys.exit(main())
