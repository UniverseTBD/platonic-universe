#!/usr/bin/env python3
"""Merge partial pu_solve outputs back into a canonical task parquet.

When pu_solve.py is run with PU_SOLVE_PAIR_RANGE=start:end, it writes
<survey>__<model>.partial.<start>-<end>.parquet instead of the canonical
<survey>__<model>.parquet. After all partitions for a task have completed,
run this script to concatenate them.

Usage:
    python merge_partials.py <survey> <model> [--out-dir DIR]
                                              [--upload-to-hf REPO]
                                              [--keep-partials]

Example:
    python merge_partials.py legacysurvey paligemma_28b_28b \
        --out-dir ./solve_out \
        --upload-to-hf <owner>/pu-solve-results
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import polars as pl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("survey")
    ap.add_argument("model")
    ap.add_argument("--out-dir", default=os.environ.get("PU_SOLVE_OUT", "./solve_out"))
    ap.add_argument("--upload-to-hf", default="",
                    help="If set, upload the merged parquet to this HF dataset under done/")
    ap.add_argument("--keep-partials", action="store_true",
                    help="Don't delete partials after merging.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    tag = f"{args.survey}__{args.model}"
    final = out_dir / f"{tag}.parquet"
    partials = sorted(out_dir.glob(f"{tag}.partial.*.parquet"))

    if not partials:
        print(f"[fatal] no partials found for {tag} in {out_dir}", file=sys.stderr)
        return 1
    print(f"merging {len(partials)} partials for {tag}:")
    for p in partials:
        print(f"  - {p.name}  ({p.stat().st_size / 1e6:.2f} MB)")

    dfs = [pl.read_parquet(p) for p in partials]
    merged = pl.concat(dfs, how="vertical_relaxed")

    # Pair-alignment rows can be deduped by (block_a_idx, block_b_idx, metric).
    # Shape rows are produced redundantly across partitions (any partition that
    # touches a column writes its shape row). Dedupe by (block_a_idx, side, metric).
    pair_mask = ~pl.col("metric").str.starts_with("shape_")
    pair_df  = (merged.filter(pair_mask)
                       .unique(subset=["block_a_idx", "block_b_idx", "metric"],
                               keep="first"))
    shape_df = (merged.filter(~pair_mask)
                       .unique(subset=["block_a_idx", "block_b_name", "metric"],
                               keep="first"))
    final_df = pl.concat([pair_df, shape_df], how="vertical_relaxed")

    print(f"\nrow counts: pair={len(pair_df)}, shape={len(shape_df)}, "
          f"total={len(final_df)}")
    final_df.write_parquet(final, compression="zstd")
    print(f"wrote {final}  ({final.stat().st_size / 1e6:.2f} MB)")

    if args.upload_to_hf:
        from huggingface_hub import HfApi
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("[warn] HF_TOKEN not set; skipping upload")
        else:
            api = HfApi(token=token)
            api.upload_file(
                path_or_fileobj=str(final),
                path_in_repo=f"done/{tag}.parquet",
                repo_id=args.upload_to_hf,
                repo_type="dataset",
                commit_message=f"merged partial outputs for {tag}",
            )
            print(f"uploaded to {args.upload_to_hf}/done/{tag}.parquet")
            # Also delete the running marker if present (tolerant of absence).
            try:
                api.delete_file(
                    path_in_repo=f"running/{tag}.running",
                    repo_id=args.upload_to_hf,
                    repo_type="dataset",
                    commit_message=f"release {tag} after merge",
                )
                print(f"removed running/{tag}.running")
            except Exception as e:
                print(f"[note] couldn't remove running marker: {type(e).__name__}: {e}")

    if not args.keep_partials:
        for p in partials:
            p.unlink()
        print(f"deleted {len(partials)} partial files")

    return 0


if __name__ == "__main__":
    sys.exit(main())
