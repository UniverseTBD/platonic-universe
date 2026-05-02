#!/usr/bin/env python3
"""Pull every done/*.parquet from the regression coordination dataset
and concatenate into a single parquet (and optionally a JSON in the
schema consumed by Smith42's plotting branch).

Usage:
    python aggregate.py \\
        --pull-from <owner>/pu-regress-results \\
        --out-dir   /path/to/derived
        [--json-out /path/to/r2_vs_params.json]

Env defaults:
    PU_REGRESS_RESULTS_REPO    used if --pull-from absent
    HF_TOKEN                   required for any HF interaction
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--pull-from", default=os.environ.get("PU_REGRESS_RESULTS_REPO"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--json-out", type=Path, default=None,
                   help="Optional path to also write a {modality:{model:{size:{prop:{r2_mean,r2_std}}}}} JSON.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.pull_from:
        print("[fatal] --pull-from required (or set $PU_REGRESS_RESULTS_REPO)",
              file=sys.stderr)
        return 2

    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    files = [f for f in api.list_repo_files(args.pull_from, repo_type="dataset")
             if f.startswith("done/") and f.endswith(".parquet")]
    if not files:
        print(f"[fatal] no done/*.parquet in {args.pull_from}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.out_dir / "_regress_cache"
    cache.mkdir(exist_ok=True)
    print(f"pulling {len(files)} parquets from {args.pull_from}")
    for f in files:
        hf_hub_download(args.pull_from, f, repo_type="dataset",
                        local_dir=str(cache), token=token)

    locals_ = sorted((cache / "done").glob("*.parquet"))
    df = pl.concat([pl.read_parquet(p) for p in locals_])
    print(f"merged: {len(df):,} rows, "
          f"{df.select(['modality','model_alias','model_size']).unique().height} tuples")

    out_pq = args.out_dir / "regress_summary.parquet"
    df.write_parquet(out_pq)
    print(f"wrote {out_pq}")

    if args.json_out:
        # Smith42-compatible nested layout:
        # { modality: { model_alias: { size: { property: { r2_mean, r2_std } } } } }
        nested: dict = {}
        for row in df.iter_rows(named=True):
            entry = (nested.setdefault(row["modality"], {})
                            .setdefault(row["model_alias"], {})
                            .setdefault(row["model_size"], {})
                            .setdefault(row["property"], {}))
            entry["r2_mean"] = row["r2_mean"]
            if "r2_std" in row:
                entry["r2_std"] = row["r2_std"]
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(nested, f, indent=2, default=float)
        print(f"wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
