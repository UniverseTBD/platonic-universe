"""Re-extract layer-wise embeddings from raw images for the paper.

Iterates over a (model × survey) grid and uses the existing pu adapter
registry to compute layer-wise activations for every block. Output is one
parquet per (model, survey) tuple, with one column per layer (lists of
floats per row), matching the schema consumed by `compute/pu_solve.py`.

Designed for batch GPU runs. Total wall time scales with the number of
(model × survey) pairs and the size of each model.

Usage:
    python extract_layerwise.py \\
        --models  vit_base dino_giant convnext_large \\
        --surveys jwst legacysurvey desi sdss \\
        --out-dir ./embeddings_out \\
        [--upload-to <owner>/<dataset_repo>]

Env vars:
    HF_TOKEN   required if any source dataset is gated.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_MODELS = [
    # vision-only families: vit, dino, dinov3, convnext, vit-mae, ijepa, vjepa,
    # astropt. vision-language: paligemma, llava-1.5. List the full set of
    # adapters you have registered in pu/models for full reproduction.
]
DEFAULT_SURVEYS = ["jwst", "legacysurvey", "desi", "sdss"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--models",  nargs="+", default=DEFAULT_MODELS,
                   help="Model adapter aliases (must be registered in pu.models).")
    p.add_argument("--surveys", nargs="+", default=DEFAULT_SURVEYS,
                   help="Dataset adapter aliases (must be registered in pu.pu_datasets).")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Where to write per-tuple <survey>/<model>_blocks_layerwise.parquet")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--upload-to", default=None,
                   help="If set, upload each parquet to this HF dataset repo.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip (model, survey) tuples whose output parquet already exists locally.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.models:
        print("[fatal] --models must be non-empty. Use the aliases registered "
              "in pu/models. See `python -c 'from pu.models import list_adapters; "
              "print(list_adapters())'`.", file=sys.stderr)
        return 2

    # Defer the heavy imports so --help is fast.
    from pu.experiments import run_layerwise_extraction
    args.out_dir.mkdir(parents=True, exist_ok=True)

    api = None
    if args.upload_to:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))

    n_total = len(args.models) * len(args.surveys)
    n_done = 0
    for survey in args.surveys:
        for model in args.models:
            n_done += 1
            tag = f"{survey}/{model}_blocks_layerwise.parquet"
            local = args.out_dir / tag
            if args.skip_existing and local.exists():
                print(f"[skip {n_done}/{n_total}] {tag} already present")
                continue

            print(f"[extract {n_done}/{n_total}] {tag}")
            local.parent.mkdir(parents=True, exist_ok=True)
            try:
                run_layerwise_extraction(
                    model_alias=model,
                    survey_alias=survey,
                    output_path=local,
                    batch_size=args.batch_size,
                )
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
                continue

            if api is not None:
                api.upload_file(
                    path_or_fileobj=str(local),
                    path_in_repo=tag,
                    repo_id=args.upload_to,
                    repo_type="dataset",
                    commit_message=f"layerwise embeddings: {tag}",
                )
                print(f"  uploaded to {args.upload_to}/{tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
