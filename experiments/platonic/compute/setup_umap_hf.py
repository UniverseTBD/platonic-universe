#!/usr/bin/env python3
"""One-time bootstrap of the shared HF dataset used by pu_umap.py for
cross-cluster coordination of the UMAP/kNN-purity phase.

After all pu_solve outputs land, run this once on a laptop, then upload
candidates.parquet + the four labels_*.parquet files into static/ via
upload_candidates_and_labels() below.

Usage:
    HF_TOKEN=hf_xxx python pu_umap_setup_hf.py <owner>/pu-umap-results
"""
import os
import sys


def main():
    if len(sys.argv) != 2:
        print("usage: HF_TOKEN=... python pu_umap_setup_hf.py "
              "<owner>/<dataset_name>", file=sys.stderr)
        sys.exit(2)

    repo_id = sys.argv[1]
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[fatal] HF_TOKEN env var not set", file=sys.stderr)
        sys.exit(1)

    from huggingface_hub import HfApi, create_repo
    api = HfApi(token=token)

    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
        token=token,
    )
    print(f"[ok] dataset ready: https://huggingface.co/datasets/{repo_id}")

    for placeholder in ("done/.gitkeep", "running/.gitkeep", "static/.gitkeep"):
        api.upload_file(
            path_or_fileobj=b"",
            path_in_repo=placeholder,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"bootstrap {placeholder}",
        )
        print(f"[ok] uploaded {placeholder}")

    print()
    print("Next:")
    print("  1. Run 01_select_candidates.py once all pu_solve jobs are done.")
    print("  2. Upload static inputs:")
    print(f"       python -c \"from pu_umap_setup_hf import upload_candidates_and_labels;"
          f" upload_candidates_and_labels('{repo_id}')\"")
    print(f"  3. export PU_UMAP_RESULTS_REPO={repo_id}")
    print("  4. sbatch pu_umap.sh   # on each cluster")


def upload_candidates_and_labels(repo_id):
    """Upload candidates.parquet + labels_*.parquet to static/ on the dataset."""
    import os

    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set")
    api = HfApi(token=token)

    HERE = os.path.dirname(os.path.abspath(__file__))
    files = [
        ("derived/candidates.parquet",          "static/candidates.parquet"),
        ("labels/labels_desi.parquet",          "static/labels_desi.parquet"),
        ("labels/labels_jwst.parquet",          "static/labels_jwst.parquet"),
        ("labels/labels_legacysurvey.parquet",  "static/labels_legacysurvey.parquet"),
        ("labels/labels_sdss.parquet",          "static/labels_sdss.parquet"),
    ]
    for local_rel, remote in files:
        local = os.path.join(HERE, "platonic-universe", local_rel)
        if not os.path.exists(local):
            local = os.path.join(HERE, local_rel)
        if not os.path.exists(local):
            print(f"[skip] {remote}: local file not found")
            continue
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"static input: {remote}",
        )
        print(f"[ok] uploaded {remote}  ({os.path.getsize(local)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
