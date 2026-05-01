#!/usr/bin/env python3
"""One-time setup of the shared HF dataset used by pu_solve.py for
cross-cluster coordination.

Run once on either of the two clusters (or your laptop) to create a
private dataset both clusters will write to. After it exists, either
side can begin queueing pu_solve.py jobs with PU_SOLVE_RESULTS_REPO
pointing at this dataset.

Usage:
    HF_TOKEN=hf_xxx python pu_solve_setup_hf.py <owner>/pu-solve-results
"""
import os
import sys


def main():
    if len(sys.argv) != 2:
        print("usage: HF_TOKEN=... python pu_solve_setup_hf.py "
              "<owner>/<dataset_name>", file=sys.stderr)
        sys.exit(2)

    repo_id = sys.argv[1]
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[fatal] HF_TOKEN env var not set", file=sys.stderr)
        sys.exit(1)

    from huggingface_hub import HfApi, create_repo
    api = HfApi(token=token)

    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=True,
            token=token,
        )
        print(f"[ok] dataset ready: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"[fatal] failed to create/access {repo_id}: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    # Bootstrap with placeholder folders by uploading two empty marker files.
    # HF dataset listings show files, not directories, so this gives both
    # clusters something to list against on their first call.
    for placeholder in ("done/.gitkeep", "running/.gitkeep"):
        try:
            api.upload_file(
                path_or_fileobj=b"",
                path_in_repo=placeholder,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"bootstrap {placeholder}",
            )
            print(f"[ok] uploaded {placeholder}")
        except Exception as e:
            print(f"[warn] {placeholder}: {type(e).__name__}: {e}")

    print()
    print("Next:")
    print(f"  export PU_SOLVE_RESULTS_REPO={repo_id}")
    print("  sbatch pu_solve.sh   # on each cluster")
    print()
    print("To grant your coauthor write access:")
    print(f"  https://huggingface.co/datasets/{repo_id}/settings")
    print("  → Members → invite their HF username with `write` permission.")


if __name__ == "__main__":
    main()
