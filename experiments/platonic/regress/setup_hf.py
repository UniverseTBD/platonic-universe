#!/usr/bin/env python3
"""One-time bootstrap of the shared HF dataset used by pu_regress.py for
cross-cluster coordination.

Run once on any machine with HF_TOKEN set; creates a private dataset
with `done/` and `running/` placeholder folders. After it exists, any
number of workers can claim and run tuples concurrently.

Usage:
    HF_TOKEN=hf_xxx python setup_hf.py <owner>/<dataset_name>
"""
import os
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: HF_TOKEN=... python setup_hf.py <owner>/<dataset_name>",
              file=sys.stderr)
        return 2
    repo_id = sys.argv[1]
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[fatal] HF_TOKEN env var not set", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi, create_repo
    api = HfApi(token=token)
    create_repo(repo_id=repo_id, repo_type="dataset",
                private=True, exist_ok=True, token=token)
    print(f"[ok] dataset ready: https://huggingface.co/datasets/{repo_id}")

    for placeholder in ("done/.gitkeep", "running/.gitkeep"):
        api.upload_file(path_or_fileobj=b"", path_in_repo=placeholder,
                        repo_id=repo_id, repo_type="dataset",
                        commit_message=f"bootstrap {placeholder}")
        print(f"[ok] {placeholder}")

    print()
    print("Next:")
    print(f"  export PU_REGRESS_RESULTS_REPO={repo_id}")
    print("  python pu_regress.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
