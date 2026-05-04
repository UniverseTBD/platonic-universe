"""Reproduce the vjepa Arrow/OOM failure on a small slice of the real
Smith42/legacysurvey_hsc_crossmatched dataset, and verify this PR fixes it.

Runs the exact flow run_extraction.py uses: stream a slice, materialize
to an in-memory Dataset, apply HFCrossmatchedAdapter's .map(processor),
feed through a DataLoader, run one forward pass through vjepa-large.

Per-sample Arrow bytes:
    pre-PR:  16 × 3 × 256 × 256 × 4 B = 12.58 MB
    post-PR:  1 × 3 × 256 × 256 × 4 B =  0.79 MB

At N=300 (two modes cached), pre-PR accumulates ~7.5 GB of Arrow data
during .map() and is OOM-killed on a 31 GB box; post-PR fits in ~500 MB
and completes cleanly.

Observed locally (31 GB RAM, vjepa2-vitl):
    pre-PR  N=300 → exit 137 (SIGKILL) during [4/4] .map()
    post-PR N=300 → SUCCESS, ~7 min, hsc batch shape (2, 1, 3, 256, 256)

Usage:
    python scripts/vjepa_real_slice_repro.py <N>
"""
import os
import sys
import time
import traceback

# Point HF caches at /tmp (189 GB free on this box; ~/.cache only has ~80 GB).
_TMP = "/tmp/pu_slice_hf_cache"
os.environ["HF_HOME"] = _TMP
os.environ["HF_HUB_CACHE"] = f"{_TMP}/hub"
os.environ["HF_DATASETS_CACHE"] = f"{_TMP}/datasets"
os.makedirs(_TMP, exist_ok=True)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(_REPO_ROOT)  # data/percentiles.json is loaded by path
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from pu.models import get_adapter

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
MODEL_ALIAS = "vjepa"
SIZE = "large"
HF_NAME = "facebook/vjepa2-vitl-fpc64-256"
HF_DS = "Smith42/legacysurvey_hsc_crossmatched"


def main():
    print(f"N = {N}")
    print(f"commit: ", end="", flush=True)
    os.system("git -C /home/me/Desktop/platonic-universe rev-parse --short HEAD")

    print(f"\n[1/4] streaming first {N} rows from {HF_DS} ...")
    t0 = time.time()
    stream = load_dataset(HF_DS, split="train", streaming=True)
    # Drop the ~70 metadata columns up front — we only need the two image
    # columns — so Dataset.from_list doesn't choke serializing nested Arrow.
    stream = stream.select_columns(["hsc_image", "legacysurvey_image"])
    rows = []
    for i, row in enumerate(tqdm(stream.take(N), total=N, desc="stream")):
        rows.append(row)
        if i + 1 >= N:
            break
    print(f"  streamed {len(rows)} rows in {time.time()-t0:.1f} s")

    print(f"\n[2/4] materializing to in-memory Dataset ...")
    t0 = time.time()
    ds_mem = Dataset.from_list(rows)
    print(f"  done in {time.time()-t0:.1f} s — columns: {ds_mem.column_names}")

    print(f"\n[3/4] loading {MODEL_ALIAS} {SIZE} ...")
    adapter_cls = get_adapter(MODEL_ALIAS)
    adapter = adapter_cls(HF_NAME, SIZE, alias=MODEL_ALIAS)
    adapter.load()
    print(f"  hookable modules: {adapter.get_num_layers()}")

    print(f"\n[4/4] running HFCrossmatchedAdapter-style .map() + DataLoader ...")
    modes = ["hsc", "legacysurvey"]
    processor = adapter.get_preprocessor(modes, resize=True, resize_mode="match")
    t0 = time.time()
    ds = (
        ds_mem.select_columns([f"{m}_image" for m in modes])
              .map(processor)
              .remove_columns([f"{m}_image" for m in modes])
    )
    ds = ds.with_format("torch")
    print(f"  .map() completed in {time.time()-t0:.1f} s")

    dl = DataLoader(ds, batch_size=2, num_workers=0)
    with torch.no_grad():
        for i, batch in enumerate(dl):
            out = adapter.embed_all_layers_for_mode(batch, "hsc", granularity="blocks")
            if i == 0:
                print(f"  first batch OK — {len(out)} cols, hsc shape {tuple(batch['hsc'].shape)}")
            if i >= 2:
                break
    print("\nSUCCESS")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
