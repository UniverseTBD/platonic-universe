#!/usr/bin/env python3
"""Stream parquet → .npy conversion for the COSMOS-Web embeddings hosted at
``UniverseTBD/pu-embeddings/cosmosweb/``.

Each parquet is downloaded one at a time, converted to the .npy file name
expected by ``probe_weight_analysis.py``, and the parquet is deleted before
the next download. Peak local disk usage during conversion is bounded by
the largest single embedding (~1 GB for llava_15_13b / paligemma_28b).

Output layout:
  <OUT_DIR>/<telescope>_embeddings_<DS_TAG>_<alias>_<size>_<N_USE>.npy

Configuration via env vars (defaults reproduce the paper's basket):
  STREAM_REPO          UniverseTBD/pu-embeddings  (HF dataset id)
  STREAM_FOLDER        cosmosweb                  (sub-folder in the repo)
  STREAM_DS_TAG        cosmosweb-hsc-jwst-high-snr-pil2
  STREAM_N_USE         45000
  STREAM_OUT_DIR       analysis/probe_smoke/embeddings
  STREAM_TMP_DIR       /tmp/pu_emb_stream         (per-file scratch)
  STREAM_SUBSET        ""                         (comma-separated <fam>:<size>
                                                   pairs to limit; empty = all)

Usage:
  PWA_DATASET=Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2 \
  python experiments/cosmosweb/stream_embeddings_to_npy.py
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import polars as pl
from huggingface_hub import hf_hub_download

# Full basket — same as probe_weight_analysis.py
DEFAULT_MODEL_MAP = {
    "vit":       ["base", "large", "huge"],
    "clip":      ["base", "large"],
    "dinov3":    ["vits16", "vits16plus", "vitb16", "vitl16",
                  "vith16plus", "vit7b16"],
    "convnext":  ["nano", "tiny", "base", "large"],
    "ijepa":     ["huge", "giant"],
    "vjepa":     ["large", "huge", "giant"],
    "astropt":   ["015M", "095M", "850M"],
    "vit-mae":   ["base", "large", "huge"],
    "paligemma": ["3b", "10b", "28b"],
    "llava_15":  ["7b", "13b"],
}
TELESCOPES = ("hsc", "jwst")


def parse_subset(s: str) -> list[tuple[str, str]] | None:
    """Parse comma-separated <fam>:<size> pairs. Empty → None (= use all)."""
    s = (s or "").strip()
    if not s:
        return None
    out = []
    for tok in s.split(","):
        fam, size = tok.split(":")
        out.append((fam.strip(), size.strip()))
    return out


def main() -> int:
    repo     = os.environ.get("STREAM_REPO",   "UniverseTBD/pu-embeddings")
    folder   = os.environ.get("STREAM_FOLDER", "cosmosweb")
    ds_tag   = os.environ.get("STREAM_DS_TAG", "cosmosweb-hsc-jwst-high-snr-pil2")
    n_use    = int(os.environ.get("STREAM_N_USE", "45000"))
    out_dir  = Path(os.environ.get("STREAM_OUT_DIR",
                                    "analysis/probe_smoke/embeddings"))
    tmp_dir  = Path(os.environ.get("STREAM_TMP_DIR", "/tmp/pu_emb_stream"))
    subset   = parse_subset(os.environ.get("STREAM_SUBSET", ""))

    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = subset or [(fam, size) for fam, sizes in DEFAULT_MODEL_MAP.items()
                                    for size in sizes]
    print(f"[stream] {len(pairs)} (family, size) pairs × {len(TELESCOPES)} "
          f"telescopes = {len(pairs)*len(TELESCOPES)} files")
    print(f"[stream] repo={repo}  folder={folder}  out={out_dir}")

    for fam, size in pairs:
        for tele in TELESCOPES:
            target = out_dir / (
                f"{tele}_embeddings_{ds_tag}_{fam}_{size}_{n_use}.npy"
            )
            if target.exists():
                print(f"  [skip] {target.name}")
                continue
            rel = f"{folder}/{tele}_embeddings_{ds_tag}_{fam}_{size}_{n_use}.parquet"
            # Clean per-file scratch to avoid stacking
            shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            try:
                local = hf_hub_download(repo, rel, repo_type="dataset",
                                         local_dir=str(tmp_dir))
            except Exception as e:
                print(f"  [warn] {rel}: {type(e).__name__}: {e}")
                continue
            df = pl.read_parquet(local)
            col = "embeddings" if "embeddings" in df.columns else df.columns[0]
            arr = np.stack([np.asarray(v, dtype=np.float32)
                             for v in df[col].to_list()])
            np.save(target, arr)
            print(f"  {target.name}  shape={arr.shape}  ({arr.nbytes/1e6:.0f} MB)")
            shutil.rmtree(tmp_dir, ignore_errors=True)

    n_npy = len(list(out_dir.glob("*.npy")))
    print(f"\n[stream] {n_npy} .npy files in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
