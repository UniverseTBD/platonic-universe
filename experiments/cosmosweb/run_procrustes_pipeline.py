#!/usr/bin/env python3
"""End-to-end runner: Hugging Face embeddings → Procrustes pkl → paper figures.

Steps:
  1. Stream-convert UniverseTBD/pu-embeddings/cosmosweb/*.parquet to .npy.
  2. Run probe_weight_analysis.py to fit linear probes, compute the 3×3
     cosine-similarity matrices, and save the per-(family, size) Procrustes
     distance pickle.
  3. Optionally drop the pkl into a separate worktree (Mike's plotting branch)
     so plot_*_procrustes.py can render the paper figures.

Configuration via env vars (sensible defaults):
  RPP_OUT_DIR         analysis/procrustes_pipeline   (root of outputs)
  RPP_SKIP_STREAM     "0"  → "1" to skip download/conversion (.npy exist)
  RPP_PLOT_WORKTREE   ""   → optional path of a checkout containing
                              scripts/plot_*_procrustes.py + data/

The probe_weight_analysis.py script reads the same OUT_DIR via env vars
PWA_*; this runner sets them so the two scripts agree on layout.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

DATASET = "Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2"
DS_TAG  = "cosmosweb-hsc-jwst-high-snr-pil2"
N_USE   = 45000

ROOT     = Path(__file__).resolve().parent.parent.parent
OUT_DIR  = Path(os.environ.get("RPP_OUT_DIR",
                                ROOT / "analysis" / "procrustes_pipeline"))
EMB_DIR  = OUT_DIR / "embeddings"
SKIP_STREAM   = os.environ.get("RPP_SKIP_STREAM", "0") == "1"
PLOT_WORKTREE = os.environ.get("RPP_PLOT_WORKTREE", "").strip()


def step_stream() -> None:
    if SKIP_STREAM:
        print("[1/3] stream — skipped (RPP_SKIP_STREAM=1)")
        return
    print("[1/3] stream parquets → .npy")
    env = os.environ.copy()
    env.update(
        STREAM_DS_TAG=DS_TAG,
        STREAM_N_USE=str(N_USE),
        STREAM_OUT_DIR=str(EMB_DIR),
    )
    rc = subprocess.call(
        [sys.executable, str(Path(__file__).parent / "stream_embeddings_to_npy.py")],
        env=env,
    )
    if rc != 0:
        raise SystemExit(f"stream step failed with rc={rc}")


def step_probe_weight_analysis() -> None:
    print("[2/3] probe_weight_analysis.py → Procrustes pkl + cosine PDFs")
    env = os.environ.copy()
    env.update(
        PWA_DATASET=DATASET,
        PWA_OUT_DIR=str(OUT_DIR),
        PWA_EMB_DIR=str(EMB_DIR),
        PWA_N_USE=str(N_USE),
        # Embeddings on UniverseTBD/pu-embeddings are NOT the upsampled flavour.
        PWA_UPSAMPLE_SUFFIX="",
    )
    rc = subprocess.call(
        [sys.executable, str(Path(__file__).parent / "probe_weight_analysis.py")],
        env=env,
    )
    if rc != 0:
        # The script may exit non-zero on a late cosmetic plot bug while still
        # having produced the pkl. Treat missing pkl as fatal.
        pass
    pkl = OUT_DIR / f"procrustes_distances_{N_USE}galaxies.pkl"
    if not pkl.exists():
        raise SystemExit(f"expected {pkl} not produced — check upstream errors")
    print(f"  produced {pkl}  ({pkl.stat().st_size//1024} KB)")


def step_publish_to_worktree() -> None:
    if not PLOT_WORKTREE:
        print("[3/3] worktree drop — skipped (RPP_PLOT_WORKTREE not set)")
        return
    src = OUT_DIR / f"procrustes_distances_{N_USE}galaxies.pkl"
    dst_dir = Path(PLOT_WORKTREE) / "data"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"procrustes_distances_{N_USE}galaxies_upsampled.pkl"
    shutil.copy2(src, dst)
    print(f"[3/3] copied {src.name} → {dst}")
    print("       run scripts/plot_*_procrustes.py from that worktree to render")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    step_stream()
    step_probe_weight_analysis()
    step_publish_to_worktree()
    print("\n[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
