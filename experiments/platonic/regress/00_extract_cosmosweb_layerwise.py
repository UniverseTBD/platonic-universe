#!/usr/bin/env python3
"""Re-extract cosmosweb layerwise embeddings in guaranteed row order.

The externally-provided ``data/cosmosweb/paligemma_3b_3b_blocks_layerwise.parquet``
had globally scrambled rows (extractor evidently ran with num_workers>0 on a
streaming dataset), so joining physics labels by index gave meaningless probes
(R2 pinned at the -d/n_train "random labels" floor for every layer/property,
including mag_r which a vision layer normally predicts at R2~=0.9).

This re-extracts with **num_workers=0** streaming, so row i corresponds to
catalog row i. It reuses:
  - the paper's cosmosweb preprocessing (``make_preprocessor`` from step 1:
    upsample -> 224 -> HF processor with the "<image> " prompt), and
  - the adapter's ``embed_all_layers_for_mode`` (blocks granularity).
``hidden_states_last`` is the masked-mean final decoder state, bit-identical to
``embed_for_mode`` -> the paper's published embedding.

Output columns match the probe's parser: ``{layer_key}_{hsc,jwst}`` e.g.
``model.vision_tower.encoder.layers.0_hsc``, ``model.language_model.layers.0_hsc``,
``hidden_states_last_hsc`` (+ ``_jwst``).

Env
---
  PU_EXTRACT_MODEL  adapter alias           (default paligemma_3b)
  PU_EXTRACT_HFID   hf model id             (default google/paligemma2-3b-mix-224)
  PU_EXTRACT_SIZE   size tag                (default 3b)
  PU_EXTRACT_N      rows to extract         (default 45000, streaming order)
  PU_EXTRACT_BATCH  batch size              (default 32)
  PU_EXTRACT_LIMIT  cap rows for a smoke test of columns/shapes (0 = off)
  PU_EXTRACT_OUT    output parquet
"""
from __future__ import annotations

import datetime
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASET = "Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2"
MODES = ("hsc", "jwst")


def log(*a):
    print(datetime.datetime.now().strftime("[%H:%M:%S]"), *a, flush=True)


def _load_step1():
    """Import 01_extract_and_probe.py by path (digit-led module name) for its
    cosmosweb ``make_preprocessor``; default its HF-pipeline env vars so import
    has no side effects."""
    tmp = Path(tempfile.gettempdir())
    # Real token (env HF_TOKEN, else the cached CLI login) — the model is gated,
    # so a bogus placeholder here would poison transformers auth (401).
    from huggingface_hub import get_token
    real_token = get_token()
    for k, v in {
        "HF_TOKEN": real_token or "unused",
        "PU_REGRESS_RESULTS_REPO": "unused/unused",
        "PU_REGRESS_OUT": str(tmp / "pu_lw_out"),
        "PU_REGRESS_LOCKS": str(tmp / "pu_lw_locks"),
    }.items():
        os.environ.setdefault(k, v)
    spec = importlib.util.spec_from_file_location(
        "pu_regress_step1", HERE / "01_extract_and_probe.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    step1 = _load_step1()
    alias = os.environ.get("PU_EXTRACT_MODEL", "paligemma_3b")
    hfid = os.environ.get("PU_EXTRACT_HFID", "google/paligemma2-3b-mix-224")
    size = os.environ.get("PU_EXTRACT_SIZE", "3b")
    N = int(os.environ.get("PU_EXTRACT_N", "45000"))
    bs = int(os.environ.get("PU_EXTRACT_BATCH", "32"))
    limit = int(os.environ.get("PU_EXTRACT_LIMIT", "0"))
    n_take = limit if limit > 0 else N
    out = Path(os.environ.get(
        "PU_EXTRACT_OUT",
        str(ROOT / "data" / "cosmosweb" / f"{alias}_{size}_blocks_layerwise_ordered.parquet")))
    log(f"model={alias} {hfid} size={size} N={n_take} bs={bs} "
        f"cuda={torch.cuda.is_available()} out={out}")

    from pu.models import get_adapter
    adapter = get_adapter(alias)(hfid, size, alias=alias)
    adapter.load()

    pf = {m: step1.make_preprocessor(adapter, m) for m in MODES}

    def proc(example):
        d = {}
        for m in MODES:
            d.update(pf[m](example))
        return d

    ds = (
        load_dataset(DATASET, split="train", streaming=True)
        .select_columns([f"{m}_images" for m in MODES])
        .map(proc)
        .remove_columns([f"{m}_images" for m in MODES])
        .select_columns(list(MODES))
    )
    if hasattr(ds, "with_format"):
        ds = ds.with_format("torch")
    dl = DataLoader(ds, batch_size=bs, num_workers=0)  # num_workers=0: preserve order

    cols = {m: {} for m in MODES}  # cols[mode][layer_key] -> list of (B,d) arrays
    n_done = 0
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            for m in MODES:
                lw = adapter.embed_all_layers_for_mode(batch, m, granularity="blocks")
                for k, emb in lw.items():
                    # Drop lm_head — its width is the ~256k vocab, which is both
                    # pathological for a probe (pu_solve.py:99) and overflows
                    # polars' 2^31 element limit at 45k rows.
                    if "lm_head" in k or emb.shape[1] > 100_000:
                        continue
                    cols[m].setdefault(k, []).append(emb.float().cpu().numpy())
            n_done += len(batch[MODES[0]])
            if bi % 20 == 0:
                log(f"  batch {bi}  rows {n_done}/{n_take}")
            if n_done >= n_take:
                break

    data = {}
    for m in MODES:
        for k, chunks in cols[m].items():
            data[f"{k}_{m}"] = np.concatenate(chunks, 0)[:n_take]
    log(f"assembling {len(data)} columns x {n_take} rows")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        df = pl.DataFrame({k: pl.Series(k, v) for k, v in data.items()})
        df.write_parquet(out, compression="zstd")
    except Exception as e:  # noqa: BLE001 — don't lose ~1h of compute on a write hiccup
        npz = out.with_suffix(".npz")
        log(f"[warn] parquet write failed ({type(e).__name__}: {e}); "
            f"dumping raw arrays -> {npz} (rebuild parquet from these)")
        np.savez(npz, **data)
        raise
    log(f"[done] wrote {df.height} rows x {df.width} cols -> {out}")
    # Show the layer columns found, for a quick eyeball.
    log("columns:", sorted(data.keys())[:3], "...", f"({df.width} total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
