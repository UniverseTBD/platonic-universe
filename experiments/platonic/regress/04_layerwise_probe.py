#!/usr/bin/env python3
"""Step 4 — layerwise physics probe for PaliGemma 2 (3B) embeddings.

The paper (*The Platonic Universe*) probes only ONE layer per model. This
script instead probes EVERY layer of the PaliGemma 2 stack — the SigLIP
vision tower (27 layers) through the Gemma-2 decoder (26 layers) plus the
published masked-mean final hidden state — against the three canonical galaxy
properties (redshift, stellar mass, sSFR), for both HSC and JWST.

Input: a cosmosweb layerwise parquet with one fixed_size_list column per layer
per modality (columns suffixed ``_hsc`` / ``_jwst``); no physics labels inside.
Labels are joined by row index from the cosmosweb catalog (streamed in the same
order the embeddings were written).

Output: ``data/paligemma_3b_layerwise_probe.parquet`` — one tidy row per
(modality, stack, layer_index, property) with r2_mean / r2_std / is_published.

The probe recipe (StandardScaler per train fold, OLS, 10 random 80/20 splits,
2000 held out) is reused verbatim from ``01_extract_and_probe.py`` so it is the
exact published recipe — see ``_probe_published`` there.

Environment
-----------
    PU_LAYERWISE_INPUT    glob for the layerwise parquet
                          (default data/cosmosweb/paligemma_3b*blocks_layerwise.parquet)
    PU_LAYERWISE_OUTPUT   output parquet (default data/paligemma_3b_layerwise_probe.parquet)
    PU_LAYERWISE_N_USE    catalog rows / labels to use (default 45000, paper setting)
    PU_LAYERWISE_ONLY     substring filter on column names (smoke test, e.g. hidden_states_last)
    PU_LAYERWISE_LIMIT    cap number of embedding columns processed (0 = all)
"""
from __future__ import annotations

import glob
import importlib.util
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA_DIR = ROOT / "data"

PROPERTIES = ("redshift", "mass", "sSFR")


def _load_step1():
    """Import 01_extract_and_probe.py by path (module name starts with a digit).

    Its top level reads four HF-pipeline env vars and builds an HfApi. We only
    want the catalog + probe helpers, so default those vars to throwaway values
    (no network happens at import) if the caller hasn't set them.
    """
    tmp = Path(tempfile.gettempdir())
    for key, val in {
        "HF_TOKEN": "unused",
        "PU_REGRESS_RESULTS_REPO": "unused/unused",
        "PU_REGRESS_OUT": str(tmp / "pu_layerwise_out"),
        "PU_REGRESS_LOCKS": str(tmp / "pu_layerwise_locks"),
    }.items():
        os.environ.setdefault(key, val)
    spec = importlib.util.spec_from_file_location(
        "pu_regress_step1", HERE / "01_extract_and_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_col(col: str) -> dict | None:
    """Parse an embedding column name into layer metadata, or None to skip.

    Keys on substrings so minor naming drift is tolerated:
      ...vision_tower...layers.{i}   -> vision  (order i)
      ...language_model.layers.{i}   -> llm     (order 1000+i, after vision)
      hidden_states_last / embeddings-> final   (order 2000, the published point)
      lm_head / anything else        -> skip
    """
    for mod in ("hsc", "jwst"):
        if col.endswith("_" + mod):
            base = col[: -(len(mod) + 1)]
            break
    else:
        return None

    m = re.search(r"vision_tower.*\.layers\.(\d+)$", base)
    if m:
        i = int(m.group(1))
        return dict(modality=mod, stack="vision", layer_index=i,
                    order=i, is_published=False, col=col)
    m = re.search(r"language_model\.layers\.(\d+)$", base)
    if m:
        i = int(m.group(1))
        return dict(modality=mod, stack="llm", layer_index=i,
                    order=1000 + i, is_published=False, col=col)
    if "hidden_states_last" in base or base == "embeddings":
        return dict(modality=mod, stack="final", layer_index=-1,
                    order=2000, is_published=True, col=col)
    return None  # lm_head (pathological d) or unknown -> drop


def read_matrix(path: str, col: str) -> np.ndarray:
    """Read one fixed_size_list column as a (n, d) float32 array.

    One column at a time: ~0.5 GB for a 2304-d layer over 56k rows, never the
    whole ~30 GB file.
    """
    fsl = pq.read_table(path, columns=[col]).combine_chunks().column(0).chunk(0)
    d = fsl.type.list_size
    flat = fsl.values.to_numpy(zero_copy_only=False)
    return flat.reshape(len(fsl), d).astype(np.float32, copy=False)


def main() -> int:
    step1 = _load_step1()
    log = step1.log

    in_glob = os.environ.get(
        "PU_LAYERWISE_INPUT",
        str(DATA_DIR / "cosmosweb" / "paligemma_3b*blocks_layerwise.parquet"))
    matches = sorted(glob.glob(in_glob))
    if not matches:
        log(f"[fatal] no input parquet matches {in_glob!r}")
        return 1
    path = matches[0]
    out_path = Path(os.environ.get(
        "PU_LAYERWISE_OUTPUT", str(DATA_DIR / "paligemma_3b_layerwise_probe.parquet")))
    n_use = int(os.environ.get("PU_LAYERWISE_N_USE", "45000"))
    only = os.environ.get("PU_LAYERWISE_ONLY", "")
    limit = int(os.environ.get("PU_LAYERWISE_LIMIT", "0"))

    log(f"input={path}")
    log(f"output={out_path}  N_USE={n_use}  only={only!r}  limit={limit}")

    # Parse layer columns from the schema alone (no data read yet).
    schema = pq.ParquetFile(path).schema_arrow
    cols = [parse_col(c) for c in schema.names]
    cols = [c for c in cols if c is not None]
    if only:
        cols = [c for c in cols if only in c["col"]]
    cols.sort(key=lambda c: (c["modality"], c["order"]))
    if limit > 0:
        cols = cols[:limit]
    log(f"{len(cols)} embedding columns to probe "
        f"(x{len(PROPERTIES)} properties = {len(cols) * len(PROPERTIES)} probes)")

    catalog = step1.collect_catalog(n_use)

    rows = []
    for ci, c in enumerate(cols, 1):
        Z = read_matrix(path, c["col"])
        log(f"[{ci}/{len(cols)}] {c['col']}  Z={Z.shape}")
        for prop in PROPERTIES:
            y_raw = catalog[prop]
            n = min(len(Z), len(y_raw))
            mask, y = step1._prep_property(prop, y_raw[:n])
            try:
                r2_mean, r2_std = step1._probe_published(Z[:n], y, mask)
            except Exception as e:  # noqa: BLE001
                log(f"    {prop}: probe failed: {type(e).__name__}: {e}")
                r2_mean, r2_std = float("nan"), float("nan")
            log(f"    R2({prop}) = {r2_mean:.4f} ± {r2_std:.4f}  "
                f"(n_valid={int(mask.sum())})")
            rows.append({
                "modality": c["modality"],
                "stack": c["stack"],
                "layer_index": c["layer_index"],
                "order": c["order"],
                "property": prop,
                "r2_mean": r2_mean,
                "r2_std": r2_std,
                "is_published": c["is_published"],
                "d": int(Z.shape[1]),
                "n": int(n),
                "col": c["col"],
            })
        del Z

    import polars as pl
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(out_path, compression="zstd")
    log(f"[done] wrote {len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
