"""Aggregate solve outputs and pick candidate (model, survey, block) triples
worth pulling embeddings for.

Inputs:  ~/Desktop/platonic-universe/solve_out/done/*.parquet
Outputs: ~/Desktop/platonic-universe/derived/per_block.parquet
         ~/Desktop/platonic-universe/derived/candidates.parquet
"""
import glob
import os

import polars as pl

ROOT     = "/home/me/Desktop/platonic-universe"
SOLVE    = f"{ROOT}/solve_out/done"
DERIVED  = f"{ROOT}/derived"
TOP_K    = 3   # candidates per (survey, model)

os.makedirs(DERIVED, exist_ok=True)
paths = sorted(glob.glob(f"{SOLVE}/*.parquet"))
print(f"{len(paths)} solve parquets")

df = pl.concat([pl.read_parquet(p) for p in paths])
print(f"total rows: {len(df):,}")
print(f"surveys: {df['survey'].unique().to_list()}")
print(f"models : {sorted(df['model'].unique().to_list())}")
print(f"metrics: {sorted(df['metric'].unique().to_list())}")

# wide shape table per (survey, model, side, block_idx)
shape = (df.filter(pl.col("metric").str.starts_with("shape_"))
           .with_columns(side=pl.col("block_b_name"))
           .select(["survey", "model", "side", "block_a_idx", "metric", "score"])
           .pivot(values="score",
                  index=["survey", "model", "side", "block_a_idx"],
                  on="metric"))

# alignment: cka rows
align = df.filter(pl.col("metric") == "cka")

# per-(model, survey, hsc-block) summary: mean + max calibrated cka against everything on the other side
plat = (align.group_by(["survey", "model", "block_a_idx", "block_a_name"])
              .agg(mean_cka=pl.col("calibrated_score").mean(),
                   max_cka =pl.col("calibrated_score").max(),
                   n_pairs =pl.len()))

# join the HSC-side shape stats onto each (survey, model, block_a_idx)
hsc_shape = shape.filter(pl.col("side") == "hsc").drop("side")
joined = plat.join(hsc_shape, on=["survey", "model", "block_a_idx"], how="left")

joined.write_parquet(f"{DERIVED}/per_block.parquet")
print(f"\n[per_block.parquet] {len(joined)} rows -> {DERIVED}/per_block.parquet")

# candidates: top-K per (survey, model) by mean_cka
cand = (joined.sort("mean_cka", descending=True)
              .group_by(["survey", "model"], maintain_order=True)
              .head(TOP_K))
cand.write_parquet(f"{DERIVED}/candidates.parquet")
print(f"[candidates.parquet] {len(cand)} rows (top-{TOP_K} per (survey, model)) -> {DERIVED}/candidates.parquet")

# preview
with pl.Config(tbl_cols=-1, tbl_width_chars=200, tbl_rows=12):
    print("\nfirst candidates:")
    print(cand.select(["survey", "model", "block_a_idx", "block_a_name",
                       "mean_cka", "max_cka",
                       "shape_intrinsic_dim_twoNN", "shape_anisotropy",
                       "shape_effective_rank"]))
