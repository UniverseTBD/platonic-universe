# Experiment 2: Triple Co-location

**Do foundation models converge because they find the physical manifold?**

Three "special" depths are defined by three *independent* measurements, and the
hypothesis under test is that they **coincide**:

| depth | defined by |
|---|---|
| `L_conv` | the depth where cross-model neighbourhood agreement peaks |
| `L*` | the depth where the manifold emerges (minimum intrinsic dimension; from Experiment 1) |
| `L_phys` | the depth where alignment with the physical catalogue peaks |

If all three coincide for a model, that is the strongest available evidence
that the model converges *because* it finds the physical manifold, rather than
some shared statistical nuisance.

The single analysis notebook is [`triple_colocation.ipynb`](triple_colocation.ipynb).
It implements Steps 1 to 8 of the experiment specification and produces every
figure in [`figs/`](figs).

## What runs by default

The committed notebook runs on a **synthetic universe with a planted manifold
depth** ([`synthetic.py`](synthetic.py)). This is a deliberate validation
choice: co-location is a claim about *recovering a hidden depth*, and the only
way to show the pipeline recovers the right answer is to plant a known one. Each
synthetic "model" is a per-layer embedding stack whose signal peaks at a chosen
relative depth; at that layer the embedding is a low-dimensional image of a
**shared** physical latent, so by construction `L_conv`, `L*` and `L_phys` all
coincide there. Two controls exercise Step 7:

- an **instrument-confounded** model whose peak images the *nuisance* catalogue
  (seeing, sky, S/N) instead of the physical one: physics overlap stays flat
  while the nuisance overlap peaks, and the pipeline must flag it;
- an **untrained** model that is isotropic noise at every layer: all calibrated
  scores should sit near zero.

No astrophysical result is claimed from synthetic data. The demonstrated claim
is narrower and honest: the pipeline recovers a planted depth and its
uncertainty, and the controls fire as they should. On this run all four physics
models co-locate (at heterogeneous depths, 0.55 to 0.69), the instrument control
is flagged, and the untrained control shows no alignment anywhere.

## Running it

```bash
pip install -e ".[colocation]"
jupyter nbconvert --to notebook --execute --inplace \
    experiments/colocation/triple_colocation.ipynb
```

The core primitives live in [`pu.metrics.colocation`](../../src/pu/metrics/colocation.py)
and need only numpy + scikit-learn (both core deps). The `[colocation]` extra
adds the notebook stack (matplotlib, jupyter).

## Running on real astronomy embeddings

Set `USE_REAL_DATA = True` in the config cell and point it at:

- `REAL_PARQUET_DIR`: a directory of `*_layerwise.parquet` files produced by
  `pu.experiments_layerwise.extract_all_layers` (columns `{layer}_{mode}`);
- `REAL_CATALOG`: a parquet catalogue, **row-aligned** with the embeddings,
  with the physical columns (`REAL_PHYS_COLS`) and nuisance columns
  (`REAL_NUIS_COLS`).

Nothing else changes: the same functions consume real and synthetic layer
stacks identically. The scientific question then is whether the real `L_conv`,
`L*` and `L_phys` coincide, whether their peak locations are sample-size stable,
and whether physics beats nuisance.

The COSMOS-Web catalogue used elsewhere in this repo
([`experiments/cosmosweb`](../cosmosweb)) exposes the physical columns
(`redshift`, `mag_g`, `mag_r`, `lp_mass`, `lp_ssfr`); a nuisance catalogue
(seeing, sky background, S/N) is needed for the Step 7 control.

## The DADApy note (why the default backend is pure numpy)

The overlap atom is exactly
`dadapy.Data(A).return_data_overlap(coordinates=B, k=k)`: for each row, the
fraction of its `k` nearest neighbours in space A that are also among its `k`
nearest neighbours in space B, averaged over rows. DADApy is supported as an
**optional** backend (`overlap(..., backend="dadapy")`), but it is not the
default and not a dependency: its compiled Cython extension does not build
against numpy >= 2 (it references the removed `_PyArray_Descr.subarray` C
field), and this repo pins `numpy>=2.3`. The pure-numpy path is numerically the
same quantity, adds the "build the base kNN graph once, reuse it across the row"
optimisation the spec asks for, and runs on the repo's own stack. To use the
DADApy backend, install `dadapy` in a separate `numpy<2` environment;
`tests/test_colocation.py` has a gated test that checks the two backends agree
there.

## Method summary (per step)

1. **`L_conv`.** For every model pair, the full calibrated cross-layer overlap
   matrix `M_cal[i, j]`; each A-layer graph is built once and reused across the
   row of B-layers and across the 10 permutation nulls. The peak cell gives a
   peak layer per model; `L_conv` is the median over pairs.
2. **`L_phys`.** Per-layer calibrated overlap with the standardised physical
   catalogue `P_std` (null permutes `P_std`'s rows); `L_phys` is the argmax.
   Supporting curves: held-out probe `R^2` (`RidgeCV` in
   `cross_val_score(KFold(5, shuffle=True, random_state=0))`) and held-out CCA
   (`CCA(n_components=p)`, 70/30 split).
3. **Assembly.** The three relative depths per model, bootstrapped on 10 random
   half-samples (uncertainty = interquartile range).
4. **Coincidence test.** A model co-locates iff all three pairwise gaps are at
   most `0.10` and the three bootstrap `[q25, q75]` intervals mutually overlap.
   Depths are never averaged across models before the test.
5. **Alignment vs sample size.** Values fall with `n`; peak *locations* must
   stay put. Reports the standard deviation of the recovered peak locations.
6. **Interpretation table.** Every model is classified by the relationship among
   its three depths and its controls.
7. **Controls.** Nuisance catalogue `Q_std` in place of `P_std`; untrained model;
   `k/N` annotated on raw alignment figures.
8. **Deliverable.** This notebook, with the co-location figure, the calibrated
   cross-layer heatmap, `s_phys` vs depth with null band, the alignment-vs-n
   figures, the interpretation table, the control results, and a summary.

## Files

| file | purpose |
|---|---|
| `triple_colocation.ipynb` | the single analysis notebook (Steps 1 to 8) |
| `synthetic.py` | planted-ground-truth universe generator for validation |
| `figs/` | figures written by the notebook |
| `../../src/pu/metrics/colocation.py` | the reusable primitives |
| `../../tests/test_colocation.py` | unit + integration tests |
