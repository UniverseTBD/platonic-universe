# Residualize tier

Does the cross-model alignment survive removing the astrophysics? For a pair of
models we residualize the confounds (`redshift`, `mass`, `sSFR`) out of each
embedding and measure null-calibrated **CKA** and **MKNN** before vs after.

Embeddings are the pre-computed COSMOS-Web parquets on
`UniverseTBD/pu-embeddings` (`cosmosweb/.../embeddings/*_45000.parquet`),
row-aligned by stream order to the LePHARE labels in
`Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2`. CPU-only, no extraction.

## Files

- `pu.metrics.deconfound.residualize` — least-squares projection onto the
  orthogonal complement of a polynomial confound basis (`residualize_crossfit`
  is the out-of-fold variant).
- `residualize_align.py` — calibrated CKA/MKNN before/after, with a probe-R²
  removal check.
- `residualize_control.py` — random-subspace control (physics vs random
  removal at matched rank / matched variance).
- `slurm/residualize_hf.sh`, `slurm/control_hf.sh` — one-pair SBATCH wrappers.
- `queue_pairs.sh`, `queue_control.sh` — fan out over all pairs of a model set.

## Run

```bash
cd experiments/platonic/residualize
sbatch slurm/residualize_hf.sh                 # vit/base vs convnext/base
bash queue_pairs.sh                            # all ViT×ConvNeXt pairs
CV=5 bash queue_pairs.sh                        # cross-fitted (leakage check)
bash queue_control.sh                          # random-direction control
```

Results: `RESULTS.md` (before/after + leakage check), `RESULTS_control.md`
(random-direction control).
