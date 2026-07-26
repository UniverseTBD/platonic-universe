# Residualizing astrophysics out of foundation-model embeddings — results

Calibrated **CKA** (global) and **MKNN** (local) similarity between model pairs, **before vs after** regressing the astrophysics (`redshift`, `mass`, `sSFR`) out of both models' embeddings, on the COSMOS-Web HSC sample.

## Setup

- **Data:** pre-computed cosmosweb embeddings from `UniverseTBD/pu-embeddings` (`cosmosweb/.../embeddings/*_45000.parquet`), row-aligned by stream order to the LePHARE labels in `Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2`. No extraction, CPU-only.
- **Confounds removed:** redshift, mass, sSFR — degree-3 polynomial basis (bias + 3×3 = 10 columns).
- **Residualization:** orthogonal projection `Z − B·lstsq(B,Z)` fit on all 44913 valid rows (in-sample), cross-checked against 5-fold cross-fitted (out-of-fold) residualization.
- **Calibration:** permutation null (Algorithm 1 of `calibrated_similarity`), K=200, on a random subsample of n=3000, averaged over 3 draws. `retained = calibrated_after / calibrated_before`.

## In-sample: calibrated CKA / MKNN, before → after

| pair | CKA before | CKA after | **CKA retained** | MKNN before | MKNN after | **MKNN retained** |
|---|---:|---:|---:|---:|---:|---:|
| convnext/base vs convnext/large | 0.725 | 0.666 | 91.9% | 0.199 | 0.193 | 96.9% |
| convnext/nano vs convnext/base | 0.594 | 0.483 | 81.3% | 0.124 | 0.114 | 92.1% |
| convnext/nano vs convnext/large | 0.577 | 0.475 | 82.4% | 0.133 | 0.121 | 91.2% |
| convnext/nano vs convnext/tiny | 0.689 | 0.614 | 89.2% | 0.162 | 0.148 | 91.2% |
| convnext/tiny vs convnext/base | 0.723 | 0.622 | 86.0% | 0.150 | 0.140 | 93.4% |
| convnext/tiny vs convnext/large | 0.637 | 0.544 | 85.4% | 0.148 | 0.141 | 95.2% |
| vit/base vs convnext/base | 0.523 | 0.399 | 76.2% | 0.089 | 0.083 | 93.8% |
| vit/base vs convnext/large | 0.472 | 0.376 | 79.5% | 0.096 | 0.093 | 97.5% |
| vit/base vs convnext/nano | 0.481 | 0.379 | 78.9% | 0.080 | 0.074 | 91.7% |
| vit/base vs convnext/tiny | 0.508 | 0.382 | 75.2% | 0.082 | 0.074 | 91.3% |
| vit/base vs vit/huge | 0.586 | 0.461 | 78.6% | 0.128 | 0.119 | 93.3% |
| vit/base vs vit/large | 0.605 | 0.483 | 79.9% | 0.134 | 0.127 | 94.9% |
| vit/huge vs convnext/base | 0.593 | 0.451 | 76.2% | 0.113 | 0.105 | 92.8% |
| vit/huge vs convnext/large | 0.477 | 0.366 | 76.7% | 0.118 | 0.108 | 91.3% |
| vit/huge vs convnext/nano | 0.549 | 0.483 | 88.0% | 0.121 | 0.109 | 90.1% |
| vit/huge vs convnext/tiny | 0.613 | 0.483 | 78.8% | 0.119 | 0.108 | 90.1% |
| vit/large vs convnext/base | 0.542 | 0.396 | 73.1% | 0.089 | 0.087 | 97.7% |
| vit/large vs convnext/large | 0.507 | 0.391 | 77.1% | 0.097 | 0.092 | 94.6% |
| vit/large vs convnext/nano | 0.517 | 0.416 | 80.6% | 0.081 | 0.075 | 92.3% |
| vit/large vs convnext/tiny | 0.543 | 0.402 | 74.2% | 0.082 | 0.076 | 92.7% |
| vit/large vs vit/huge | 0.677 | 0.541 | 80.0% | 0.145 | 0.140 | 96.9% |
| **mean** | | | **80.4%** | | | **93.4%** |

## Leakage check: in-sample vs cross-fitted (CV=5)

If the in-sample deconfounding were leaking, the cross-fitted (out-of-fold) retention would differ. It doesn't — the confound basis is 10-dim vs n≈45k, so in-sample ≈ out-of-fold.

| pair | CKA retained (in-sample) | CKA retained (CV=5) | Δ pp | MKNN retained (in-sample) | MKNN retained (CV=5) | Δ pp |
|---|---:|---:|---:|---:|---:|---:|
| convnext/base vs convnext/large | 91.9% | 91.9% | +0.0 | 96.9% | 96.7% | -0.2 |
| convnext/nano vs convnext/base | 81.3% | 81.4% | +0.0 | 92.1% | 91.9% | -0.2 |
| convnext/nano vs convnext/large | 82.4% | 82.4% | +0.0 | 91.2% | 91.3% | +0.0 |
| convnext/nano vs convnext/tiny | 89.2% | 89.2% | +0.0 | 91.2% | 91.1% | -0.1 |
| convnext/tiny vs convnext/base | 86.0% | 86.0% | +0.0 | 93.4% | 93.3% | -0.1 |
| convnext/tiny vs convnext/large | 85.4% | 85.4% | +0.0 | 95.2% | 95.1% | -0.2 |
| vit/base vs convnext/base | 76.2% | 76.3% | +0.0 | 93.8% | 93.8% | -0.0 |
| vit/base vs convnext/large | 79.5% | 79.6% | +0.0 | 97.5% | 97.2% | -0.3 |
| vit/base vs convnext/nano | 78.9% | 78.9% | +0.1 | 91.7% | 91.4% | -0.3 |
| vit/base vs convnext/tiny | 75.2% | 75.2% | +0.0 | 91.3% | 91.3% | +0.0 |
| vit/base vs vit/huge | 78.6% | 78.7% | +0.0 | 93.3% | 93.1% | -0.2 |
| vit/base vs vit/large | 79.9% | 79.9% | +0.0 | 94.9% | 94.8% | -0.1 |
| vit/huge vs convnext/base | 76.2% | 76.2% | +0.0 | 92.8% | 93.0% | +0.2 |
| vit/huge vs convnext/large | 76.7% | 76.8% | +0.0 | 91.3% | 91.4% | +0.1 |
| vit/huge vs convnext/nano | 88.0% | 88.1% | +0.0 | 90.1% | 90.1% | -0.0 |
| vit/huge vs convnext/tiny | 78.8% | 78.8% | +0.0 | 90.1% | 90.0% | -0.1 |
| vit/large vs convnext/base | 73.1% | 73.2% | +0.0 | 97.7% | 97.7% | +0.0 |
| vit/large vs convnext/large | 77.1% | 77.1% | +0.0 | 94.6% | 94.5% | -0.1 |
| vit/large vs convnext/nano | 80.6% | 80.6% | +0.0 | 92.3% | 92.6% | +0.3 |
| vit/large vs convnext/tiny | 74.2% | 74.2% | +0.0 | 92.7% | 92.8% | +0.1 |
| vit/large vs vit/huge | 80.0% | 80.0% | +0.0 | 96.9% | 97.0% | +0.1 |

**Max |Δ retained| across all pairs: CKA 0.1 pp, MKNN 0.3 pp** → in-sample residualization is not leaking.

## Removal check (probe R²)

Linear-probe R² for each property before vs after residualization, for the reference pair. Healthy *before* R² also confirms the embedding↔label row-alignment; *after* ≈ 0 confirms removal.

| model | property | R² before | R² after |
|---|---|---:|---:|
| convnext/base | redshift | +0.398 | -0.074 |
| convnext/large | redshift | +0.409 | -0.112 |
| convnext/base | mass | +0.479 | -0.075 |
| convnext/large | mass | +0.496 | -0.125 |
| convnext/base | sSFR | +0.450 | -0.075 |
| convnext/large | sSFR | +0.464 | -0.124 |

## Interpretation

- Jointly removing `redshift`, `mass`, `sSFR` drops **global CKA by ~20%** (mean 80% retained, range 73–92%) and **local MKNN by ~7%** (mean 93% retained, range 90–98%).
- Both decrease, but **local neighbourhood structure (MKNN) is markedly more robust than global geometry (CKA)** — ~93% of the local alignment survives vs ~80% of the global. So a substantial part of the *global* cross-model similarity is carried by these three astrophysical properties, while the *local* agreement is mostly carried by structure beyond them. Consistent with the paper's local-vs-global (Aristotelian) reading.
- **Leakage-free:** in-sample and cross-fitted (out-of-fold) residualization agree to <0.5 pp on every pair, as expected for a 10-dim confound basis on ~45k samples.

### Notes

- The three confounds are removed **jointly** — one least-squares projection onto the 10-dim polynomial confound subspace `[1, z, z², z³, m, m², m³, s, s², s³]`, not three separate regressions — so correlations among redshift/mass/sSFR are partialled out correctly.
- Reproduced across the in-sample and cross-fitted batches (agree to ~4 decimals); the reference pair matches the standalone check. (An earlier version removed only `redshift` due to a since-fixed `sbatch --export` comma-splitting bug; this file is the corrected redshift+mass+sSFR result.)
