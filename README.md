# Platonic Universe — code for "Do Foundation Models See the Same Sky?"

This repository contains the code and reproduction recipes for the paper
*The Platonic Universe: Do Foundation Models See the Same Sky?* (under
double-blind review). It tests the Platonic Representation Hypothesis on
astronomical foundation models by measuring local (MKNN) and global (CKA)
representational alignment across architectures and across imaging /
spectroscopic modalities.

## Install

```
pip install uv
uv sync
uv pip install .
```

Python ≥ 3.11 required.

## Where the data lives

All datasets and model checkpoints used by the paper are mirrored on
Hugging Face under the anonymous account `HCVYM5w6Gn`:

| Repo | Contents |
|---|---|
| `HCVYM5w6Gn/jwst_hsc_crossmatched` | HSC × JWST imagery |
| `HCVYM5w6Gn/legacysurvey_hsc_crossmatched` | HSC × Legacy Survey imagery |
| `HCVYM5w6Gn/desi_hsc_crossmatched` | HSC × DESI spectra |
| `HCVYM5w6Gn/sdss_hsc_crossmatched` | HSC × SDSS spectra |
| `HCVYM5w6Gn/cosmosweb-hsc-jwst-high-snr-pil2` | COSMOS-Web HSC × JWST with physics labels |
| `HCVYM5w6Gn/specformer_desi` | Pre-computed Specformer embeddings of DESI |
| `HCVYM5w6Gn/SDSS_Interpolated` | SDSS spectra interpolated to DESI grid |
| `HCVYM5w6Gn/pu-embeddings` | Per-(model, modality) frozen embeddings |
| `HCVYM5w6Gn/astroPT_v2.0` (model) | AstroPT v2 checkpoints |

`HF_TOKEN` is **not** required — every mirror is public.

## Reproducing the paper's figures

The paper's main and appendix figures are produced by the scripts under
`scripts/` from a single committed input file
(`r2_vs_params_45000galaxies_upsampled.json`) plus a small Procrustes
distances pickle. The pickle is regenerable end-to-end from the HF
mirrors above:

```
# Stream-convert the cosmosweb embedding parquets to .npy
python experiments/cosmosweb/stream_embeddings_to_npy.py

# Fit linear probes + Procrustes distances + cosine-similarity figures
python experiments/cosmosweb/probe_weight_analysis.py

# Render every paper figure that depends on the pickle
python scripts/plot_crossarchitectural_procrustes.py
python scripts/plot_crossmodal_procrustes.py
python scripts/plot_crossmodal_procrustes_per_property.py
```

The MKNN/CKA-based figures (`intramodal*.pdf`, `crossmodal*.pdf`,
`crossarchitectural.pdf`) read directly from
`r2_vs_params_45000galaxies_upsampled.json` plus the per-(survey, model)
parquets pulled from `HCVYM5w6Gn/pu-solve-results` — no extraction
needed:

```
python scripts/plot_r2_vs_params.py
python scripts/plot_intramodal.py
python scripts/plot_crossmodal.py
```

End-to-end reproduction details, including the bandwidth-bounded
streaming protocol used to convert ~30 GB of cosmosweb parquets to
`.npy`, are documented in
[`experiments/cosmosweb/PROCRUSTES_PIPELINE.md`](experiments/cosmosweb/PROCRUSTES_PIPELINE.md).

## Layout

```
src/pu/                        package: model adapters, dataset adapters,
                                metric implementations, CLI
experiments/cosmosweb/          cosmosweb embedding extraction + probe
                                analysis pipeline
experiments/platonic/           crossmatched-survey solve / UMAP pipeline
scripts/                        plotting scripts that produce the paper's
                                main and appendix figures
tests/                          unit tests
```

## License

AGPLv3.
