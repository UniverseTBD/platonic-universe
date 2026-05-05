# Platonic Universe — code for "Do Foundation Models See the Same Sky?"

This repository contains the code and reproduction recipes for the paper
*The Platonic Universe: Do Foundation Models See the Same Sky?*
(under double-blind review). It tests the Platonic Representation
Hypothesis on astronomical foundation models by measuring local (MKNN)
and global (CKA) representational alignment across architectures and
across imaging / spectroscopic modalities.

## Step 1 — Download the code

In the top right of the anonymous.4open.science page you will see a
**ZIP** button:

![ZIP button location](docs/zip_button.png)

Click it. It downloads `<repo>.zip`. Unzip it and `cd` into the unzipped
folder.

## Step 2 — Install

Install `uv` (fast Python package manager) — see the official
instructions at <https://docs.astral.sh/uv/getting-started/installation/>.
Then, from the repo root:

```
uv sync
uv pip install .
```

Python ≥ 3.11 required. No HF login or token is needed — every dataset
mirror is public.

## Step 3 — Reproduce every paper figure

```
bash reproduce_figures.sh
```

That single command pulls the necessary embeddings from Hugging Face,
computes the Procrustes distances + cosine-similarity matrices, and
renders every main-paper and appendix figure to `figs/`.

## Where the data lives

All datasets and model checkpoints used by the paper are mirrored on
Hugging Face under the anonymous account `HCVYM5w6Gn`. They are pulled
automatically by `reproduce_figures.sh`; you do not need to download
anything by hand.

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

## Layout

```
src/pu/                package: model adapters, dataset adapters, metrics, CLI
experiments/cosmosweb/  cosmosweb embedding extraction + probe analysis
experiments/platonic/   crossmatched-survey solve / UMAP pipeline
scripts/                figure plotters
tests/                  unit tests
reproduce_figures.sh    one-shot recipe
```

## License

AGPLv3.
