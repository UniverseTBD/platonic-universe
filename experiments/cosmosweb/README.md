# COSMOS-Web reproduction scripts

Self-contained scripts to reproduce the COSMOS-Web physics-validation experiments
on top of the `pu` package. These are not part of the installed package; they
import from `pu` and run as standalone scripts.

## Install

From the repo root:

```bash
pip install -e ".[cosmosweb]"
```

This installs `pu` plus the extra dependencies the scripts here need
(`scikit-learn`, `scipy`, `matplotlib`, `tqdm`).

## What's here

| File | Purpose |
|---|---|
| `run_intramodal.py` | Per-survey embedding + MKNN + linear-probe scaling |
| `run_crossmodal.py` | HSC ↔ JWST cross-modal version of the same |
| `probe_weight_analysis.py` | Inspect linear-probe weights on the embeddings |
| `plot_mknn_vs_r2.py` | Plotting utility for MKNN-vs-$R^2$ scaling figures |
| `test_linear_probe.py` | Minimal end-to-end test on convnext + cosmosweb |

## Run

```bash
python experiments/cosmosweb/run_intramodal.py
python experiments/cosmosweb/run_crossmodal.py
```

Both scripts write PDFs of figures and `.npy` cached embeddings to `embeddings/`.

## Dataset

The scripts pull from
[`HCVYM5w6Gn/cosmosweb-hsc-jwst-high-snr-pil2`](https://huggingface.co/datasets/HCVYM5w6Gn/cosmosweb-hsc-jwst-high-snr-pil2)
(set `HF_TOKEN` if private). The dataset adapter
(`pu.pu_datasets.cosmosweb`) and the physics-probing utilities
(`pu.metrics.physics`, `pu.metrics.neighbors.mknn_neighbor_input`) are part of
the main `pu` package and are re-used by these scripts.

## Notes

- These scripts were originally contributed in PR #58 by . They have
  been moved here, unchanged, so the main package stays small. The shared
  utilities they rely on were merged into core.
- Run on a GPU. CPU runs will be very slow.
