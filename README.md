# The Platonic Universe: Testing The Platonic Representation Hypothesis With Astronomical Data 🔮 💫

This repository contains the code for testing the **Platonic Representation Hypothesis (PRH)** on astronomical data, as described in our paper "The Platonic Universe: Do Foundation Models See the Same Sky?"

## Background & Motivation

The Platonic Representation Hypothesis suggests that neural networks trained with different objectives on different data modalities converge toward a shared statistical model of reality in their representation spaces. As models become larger and are trained on more diverse tasks, they should converge toward a "Platonic ideal" representation of underlying reality.

### Why Astronomy?

Astronomical observations provide an ideal testbed for the PRH because:
- **Shared Physical Origin**: Different astronomical observations (images, spectra, photometry) all emerge from the same underlying physics
- **Multiple Modalities**: We can compare representations across fundamentally different data types (like optical images, infrared images, and spectroscopy)
- **Scale**: Modern astronomical surveys provide the data volume necessary to test convergence across multiple model architectures

Our results (below) show that **larger models exhibit more similar representations**, even when trained across different data modalities. This suggests that astronomical foundation models may be able to leverage pre-trained general-purpose architectures.

<img src="https://github.com/UniverseTBD/platonic-universe/blob/main/figs/mknn.png" width=100%/>


## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/UniverseTBD/platonic-universe.git](https://github.com/UniverseTBD/platonic-universe.git)
    cd platonic-universe    
    ```

2.  **Install dependencies using uv:**
    ```bash
    pip install uv
    uv sync
    ```
3. **Install the package**
   ```bash
   uv pip install .
   ```
> To install without uv, you can create a virtual environment and install the requirements manually:
> ```bash
> python -m venv pu_env
> source pu_env/bin/activate
> pip install .
> ## or to install with SAM2 support:
> pip install ".[sam2]"

### Quick Start: Running Experiments

There are two methods to run experiments:

1. **Using platonic_universe CLI:**
    ```bash
    platonic_universe run --model vit --mode jwst 
    ```

2. **Using python pakage directly:**
    ```python
    import pu 

    pu.run_experiment("vit", "sdss", batch_size=64, num_workers=1, knn_k=10)
    ``` 



### Supported Models & Datasets

**Models Tested:**
- **Vision Transformers (ViT)**: Base, Large, Huge
- **DINOv2**: Small, Base, Large, Giant
- **ConvNeXtv2**: Nano, Tiny, Base, Large
- **IJEPA**: Huge, Giant
- **AstroPT**: Astronomy-specific transformer (Small, Base, Large)
- **Specformer**: Spectroscopy-specific model (Enable by uncommenting relevant lines in 'experiments.py')

**Astronomical Datasets:**
- **HSC (Hyper Suprime-Cam)**: Ground-based optical imaging (reference baseline)
- **JWST**: Space-based infrared imaging
- **Legacy Survey**: Ground-based optical imaging
- **DESI**: Spectroscopy

### Understanding the Results

The code measures **representational alignment** using the Mutual k-Nearest Neighbour (MKNN) metric:

```python
from pu.metrics import mknn

# Calculate MKNN score between two embedding sets
score = mknn(embeddings_1, embeddings_2, k=10)
print(f"MKNN alignment score: {score:.4f}")
```

**Higher MKNN scores** indicate more similar representations between models or modalities.

# Notes on Specformer:

Note: the validation used ConvNeXtv2-Base (facebook/convnextv2-base-22k-224) for the image side. If more validation across other sizes/families is wanted we can run it, but it shouldn't be required — this confirms the SpecFormer adapter produces correct embeddings and the cross-modal MKNN is consistent with the paper.

Re: whether platonic_universe run --model specformer --mode desi gives the same result — not directly. That command generates and saves the SpecFormer embeddings to parquet, but since specformer is spectral-only it skips HSC and metric computation. To reproduce this MKNN via the CLI you'd:

    platonic_universe run --model specformer --mode desi → fresh specformer embeddings
    platonic_universe run --model convnext --mode desi → ConvNeXtv2 HSC embeddings (+ pre-computed DESI embeddings from Smith42/specformer_desi)
    platonic_universe compare <specformer_parquet> --ref <convnext_parquet> --mode desi

Step 2 already produces an MKNN using the pre-computed specformer embeddings from HuggingFace, so that's the quickest way to sanity-check. The validation script above computes embeddings fresh end-to-end as an independent check.

## Contributing

This project is open source under the AGPLv3.

We welcome contributions! Please feel free to open a pull request to:
- Add support for new model architectures
- Include additional astronomical datasets
- Implement alternative similarity metrics
- Improve preprocessing pipelines

We also hang out on the UTBD Discord, [so feel free to reach out there!](https://discord.gg/VQvUSWxnu9)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{utbd2025,
	author = {{UniverseTBD} and Duraphe, K. and Smith, M. J. and Sourav, S. and Wu, J. F.},
	title = {{The Platonic Universe: Do Foundation Models See the Same Sky?}},
	journal = {ArXiv e-prints},
	year = {2025},
	eprint = {2509.19453},
	doi = {10.48550/arXiv.2509.19453}
}
```
