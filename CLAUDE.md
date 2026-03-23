# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Platonic Universe tests the **Platonic Representation Hypothesis** on astronomical data — whether foundation models trained on different objectives/modalities converge toward shared representations. It compares embeddings from various vision models (ViT, DINOv2, ConvNeXt, IJEPA, AstroPT, SAM2, etc.) across astronomical datasets (HSC, JWST, Legacy Survey, DESI) using representational similarity metrics (primarily Mutual k-Nearest Neighbour).

## Commands

```bash
# Install
uv sync && uv pip install .

# Run tests
uv run pytest                    # all tests
uv run pytest tests/test_metrics_kernel.py  # single file
uv run pytest -k "test_cka"     # single test by name

# Lint
uv run ruff check src/          # ruff config in pyproject.toml: E, F, I rules; E501 ignored

# CLI (after install)
pu run --model vit --mode jwst
pu compare data/embeddings.parquet --metrics cka mknn
pu calibrate data/embeddings.parquet --metrics cka
pu benchmark --model vit --mode jwst
```

## Architecture

**Source layout**: `src/pu/` with package installed via `pyproject.toml` + `setup.py` (setup.py builds a C++/pybind11 extension `pu_cka` for fast CKA computation).

**Registry pattern**: Both models and datasets use self-registering adapter registries. Adapters register themselves on import via side-effect imports in `__init__.py`. To add a new model or dataset, create an adapter module and import it in the respective `__init__.py`.

- **Models** (`src/pu/models/`): Adapters wrap HuggingFace models (`hf.py` covers ViT/DINO/ConvNeXt/IJEPA/VJEPA/Hiera/MAE), AstroPT (`astropt.py`), and SAM2 (`sam2.py`). All inherit from `BaseAdapter` in `base.py`. Registry in `registry.py`.
- **Datasets** (`src/pu/pu_datasets/`): Adapters for HF crossmatched datasets (`hf_crossmatched.py` covers JWST/Legacy Survey), SDSS (`sdss.py`), and DESI (`desi.py`). All inherit from `BaseDatasetAdapter` in `base.py`. Registry in `registry.py`.
- **Metrics** (`src/pu/metrics/`): Organized by family — kernel (`cka`, `mmd`), geometric (`procrustes`, `frechet`), CCA (`svcca`, `pwcca`), spectral, information-theoretic, neighbor-based (`mknn`, `jaccard`), regression. The `io.py` module provides `compare()` for batch metric computation and parquet I/O. `METRICS_REGISTRY` maps metric names to functions.
- **Experiments** (`experiments.py`): Orchestrates the pipeline — loads model+dataset adapters, generates embeddings, computes metrics, saves results to `data/`.
- **CLI** (`__main__.py`): Four subcommands: `run`, `compare`, `calibrate`, `benchmark`.

**Key flow**: HSC is always the reference baseline. Experiments compare HSC embeddings against a second modality (JWST, Legacy Survey, SDSS, DESI) for each model size.

**C++ extension**: `src/pu/cpp/` contains CKA computation via pybind11 (`pu_cka` module), compiled with OpenMP. Used through `compute_cka_mmap()`.

## Important Details

- Python >=3.11 required
- `transformers` is installed from git HEAD (see `[tool.uv.sources]`)
- Optional `sam2` dependency: `pip install ".[sam2]"`
- Paired modes (SDSS, DESI) force `num_workers=0` to preserve draw order
- Lazy imports throughout to avoid loading torch/transformers when only using metrics
