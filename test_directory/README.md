# Platonic Universe

A modular Python package for multimodal astronomical data analysis, providing streamlined workflows for model loading, data fetching, embedding generation, and k-nearest neighbor comparisons.

## Features

- **Modular Architecture**: Base classes for easy extension with different vision models (DinoV2, ViT, I-JEPA)
- **Custom Cache Management**: Configurable HuggingFace cache directories with automatic space detection
- **Multimodal Support**: Handle both astronomical images and spectra with unified workflows
- **Cross-Modal Analysis**: Tools for comparing embeddings across different data modalities
- **Robust Processing**: Error handling and fallback mechanisms for production use

## Installation

```bash
pip install platonic-universe
```

For development:
```bash
git clone https://github.com/platonic-universe/platonic-universe.git
cd platonic-universe
pip install -e .[dev]
```

## Quick Start

```python
from platonic_universe import VisionModelLoader, DataManager, WorkflowRunner
from platonic_universe.cache import setup_cache

# Setup custom cache directory
setup_cache("/path/to/your/cache")

# Load a vision model
loader = VisionModelLoader("dinov2")
model = loader.load_model("facebook/dinov2-base")

# Run a complete workflow
runner = WorkflowRunner()
results = runner.run_embedding_comparison(
    model_a="dinov2",
    model_b="vit", 
    dataset="hsc_sdss",
    max_samples=1000
)

print(f"Cross-modal alignment score: {results['mknn_score']:.4f}")
```

## Architecture

The package is organized into several key modules:

- `platonic_universe.models`: Base classes and model implementations
- `platonic_universe.data`: Data loading and preprocessing utilities  
- `platonic_universe.cache`: Cache management and configuration
- `platonic_universe.workflows`: High-level analysis pipelines
- `platonic_universe.utils`: Common utilities and helper functions

## Supported Models

- **DinoV2**: Facebook's self-supervised vision transformer
- **ViT**: Google's Vision Transformer (Base, Large, Huge variants)
- **I-JEPA**: Meta's Image-based Joint Embedding Predictive Architecture
- **SpecFormer**: Specialized transformer for spectroscopic data

## Supported Datasets

- HSC × SDSS cross-matched data
- HSC × JWST cross-matched data  
- DESI × HSC cross-matched data
- Custom dataset integration

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.