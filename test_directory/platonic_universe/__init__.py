"""
Platonic Universe: A modular Python package for multimodal astronomical data analysis.

This package provides streamlined workflows for model loading, data fetching, 
embedding generation, and k-nearest neighbor comparisons across different 
astronomical datasets and vision models.
"""

__version__ = "0.1.0"
__author__ = "Platonic Universe Team"
__email__ = "team@platonic-universe.org"

# Core imports
from . import models, data, cache, workflows, utils

# Convenience imports for common use cases
from .models import ModelLoader, BaseVisionModel
from .data import DatasetLoader, flux_to_pil, spectrum_to_pil
from .cache import setup_cache, pick_cache_root
from .utils import compute_mknn_prh, KNNAnalyzer, set_seed
from .workflows import WorkflowRunner

# Make key classes easily accessible
__all__ = [
    # Modules
    "models",
    "data", 
    "cache",
    "workflows",
    "utils",
    
    # Key classes and functions
    "ModelLoader",
    "BaseVisionModel",
    "DatasetLoader",
    "WorkflowRunner", 
    "KNNAnalyzer",
    
    # Utility functions
    "flux_to_pil",
    "spectrum_to_pil",
    "setup_cache",
    "pick_cache_root",
    "compute_mknn_prh",
    "set_seed",
    
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
]


def get_version():
    """Get the package version."""
    return __version__


def get_info():
    """Get package information."""
    return {
        "name": "platonic-universe",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": "A modular Python package for multimodal astronomical data analysis",
    }