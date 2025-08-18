"""
Data loading and preprocessing utilities for Platonic Universe.

This module provides functions for loading astronomical datasets,
preprocessing images and spectra, and managing data streams.
"""

from .loaders import (
    DatasetLoader,
    load_hsc_sdss,
    load_hsc_jwst, 
    load_desi_hsc,
    load_custom_dataset,
)

from .preprocessors import (
    flux_to_pil,
    spectrum_to_pil,
    interpolate_spectrum,
    ImagePreprocessor,
    SpectrumPreprocessor,
)

__all__ = [
    "DatasetLoader",
    "load_hsc_sdss",
    "load_hsc_jwst",
    "load_desi_hsc",
    "load_custom_dataset",
    "flux_to_pil",
    "spectrum_to_pil",
    "interpolate_spectrum",
    "ImagePreprocessor", 
    "SpectrumPreprocessor",
]