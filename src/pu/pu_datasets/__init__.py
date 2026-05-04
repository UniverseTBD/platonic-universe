# Import adapters so they register themselves on package import.
# These imports are unused directly but trigger registration side-effects.
from . import (
    cosmosweb,  # noqa: F401
    desi,  # noqa: F401
    desi_spectra,  # noqa: F401
    galaxies,  # noqa: F401
    hf_crossmatched,  # noqa: F401
    sdss,  # noqa: F401
)
from .registry import get_dataset_adapter, list_datasets, register_dataset

__all__ = ["register_dataset", "get_dataset_adapter", "list_datasets"]
