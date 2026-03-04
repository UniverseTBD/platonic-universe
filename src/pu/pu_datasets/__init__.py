from .registry import register_dataset, get_dataset_adapter, list_datasets

# Import adapters so they register themselves on package import.
# These imports are unused directly but trigger registration side-effects.
from . import hf_crossmatched  # noqa: F401
from . import sdss  # noqa: F401
from . import desi  # noqa: F401
from . import galaxies  # noqa: F401

__all__ = ["register_dataset", "get_dataset_adapter", "list_datasets"]
