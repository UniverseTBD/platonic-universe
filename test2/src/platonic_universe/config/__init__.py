# Import the function from the set_cache_dir.py module
from .set_cache_dir import setup_hf_environment

# Expose it so it can be imported from this subpackage
__all__ = [
    "setup_hf_environment",
]