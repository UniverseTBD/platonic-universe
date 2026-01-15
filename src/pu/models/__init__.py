from .registry import get_adapter, register_adapter, list_adapters

# Import adapters so they register themselves on package import.
# This keeps registry population implicit and backwards-compatible.
# The imports are intentionally unused here; they only trigger registration.
from . import hf
from . import astropt 
from . import sam2
from . import smolvlm 

__all__ = ["get_adapter", "register_adapter", "list_adapters"]
