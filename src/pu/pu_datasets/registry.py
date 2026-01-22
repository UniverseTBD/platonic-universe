from typing import Type, Dict, List

_REGISTRY: Dict[str, Type] = {}

def register_dataset(alias: str, adapter_cls: Type) -> None:
    """Register a dataset adapter class under a short alias."""
    _REGISTRY[alias] = adapter_cls

def get_dataset_adapter(alias: str) -> Type:
    """Retrieve a registered dataset adapter class (not an instance). Raises KeyError if not found."""
    try:
        return _REGISTRY[alias]
    except KeyError as exc:
        raise KeyError(
            f"Dataset adapter for alias '{alias}' not found. Available adapters: {sorted(list(_REGISTRY.keys()))}"
        ) from exc

def list_datasets() -> list:
    """Return a sorted list of registered dataset adapter aliases."""
    return sorted(list(_REGISTRY.keys()))

def list_physical_params(alias: str) -> List[str]:
    """Return available physical parameters for a dataset."""
    adapter_cls = get_dataset_adapter(alias)
    return adapter_cls.list_physical_params()
