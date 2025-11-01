import os
import logging
from types import SimpleNamespace
from typing import Optional, Dict, Any


# Public helpers for programmatic use (wrappers around your CLI handlers)
# Lazy imports to avoid loading transformers/torchvision when only using metrics
_run_experiment = None
_mknn = None

def _get_run_experiment():
    global _run_experiment
    if _run_experiment is None:
        from .experiments import run_experiment
        _run_experiment = run_experiment
    return _run_experiment

def _get_mknn():
    global _mknn
    if _mknn is None:
        from .metrics import run_mknn_comparison
        _mknn = run_mknn_comparison
    return _mknn

_log = logging.getLogger(__name__)
PU_CACHE_DIR: Optional[str] = None

def setup_cache_dir(path: str) -> None:
    """
    Set a directory for caches used by the package and external libs (HuggingFace, XDG).
    Creates the dir if it does not exist and sets HF_HOME / XDG_CACHE_HOME env vars.
    """
    global PU_CACHE_DIR
    os.makedirs(path, exist_ok=True)
    os.environ.setdefault("HF_HOME", path)
    os.environ.setdefault("XDG_CACHE_HOME", path)
    PU_CACHE_DIR = path
    _log.info("Cache dir set to %s", path)

def compare_models_mknn(parquet_file: str) -> Dict[str, Any]:
    """
    Wrapper around the mknn comparison function to return results as a dictionary.
    Accepts the path to a parquet file produced by `run_experiment`.
    """
    return _get_mknn()(parquet_file)

def run_experiment(*args, **kwargs):
    """
    Lazily imported wrapper around experiments.run_experiment.
    This avoids loading transformers/torchvision unless actually needed.
    """
    return _get_run_experiment()(*args, **kwargs)

__all__ = ["setup_cache_dir", "compare_models_mknn", "run_experiment"] # "get_specformer_embeddings"]
