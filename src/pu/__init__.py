import os
import logging
from typing import Optional, Dict, Any, List


# Public helpers for programmatic use (wrappers around your CLI handlers)
# Lazy imports to avoid loading transformers/torchvision when only using metrics
_run_experiment = None
_run_layerwise_comparison = None
_run_size_convergence_study = None
_run_metric_consistency_study = None

def _get_run_experiment():
    global _run_experiment
    if _run_experiment is None:
        from .experiments import run_experiment
        _run_experiment = run_experiment
    return _run_experiment

def _get_layerwise_functions():
    global _run_layerwise_comparison, _run_size_convergence_study, _run_metric_consistency_study
    if _run_layerwise_comparison is None:
        from .experiments_layerwise import (
            run_layerwise_comparison,
            run_size_convergence_study, 
            run_metric_consistency_study
        )
        _run_layerwise_comparison = run_layerwise_comparison
        _run_size_convergence_study = run_size_convergence_study
        _run_metric_consistency_study = run_metric_consistency_study
    return _run_layerwise_comparison, _run_size_convergence_study, _run_metric_consistency_study

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

def compare_models(parquet_file: str, metrics: list[str] | None = None, **kwargs) -> Dict[str, Any]:
    """
    Compare models using the new metrics API.
    Accepts the path to a parquet file produced by `run_experiment`.

    Args:
        parquet_file: Path to parquet file with embeddings
        metrics: List of metric names to compute (default: all)
        **kwargs: Additional arguments passed to metrics

    Returns:
        Dictionary with model info and computed metrics
    """
    from pu.metrics import compare_from_parquet
    return compare_from_parquet(parquet_file, metrics=metrics, **kwargs)

def run_experiment(*args, **kwargs):
    """
    Lazily imported wrapper around experiments.run_experiment.
    This avoids loading transformers/torchvision unless actually needed.
    """
    return _get_run_experiment()(*args, **kwargs)


def run_layerwise_comparison(*args, **kwargs):
    """
    Compare representations layer-by-layer between two models.
    
    See experiments_layerwise.run_layerwise_comparison for full documentation.
    """
    fn, _, _ = _get_layerwise_functions()
    return fn(*args, **kwargs)


def run_size_convergence_study(*args, **kwargs):
    """
    Study convergence of representations with model size.
    
    Compares adjacent size pairs (small->medium, medium->large) to test
    if alignment increases with model scale.
    
    See experiments_layerwise.run_size_convergence_study for full documentation.
    """
    _, fn, _ = _get_layerwise_functions()
    return fn(*args, **kwargs)


def run_metric_consistency_study(*args, **kwargs):
    """
    Study consistency of optimal layer pairs across different metrics.
    
    See experiments_layerwise.run_metric_consistency_study for full documentation.
    """
    _, _, fn = _get_layerwise_functions()
    return fn(*args, **kwargs)


# Import metrics submodule for pu.metrics.* access
from pu import metrics

__all__ = [
    "setup_cache_dir", 
    "compare_models", 
    "run_experiment", 
    "run_layerwise_comparison",
    "run_size_convergence_study",
    "run_metric_consistency_study",
    "metrics",
]
