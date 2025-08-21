# CRITICAL: Set up HuggingFace environment BEFORE any HF imports
import os
import pathlib
from pathlib import Path

def _auto_setup_hf_environment():
    """
    Automatically set up HuggingFace environment variables during package import.
    This ensures proper cache directory setup before any HF modules are imported.
    """
    # Check if environment variables are already set
    if 'HF_HOME' in os.environ:
        return  # Already configured, don't override
    
    # Default cache directory in current working directory
    cache_dir = Path.cwd() / "hf_cache"
    
    # Set all HuggingFace environment variables
    env_vars = {
        'HF_HOME': str(cache_dir.resolve()),
        'HF_DATASETS_CACHE': str((cache_dir / "datasets").resolve()),
        'HF_HUB_CACHE': str((cache_dir / "hub").resolve()),
        'TRANSFORMERS_CACHE': str((cache_dir / "transformers").resolve()),
        'HUGGINGFACE_HUB_CACHE': str((cache_dir / "hub").resolve()),
        'HF_DATASETS_OFFLINE': '0',
        'TRANSFORMERS_OFFLINE': '0',
        'HF_DATASETS_DOWNLOAD_MODE': 'reuse_dataset_if_exists_else_download',
        'HF_HUB_ENABLE_HF_TRANSFER': '0'
    }
    
    # Set environment variables
    os.environ.update(env_vars)
    
    # Create cache directories
    for path_str in [env_vars['HF_DATASETS_CACHE'], env_vars['HF_HUB_CACHE'], env_vars['TRANSFORMERS_CACHE']]:
        Path(path_str).mkdir(parents=True, exist_ok=True)

# Auto-setup environment during import
_auto_setup_hf_environment()

# Now import everything else (with proper environment already set)
from .config import setup_hf_environment
from .pipelines.main import run_multi_model_comparison, compare_models_mknn

def setup_cache_dir(cache_dir):
    """
    Override the default cache directory after import.
    
    Args:
        cache_dir (str): Path to the desired cache directory
        
    Note: This should be called immediately after import for best results.
    """
    cache_path = Path(cache_dir).resolve()
    
    env_vars = {
        'HF_HOME': str(cache_path),
        'HF_DATASETS_CACHE': str(cache_path / "datasets"),
        'HF_HUB_CACHE': str(cache_path / "hub"),
        'TRANSFORMERS_CACHE': str(cache_path / "transformers"),
        'HUGGINGFACE_HUB_CACHE': str(cache_path / "hub"),
    }
    
    os.environ.update(env_vars)
    
    # Create directories
    for path_str in [env_vars['HF_DATASETS_CACHE'], env_vars['HF_HUB_CACHE'], env_vars['TRANSFORMERS_CACHE']]:
        Path(path_str).mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Cache directory updated to: {cache_path}")

# Define what gets imported with 'from platonic_universe import *'
__all__ = [
    "setup_hf_environment",
    "setup_cache_dir",
    "run_multi_model_comparison", 
    "compare_models_mknn",
]
