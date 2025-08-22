import logging
import os
from pathlib import Path
from ._registry import DATASET_REGISTRY # Import the registry from the local file

# Delay HuggingFace imports until needed to allow environment setup first
def _import_datasets():
    """Lazy import of datasets to allow environment setup first."""
    global load_dataset, Dataset, concatenate_datasets, IterableDataset
    if 'load_dataset' not in globals():
        from datasets import load_dataset, Dataset, concatenate_datasets, IterableDataset
    return load_dataset, Dataset, concatenate_datasets, IterableDataset

def list_available_datasets():
    """Returns a list of available dataset aliases and their descriptions."""
    return {alias: info["description"] for alias, info in DATASET_REGISTRY.items()}

def _check_dataset_exists_locally(repo_id: str, cache_dir: str = None) -> bool:
    """Check if dataset already exists in local cache."""
    try:
        # Get cache directory
        if cache_dir is None:
            cache_dir = os.environ.get('HF_DATASETS_CACHE', os.path.expanduser('~/.cache/huggingface/datasets'))
        
        # Check if any version of this dataset exists locally
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            return False
            
        # Look for dataset cache directories (simplified check)
        repo_name = repo_id.replace('/', '___')
        for item in cache_path.iterdir():
            if item.is_dir() and repo_name in item.name:
                logging.info(f"Found existing cache for dataset '{repo_id}' at {item}")
                return True
        return False
    except Exception:
        return False

def load_dataset_from_alias(alias: str, streaming: bool = None, max_samples: int = None, **kwargs):
    """
    Loads a pre-defined dataset using a short alias with streaming support.

    Args:
        alias (str): The short name of the dataset to load (e.g., 'mydata').
        streaming (bool): If True, use streaming mode. If None, auto-detect based on cache.
        max_samples (int): Maximum number of samples to load (only applies to streaming mode).
        **kwargs: Additional keyword arguments to pass to `datasets.load_dataset`.

    Returns:
        Dataset or IterableDataset: The loaded dataset object.
    
    Raises:
        ValueError: If the alias is not found in the registry.
    """
    load_dataset, Dataset, concatenate_datasets, IterableDataset = _import_datasets()
    
    if alias not in DATASET_REGISTRY:
        available = list(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset alias '{alias}'. Available options are: {available}")

    dataset_info = DATASET_REGISTRY[alias]
    repo_id = dataset_info["repo_id"]
    
    # Set the default split, but allow the user to override it via kwargs
    if 'split' not in kwargs:
        kwargs['split'] = dataset_info.get("default_split")

    # Auto-detect streaming mode if not specified
    if streaming is None:
        cache_dir = kwargs.get('cache_dir')
        if _check_dataset_exists_locally(repo_id, cache_dir):
            streaming = False
            logging.info(f"Dataset '{alias}' found in cache, using downloaded version")
        else:
            streaming = True
            logging.info(f"Dataset '{alias}' not in cache, using streaming mode")
    
    # Use streaming mode
    if streaming:
        logging.info(f"Streaming dataset '{alias}' from Hugging Face repo '{repo_id}'...")
        kwargs['streaming'] = True
        try:
            dataset = load_dataset(repo_id, **kwargs)
            if max_samples is not None:
                logging.info(f"Limiting stream to {max_samples} samples")
                dataset = dataset.take(max_samples)
            logging.info(f"Dataset '{alias}' streaming initialized successfully.")
            return dataset
        except Exception as e:
            logging.error(f"Failed to stream dataset '{alias}' from '{repo_id}'. Error: {e}")
            raise
    else:
        # Use downloaded mode
        logging.info(f"Loading dataset '{alias}' from cache/download '{repo_id}'...")
        kwargs.pop('streaming', None)  # Remove streaming flag if present
        try:
            dataset = load_dataset(repo_id, **kwargs)
            if max_samples is not None:
                logging.info(f"Selecting first {max_samples} samples from downloaded dataset")
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            logging.info(f"Dataset '{alias}' loaded successfully.")
            return dataset
        except Exception as e:
            logging.error(f"Failed to load dataset '{alias}' from '{repo_id}'. Error: {e}")
            raise


def load_hsc_sdss_crossmatched(streaming: bool = None, max_samples: int = None, **kwargs):
    """
    Load HSC-SDSS crossmatched data following the pattern from the script.
    
    This concatenates:
    - Smith42/sdss_hsc_crossmatched (HSC images)  
    - Shashwat20/SDSS_Interpolated (SDSS embeddings)
    
    Args:
        streaming: If True, use streaming mode. If None, auto-detect based on cache.
        max_samples: Maximum number of samples to load.
        **kwargs: Additional arguments passed to load_dataset.
    
    Returns:
        Dataset or IterableDataset: Combined dataset with both HSC images and SDSS embeddings
    """
    load_dataset, Dataset, concatenate_datasets, IterableDataset = _import_datasets()
    
    logging.info("Loading HSC-SDSS crossmatched data...")
    
    # Auto-detect streaming for both datasets
    if streaming is None:
        cache_dir = kwargs.get('cache_dir')
        hsc_sdss_cached = _check_dataset_exists_locally("Smith42/sdss_hsc_crossmatched", cache_dir)
        sdss_interp_cached = _check_dataset_exists_locally("Shashwat20/SDSS_Interpolated", cache_dir)
        streaming = not (hsc_sdss_cached and sdss_interp_cached)
        
        if streaming:
            logging.info("One or both HSC-SDSS datasets not in cache, using streaming mode")
        else:
            logging.info("Both HSC-SDSS datasets found in cache, using downloaded versions")
    
    try:
        if streaming:
            # Stream both datasets
            kwargs['streaming'] = True
            hsc_sdss = load_dataset("Smith42/sdss_hsc_crossmatched", split="train", **kwargs)
            sdss_interpolated = load_dataset("Shashwat20/SDSS_Interpolated", split="train", **kwargs)
            
            if max_samples is not None:
                logging.info(f"Limiting each stream to {max_samples} samples")
                hsc_sdss = hsc_sdss.take(max_samples)
                sdss_interpolated = sdss_interpolated.take(max_samples)
            
            # Note: For streaming datasets, we can't concatenate columns directly
            # We'll need to handle this at the processing level
            logging.info("HSC-SDSS crossmatched streaming initialized successfully.")
            return hsc_sdss, sdss_interpolated  # Return both for manual handling
        else:
            # Load downloaded datasets
            kwargs.pop('streaming', None)
            hsc_sdss = load_dataset("Smith42/sdss_hsc_crossmatched", split="train", **kwargs)
            sdss_interpolated = load_dataset("Shashwat20/SDSS_Interpolated", split="train", **kwargs)
            
            if max_samples is not None:
                logging.info(f"Selecting first {max_samples} samples from each dataset")
                hsc_sdss = hsc_sdss.select(range(min(max_samples, len(hsc_sdss))))
                sdss_interpolated = sdss_interpolated.select(range(min(max_samples, len(sdss_interpolated))))
            
            # Concatenate along axis=1 (columns)
            combined = concatenate_datasets([hsc_sdss, sdss_interpolated], axis=1)
            
            # Rename image column to hsc_image for consistency
            if "image" in combined.column_names:
                combined = combined.rename_column("image", "hsc_image")
            
            logging.info("HSC-SDSS crossmatched data loaded successfully.")
            return combined
        
    except Exception as e:
        logging.error(f"Failed to load HSC-SDSS crossmatched data. Error: {e}")
        raise


# Dataset column mapping for auto-detection
DATASET_COLUMN_MAPPING = {
    "desi-hsc": {
        "columns": ("image", "spectrum"),
        "labels": ("hsc", "desi"),
        "loader": "standard"
    },
    "hsc-sdss": {
        "columns": ("hsc_image", "embedding"), 
        "labels": ("hsc", "sdss"),
        "loader": "hsc_sdss_crossmatched"
    },
    "hsc-jwst": {
        "columns": ("hsc_image", "jwst_image"),
        "labels": ("hsc", "jwst"), 
        "loader": "standard"
    },
    "hsc-legacy": {
        "columns": ("hsc_image", "legacy_image"),
        "labels": ("hsc", "legacy"),
        "loader": "standard"
    },
    "desi-spectroscopic": {
        "columns": ("embeddings",),
        "labels": ("desi",),
        "loader": "standard"
    },
    "sdss-interpolated": {
        "columns": ("embedding",),
        "labels": ("sdss",),
        "loader": "standard"
    }
}


def get_dataset_info(dataset_alias: str) -> dict:
    """
    Get column mapping and metadata for a dataset alias.
    
    Args:
        dataset_alias: The dataset alias (e.g., "desi-hsc", "hsc-sdss")
        
    Returns:
        dict: Contains columns, labels, and loader info for the dataset
    """
    if dataset_alias not in DATASET_COLUMN_MAPPING:
        raise ValueError(f"Unknown dataset alias '{dataset_alias}'. Available: {list(DATASET_COLUMN_MAPPING.keys())}")
    
    return DATASET_COLUMN_MAPPING[dataset_alias]


def load_dataset_with_info(dataset_alias: str, streaming: bool = None, max_samples: int = None, **kwargs):
    """
    Load dataset using the appropriate loader based on dataset type.
    
    Args:
        dataset_alias: The dataset alias
        streaming: If True, use streaming mode. If None, auto-detect based on cache.
        max_samples: Maximum number of samples to load.
        **kwargs: Additional arguments passed to the loader
        
    Returns:
        tuple: (dataset, dataset_info_dict)
    """
    dataset_info = get_dataset_info(dataset_alias)
    
    if dataset_info["loader"] == "hsc_sdss_crossmatched":
        dataset = load_hsc_sdss_crossmatched(streaming=streaming, max_samples=max_samples, **kwargs)
    else:
        dataset = load_dataset_from_alias(dataset_alias, streaming=streaming, max_samples=max_samples, **kwargs)
    
    return dataset, dataset_info