import logging
from ._registry import DATASET_REGISTRY # Import the registry from the local file

# Delay HuggingFace imports until needed to allow environment setup first
def _import_datasets():
    """Lazy import of datasets to allow environment setup first."""
    global load_dataset, Dataset, concatenate_datasets
    if 'load_dataset' not in globals():
        from datasets import load_dataset, Dataset, concatenate_datasets
    return load_dataset, Dataset, concatenate_datasets

def list_available_datasets():
    """Returns a list of available dataset aliases and their descriptions."""
    return {alias: info["description"] for alias, info in DATASET_REGISTRY.items()}

def load_dataset_from_alias(alias: str, **kwargs):
    """
    Loads a pre-defined dataset using a short alias.

    Args:
        alias (str): The short name of the dataset to load (e.g., 'mydata').
        **kwargs: Additional keyword arguments to pass to `datasets.load_dataset`,
                  which can override the defaults (e.g., split="test").

    Returns:
        Dataset: The loaded dataset object.
    
    Raises:
        ValueError: If the alias is not found in the registry.
    """
    load_dataset, Dataset, concatenate_datasets = _import_datasets()
    
    if alias not in DATASET_REGISTRY:
        available = list(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset alias '{alias}'. Available options are: {available}")

    dataset_info = DATASET_REGISTRY[alias]
    repo_id = dataset_info["repo_id"]
    
    # Set the default split, but allow the user to override it via kwargs
    if 'split' not in kwargs:
        kwargs['split'] = dataset_info.get("default_split")

    logging.info(f"Loading dataset '{alias}' from Hugging Face repo '{repo_id}'...")
    try:
        dataset = load_dataset(repo_id, **kwargs)
        logging.info(f"Dataset '{alias}' loaded successfully.")
        return dataset
    except Exception as e:
        logging.error(f"Failed to load dataset '{alias}' from '{repo_id}'. Error: {e}")
        raise


def load_hsc_sdss_crossmatched(**kwargs):
    """
    Load HSC-SDSS crossmatched data following the pattern from the script.
    
    This concatenates:
    - Smith42/sdss_hsc_crossmatched (HSC images)  
    - Shashwat20/SDSS_Interpolated (SDSS embeddings)
    
    Returns:
        Dataset: Combined dataset with both HSC images and SDSS embeddings
    """
    load_dataset, Dataset, concatenate_datasets = _import_datasets()
    
    logging.info("Loading HSC-SDSS crossmatched data...")
    
    try:
        # Load both datasets
        hsc_sdss = load_dataset("Smith42/sdss_hsc_crossmatched", split="train", **kwargs)
        sdss_interpolated = load_dataset("Shashwat20/SDSS_Interpolated", split="train", **kwargs)
        
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


def load_dataset_with_info(dataset_alias: str, **kwargs):
    """
    Load dataset using the appropriate loader based on dataset type.
    
    Args:
        dataset_alias: The dataset alias
        **kwargs: Additional arguments passed to the loader
        
    Returns:
        tuple: (dataset, dataset_info_dict)
    """
    dataset_info = get_dataset_info(dataset_alias)
    
    if dataset_info["loader"] == "hsc_sdss_crossmatched":
        dataset = load_hsc_sdss_crossmatched(**kwargs)
    else:
        dataset = load_dataset_from_alias(dataset_alias, **kwargs)
    
    return dataset, dataset_info