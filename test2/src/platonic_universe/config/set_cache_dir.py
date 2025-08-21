# This file sets up cache directories

import os
import pathlib

def setup_hf_environment(base_cache_dir=None, verbose_env=False, **overrides):
    """
    Sets up the Hugging Face environment variables with modular user control and error handling.
    
    Args:
        base_cache_dir (str, optional): The root directory for the cache. 
            Defaults to './hf_cache' in the current working directory.
        verbose_env (bool, optional): If True, prints all environment variables and their paths.
            Defaults to False (only shows cache directory).
        **overrides: Keyword arguments to override specific environment variables.
    
    Raises:
        ValueError: If an unrecognized keyword argument is provided.
        IOError: If a cache directory cannot be created due to permission errors.
    """
    # 1. Determine the base cache directory
    if base_cache_dir:
        root = pathlib.Path(base_cache_dir).resolve()
    else:
        root = pathlib.Path.cwd() / "hf_cache"

    # 2. Define the default paths using absolute paths
    defaults = {
        'HF_HOME': str(root.resolve()),
        'HF_DATASETS_CACHE': str((root / "datasets").resolve()),
        'HF_HUB_CACHE': str((root / "hub").resolve()),
        'TRANSFORMERS_CACHE': str((root / "transformers").resolve()),
        'HF_DATASETS_OFFLINE': '0',
        'TRANSFORMERS_OFFLINE': '0',
        'HUGGINGFACE_HUB_CACHE': str((root / "hub").resolve()),
        'HF_DATASETS_DOWNLOAD_MODE': 'reuse_dataset_if_exists_else_download',
        'HF_HUB_ENABLE_HF_TRANSFER': '0'
    }

    # 3. Apply any valid user overrides
    final_settings = defaults.copy()
    valid_override_keys = {k.lower() for k in defaults.keys()}
    
    for key, value in overrides.items():
        if key.lower() in valid_override_keys:
            env_var_key = key.upper()
            final_settings[env_var_key] = str(pathlib.Path(value).resolve())
            print(f"🔧 Overriding {env_var_key} with '{value}'")
        else:
            # ERROR HANDLING 1: Catch invalid keyword arguments
            raise ValueError(
                f"Unrecognized argument '{key}'. Valid environment variables that you can set are: {sorted(valid_override_keys)}"
            )

    # 4. Create all necessary directories
    try:
        for key, value in final_settings.items():
            if key.endswith("_CACHE") or key == 'HF_HOME':
                pathlib.Path(value).mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        # ERROR HANDLING 2: Catch filesystem errors
        raise IOError(
            f"Failed to create cache directory '{e.filename}'. "
            f"Please check your write permissions for that location. Original error: {e}"
        )
            
    # 5. Set all environment variables
    os.environ.update(final_settings)
    
    # 6. Display environment information
    if verbose_env:
        print(f"✅ Hugging Face environment configured with the following paths:")
        for key, value in final_settings.items():
            print(f"   {key}: {value}")
    else:
        print(f"✅ Hugging Face environment configured. HF_HOME is '{final_settings['HF_HOME']}'")

