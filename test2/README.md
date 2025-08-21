# This is a package that reproduces all the results of the Platonic Representation Hypothesis paper.

## Building

Either use the provided `.whl` file for the latest build, or navigate to this directory, and do `python -m hatchling build --target wheel`. It is recommended that before this, you do `uv sync` in the same directory as the `uv.lock` file (located one level above this one), which creates a `.venv` folder (invisible by default), that you can activate with `source .venv/bin/activate`. 

## Usage

After installing the package, you can run model comparisons using the high-level API. You can see the models and available datasets in `src/models/_registry.py' or 'src/datasets/_registry.py`. Be default, all huggingface cache directories are set to local. Here's a basic example:

```python
import platonic_universe as pu
import logging

# Optional: Set custom cache directory
pu.setup_cache_dir("./analysis_cache")

# Configuration
MODELS_TO_COMPARE = ["vit-base", "vit-large", "vit-huge"]
DATASET_ALIAS = "desi-hsc"  # Ensure this alias exists in your dataset registry
MAX_SAMPLES = 100
BATCH_SIZE = 16

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Run analysis for each model
for model_alias in MODELS_TO_COMPARE:
    try:
        results = pu.compare_models_mknn(
            model_alias=model_alias,
            dataset_alias=DATASET_ALIAS,
            max_samples=MAX_SAMPLES,
            use_lut_normalization=True,
            use_simple_mknn=True,
            use_bulk_processing=True,
        )
        
        print(f"\n--- Results for {model_alias} ---")
        print(f"MKNN Score: {results['mknn_score']:.4f}")
        print(f"Aligned Pairs Processed: {results['aligned_pairs_count']}")
        
    except Exception as e:
        logging.error(f"Failed to process model '{model_alias}'. Error: {e}")
```

You can also run the included sample script:

```bash
python sample_script.py
``` 
