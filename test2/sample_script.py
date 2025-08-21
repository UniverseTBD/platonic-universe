#!/usr/bin/env python3
"""
Simplified quicktest using integrated environment setup
"""

import platonic_universe as pu
import logging

# Optional: Override default cache directory
pu.setup_cache_dir("./analysis_cache")

def main():
    """
    Runs a sample analysis using the platonic_universe library's high-level pipelines.
    """
    # --- 1. Configuration ---
    MODELS_TO_COMPARE = ["vit-base", "vit-large", "vit-huge"]
    DATASET_ALIAS = "desi-hsc" # Make sure this alias exists in your dataset registry
    MAX_SAMPLES = 100
    BATCH_SIZE = 16
    K_NEIGHBORS = 10
    
    # --- 2. Setup ---
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # --- 3. Main Loop: Run the Pipeline for Each Model ---
    for model_alias in MODELS_TO_COMPARE:
        try:
            # A single call now runs the entire workflow for one model:
            # load -> process paired data -> calculate score
            results = pu.compare_models_mknn(
                model_alias=model_alias,
                dataset_alias=DATASET_ALIAS,
                max_samples=MAX_SAMPLES,
                use_lut_normalization=True,
                use_simple_mknn=True,
                use_bulk_processing=True,
            )
            
            print(f"\n--- ✅ Results for {model_alias} ---")
            print(f"MKNN Score: {results['mknn_score']:.4f}")
            print(f"Aligned Pairs Processed: {results['aligned_pairs_count']}")

        except Exception as e:
            logging.error(f"Failed to process model '{model_alias}'. Error: {e}", exc_info=True)

if __name__ == "__main__":
    main()