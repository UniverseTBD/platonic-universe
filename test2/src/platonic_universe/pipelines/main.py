import torch
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path

# Import all the building blocks from your library
from ..datasets.download_datasets import load_dataset_from_alias, load_dataset_with_info, get_dataset_info
from ..models.loading import load_model_from_alias
from ..models.embedders import get_embedder
from ..processing.main import process_paired_dataset
from ..postprocessing.metrics import mknn_score, compute_mknn_simple, mknn_score_auto, compute_mknn_simple_auto
from ..io.saving import save_analysis_results

def run_comparison_pipeline(
    model_alias: str,
    dataset_alias: str,
    output_dir: str,
    batch_size: int = 32,
    max_samples: int = None,
    k: int = 10
) -> dict:
    """
    Runs the full end-to-end pipeline for a single model.
    """
    logging.info(f"\n{'='*80}\nPROCESSING MODEL: {model_alias}\n{'='*80}")
    
    # 1. Load data and model
    dataset = load_dataset_from_alias(dataset_alias)
    model_obj = load_model_from_alias(model_alias)
    embedder = get_embedder(model_obj)
    
    # 2. Process both JWST and HSC image columns
    jwst_embeddings, jwst_lookup = process_dataset(
        dataset, embedder, "jwst_image", "jwst", batch_size, max_samples)
    
    hsc_embeddings, hsc_lookup = process_dataset(
        dataset, embedder, "hsc_image", "hsc", batch_size, max_samples)

    # Clean up GPU memory after processing
    del model_obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    if not len(jwst_embeddings) or not len(hsc_embeddings):
        logging.warning(f"Could not compute embeddings for model '{model_alias}'. Skipping.")
        return {'error': 'No embeddings generated.'}

    # 3. Combine results
    all_embeddings = np.vstack([jwst_embeddings, hsc_embeddings])
    all_lookup = jwst_lookup + hsc_lookup
    for i, entry in enumerate(all_lookup):
        entry['combined_embedding_index'] = i

    # 4. Post-process: Calculate MKNN score
    score = mknn_score(jwst_embeddings, hsc_embeddings, k=k)
    
    # 5. Assemble report and save all results
    model_variant = embedder.model.name_or_path.split('/')[-1]
    report_data = {
        'model_name': embedder.model.name_or_path,
        'model_variant': model_variant,
        'mknn_score': score,
        'k': k,
        'jwst_embeddings': len(jwst_embeddings),
        'hsc_embeddings': len(hsc_embeddings),
        'embedding_dimension': all_embeddings.shape[1]
    }
    save_analysis_results(output_dir, model_variant, all_embeddings, all_lookup, report_data)
    
    return report_data

def run_multi_model_comparison(
    model_aliases: list[str],
    dataset_alias: str,
    output_dir: str,
    **kwargs
):
    """
    Runs the comparison pipeline for a list of models and generates a final summary.
    """
    results = {}
    for model_alias in model_aliases:
        try:
            result = run_comparison_pipeline(model_alias, dataset_alias, output_dir, **kwargs)
            results[model_alias] = result
        except Exception as e:
            logging.error(f"Pipeline failed for model '{model_alias}'. Error: {e}")
            results[model_alias] = {'error': str(e)}

    # Generate and save a final summary report
    logging.info(f"\n{'='*80}\nMULTI-MODEL PROCESSING SUMMARY\n{'='*80}")
    summary_data = []
    for model_alias, result in results.items():
        if 'error' not in result:
            summary_data.append(result)
            logging.info(f"✅ {result['model_variant']}: MKNN Score = {result['mknn_score']:.4f}")
        else:
            logging.error(f"❌ {model_alias}: FAILED - {result['error']}")

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = Path(output_dir) / "multi_model_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        logging.info(f"\nFinal summary report saved to '{summary_csv_path}'")

def compare_models_mknn(
    model_alias: str,
    dataset_alias: str,
    k: int = 10,
    batch_size: int = 32,
    max_samples: int = None,
    use_lut_normalization: bool = False,
    use_simple_mknn: bool = False,
    use_bulk_processing: bool = False,
    streaming: bool = None,
    show_progress: bool = False,
    save_results: bool = False,
    results_file: str = "./mknn_results.json"
) -> dict:
    """
    Compare embeddings from two modalities using mutual k-NN.
    Auto-detects column names based on dataset alias.
    
    Args:
        model_alias: Model to use for embedding (e.g., "ijepa-vith14-22k")
        dataset_alias: Dataset to process (e.g., "desi-hsc", "hsc-sdss") 
        k: Number of neighbors for mutual k-NN
        batch_size: Batch size for processing
        max_samples: Maximum samples to process (None for all)
        use_lut_normalization: If True, use lookup table normalization; if False, use on-the-fly normalization
        use_simple_mknn: If True, use compute_mknn_simple (cosine distance); if False, use mknn_score (L2+Euclidean)
        use_bulk_processing: If True, preprocess all images into RAM first, then bulk embed for speed
        streaming: If True, use streaming mode. If None, auto-detect based on cache.
        show_progress: If True, show progress bars during processing
        save_results: If True, save MKNN results to file (default: False)
        results_file: Path to save results file (default: "./mknn_results.json")
        
    Returns:
        dict: Results with mknn_score and metadata
    """
    logging.info(f"Starting MKNN comparison for '{model_alias}' on '{dataset_alias}'")
    
    # Auto-detect dataset type and columns
    dataset_info = get_dataset_info(dataset_alias)
    
    # Special handling for desi-hsc-shuffled which needs model_alias during loading
    if dataset_alias == "desi-hsc-shuffled":
        dataset, _ = load_dataset_with_info(dataset_alias, streaming=streaming, max_samples=max_samples, model_alias=model_alias, show_progress=show_progress)
    else:
        dataset, _ = load_dataset_with_info(dataset_alias, streaming=streaming, max_samples=max_samples)
    
    model_obj = load_model_from_alias(model_alias)
    embedder = get_embedder(model_obj)

    columns = dataset_info["columns"]
    labels = dataset_info["labels"]
    
    if len(columns) == 2:
        # Two-modality comparison (e.g., HSC-DESI, HSC-SDSS, HSC-JWST)
        embeddings_1, _, embeddings_2, _ = process_paired_dataset(
            dataset=dataset,
            embedder=embedder,
            column1=columns[0], label1=labels[0],
            column2=columns[1], label2=labels[1],
            batch_size=batch_size,
            max_samples=None if streaming else max_samples,  # max_samples already applied in streaming
            dataset_alias=dataset_alias,
            use_lut_normalization=use_lut_normalization,
            use_bulk_processing=use_bulk_processing
        )
        
        del model_obj, embedder
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        logging.info("--- Calculating Final MKNN Score ---")
        if use_simple_mknn:
            score = compute_mknn_simple_auto(embeddings_1, embeddings_2, k=k)
            mknn_method = "simple"
        else:
            score = mknn_score_auto(embeddings_1, embeddings_2, k=k)
            mknn_method = "standard"
        aligned_pairs_count = len(embeddings_1)
        
    elif len(columns) == 1:
        # Single-modality dataset (e.g., pre-computed embeddings)
        raise NotImplementedError("Single-modality MKNN comparison not yet implemented")
    else:
        raise ValueError(f"Unsupported number of columns: {len(columns)}")

    result = {
        'model': model_alias,
        'dataset': dataset_alias,
        'mknn_score': score,
        'mknn_method': mknn_method,
        'k': k,
        'aligned_pairs_count': aligned_pairs_count,
        'modalities': f"{labels[0]} ↔ {labels[1]}" if len(labels) == 2 else labels[0],
        'timestamp': datetime.now().isoformat(),
        'batch_size': batch_size,
        'max_samples': max_samples,
        'use_lut_normalization': use_lut_normalization,
        'use_bulk_processing': use_bulk_processing,
        'streaming': streaming
    }
    
    # Save results to file if requested
    if save_results:
        try:
            results_path = Path(results_file)
            
            # Load existing results if file exists
            if results_path.exists():
                with open(results_path, 'r') as f:
                    existing_results = json.load(f)
                if not isinstance(existing_results, list):
                    existing_results = [existing_results]
            else:
                existing_results = []
            
            # Append new result
            existing_results.append(result)
            
            # Create directory if it doesn't exist
            results_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save updated results
            with open(results_path, 'w') as f:
                json.dump(existing_results, f, indent=2)
            
            logging.info(f"Results saved to '{results_path}'")
            
        except Exception as e:
            logging.warning(f"Failed to save results to '{results_file}': {e}")
    
    logging.info(f"Comparison complete for '{model_alias}'. MKNN Score: {score:.4f}")
    return result
