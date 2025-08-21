import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path

def save_analysis_results(
    output_dir: str,
    model_variant: str,
    embeddings: np.ndarray,
    lookup_table: list,
    report_data: dict
):
    """
    Saves all artifacts from a single model analysis run to an output directory.

    Args:
        output_dir (str): The base directory to save the files in.
        model_variant (str): The name of the model variant, used for file naming.
        embeddings (np.ndarray): The combined embedding matrix.
        lookup_table (list): The corresponding lookup table.
        report_data (dict): A dictionary containing summary statistics and the MKNN score.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving results for '{model_variant}' to '{output_path}'...")

    # 1. Save embeddings matrix
    embed_file = output_path / f"embeddings_{model_variant}.npy"
    np.save(embed_file, embeddings)

    # 2. Save lookup table as both CSV and Pickle
    lookup_file_csv = output_path / f"lookup_{model_variant}.csv"
    lookup_file_pkl = output_path / f"lookup_{model_variant}.pkl"
    pd.DataFrame(lookup_table).to_csv(lookup_file_csv, index=False)
    with open(lookup_file_pkl, 'wb') as f:
        pickle.dump(lookup_table, f)

    # 3. Save summary text file
    report_file = output_path / f"summary_{model_variant}.txt"
    with open(report_file, 'w') as f:
        f.write(f"Model: {report_data['model_name']}\n")
        f.write("="*30 + "\n")
        f.write(f"MKNN Score (k={report_data['k']}): {report_data['mknn_score']:.4f}\n")
        f.write(f"JWST embeddings: {report_data['jwst_embeddings']}\n")
        f.write(f"HSC embeddings: {report_data['hsc_embeddings']}\n")
        f.write(f"Embedding dimension: {report_data['embedding_dimension']}\n")

    logging.info(f"Successfully saved all files for '{model_variant}'.")