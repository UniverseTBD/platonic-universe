import torch
import logging
import numpy as np
from tqdm.auto import tqdm
from datasets import Dataset
from ..models.embedders import BaseEmbedder
from ..preprocessing.preparation import flux_to_pil, image_to_pil_with_lut

def process_paired_dataset(
    dataset: Dataset,
    embedder: BaseEmbedder,
    column1: str,
    label1: str,
    column2: str,
    label2: str,
    batch_size: int = 32,
    max_samples: int = None,
    dataset_alias: str = None,
    use_lut_normalization: bool = False,
    use_bulk_processing: bool = False
) -> tuple[np.ndarray, list, np.ndarray, list]:
    """
    Processes a dataset, ensuring strictly matched pairs from two image columns.

    Args:
        dataset: The Hugging Face dataset to process.
        embedder: The embedder instance for the model.
        column1/column2 (str): The names of the columns containing the paired image data.
        label1/label2 (str): Labels for each data type in the lookup tables.
        batch_size: The number of image pairs to process in each batch.
        max_samples: The maximum number of valid pairs to process.
        use_bulk_processing: If True, preprocess all images into RAM first, then bulk embed

    Returns:
        A tuple containing (embeddings1, lookup1, embeddings2, lookup2).
    """
    logging.info(f"Processing paired data from columns '{column1}' and '{column2}'...")
    
    # Track whether columns need embedding (vs pre-computed)
    needs_embedding1 = column1.endswith('_image') or column1 == 'image' or column1 == 'spectrum'
    needs_embedding2 = column2.endswith('_image') or column2 == 'image' or column2 == 'spectrum'
    
    if use_bulk_processing and (needs_embedding1 or needs_embedding2):
        return _process_paired_dataset_bulk(
            dataset, embedder, column1, label1, column2, label2,
            batch_size, max_samples, dataset_alias, use_lut_normalization,
            needs_embedding1, needs_embedding2
        )
    
    # Original incremental processing
    embeddings1, embeddings2 = [], []
    lookup1, lookup2 = [], []
    
    batch1_images, batch2_images = [], []
    batch_indices = []
    
    processed_pairs = 0
    for idx, sample in enumerate(tqdm(dataset, desc=f"Processing Paired Images")):
        if max_samples and processed_pairs >= max_samples:
            break

        # CRITICAL GATE: Only proceed if BOTH data items in the pair are valid
        try:
            # Handle different data types for each column
            data1 = sample[column1]
            data2 = sample[column2]
            
            # Convert column1 to appropriate format
            if needs_embedding1:
                # Choose normalization method
                if use_lut_normalization and dataset_alias:
                    img1 = image_to_pil_with_lut(sample, column1, dataset_alias)
                else:
                    img1 = flux_to_pil(data1)
                    
                if img1 is None:
                    continue
                batch1_images.append(img1)
            else:
                # Pre-computed embeddings
                if data1 is None:
                    continue
                batch1_images.append(np.array(data1))
            
            # Convert column2 to appropriate format
            if needs_embedding2:
                # Choose normalization method
                if use_lut_normalization and dataset_alias:
                    img2 = image_to_pil_with_lut(sample, column2, dataset_alias)
                else:
                    img2 = flux_to_pil(data2)
                    
                if img2 is None:
                    continue
                batch2_images.append(img2)
            else:
                # Pre-computed embeddings
                if data2 is None:
                    continue
                batch2_images.append(np.array(data2))

            batch_indices.append(idx)
            processed_pairs += 1

            # If batch is full, process it
            if len(batch1_images) >= batch_size:
                if needs_embedding1:
                    emb_batch1 = embedder.embed_batch(batch1_images)
                else:
                    emb_batch1 = np.array(batch1_images)
                    
                if needs_embedding2:
                    emb_batch2 = embedder.embed_batch(batch2_images)
                else:
                    emb_batch2 = np.array(batch2_images)
                    
                    for i in range(len(emb_batch1)):
                        lookup1.append({'dataset_index': batch_indices[i], 'type': label1, 'embedding_index': len(embeddings1)})
                        embeddings1.append(emb_batch1[i].flatten())
                        
                        lookup2.append({'dataset_index': batch_indices[i], 'type': label2, 'embedding_index': len(embeddings2)})
                        embeddings2.append(emb_batch2[i].flatten())

                    batch1_images, batch2_images, batch_indices = [], [], []
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e:
            logging.warning(f"Skipping pair at index {idx} due to error: {e}")

    # Process the final partial batch
    if batch1_images:
        if needs_embedding1:
            emb_batch1 = embedder.embed_batch(batch1_images)
        else:
            emb_batch1 = np.array(batch1_images)
            
        if needs_embedding2:
            emb_batch2 = embedder.embed_batch(batch2_images)
        else:
            emb_batch2 = np.array(batch2_images)
            
        for i in range(len(emb_batch1)):
            lookup1.append({'dataset_index': batch_indices[i], 'type': label1, 'embedding_index': len(embeddings1)})
            embeddings1.append(emb_batch1[i].flatten())
            
            lookup2.append({'dataset_index': batch_indices[i], 'type': label2, 'embedding_index': len(embeddings2)})
            embeddings2.append(emb_batch2[i].flatten())
            
    logging.info(f"Successfully processed {len(embeddings1)} aligned pairs.")
    return np.array(embeddings1), lookup1, np.array(embeddings2), lookup2


def _process_paired_dataset_bulk(
    dataset: Dataset,
    embedder: BaseEmbedder,
    column1: str,
    label1: str,
    column2: str,
    label2: str,
    batch_size: int,
    max_samples: int,
    dataset_alias: str,
    use_lut_normalization: bool,
    needs_embedding1: bool,
    needs_embedding2: bool
) -> tuple[np.ndarray, list, np.ndarray, list]:
    """
    Bulk processing mode: preprocess all images into RAM first, then bulk embed.
    Replicates the RAM-for-speed tradeoff from friend's script architecture.
    """
    logging.info("Using bulk processing mode (RAM-for-speed tradeoff)")
    
    # Phase 1: Bulk preprocessing all images into RAM
    logging.info("Phase 1: Bulk preprocessing images into RAM...")
    
    all_images1, all_images2 = [], []
    all_indices = []
    processed_pairs = 0
    
    for idx, sample in enumerate(tqdm(dataset, desc="Preprocessing Images")):
        if max_samples and processed_pairs >= max_samples:
            break
            
        try:
            data1 = sample[column1]
            data2 = sample[column2]
            
            # Process column1
            if needs_embedding1:
                if use_lut_normalization and dataset_alias:
                    img1 = image_to_pil_with_lut(sample, column1, dataset_alias)
                else:
                    img1 = flux_to_pil(data1)
                if img1 is None:
                    continue
                all_images1.append(img1)
            else:
                if data1 is None:
                    continue
                all_images1.append(np.array(data1))
            
            # Process column2
            if needs_embedding2:
                if use_lut_normalization and dataset_alias:
                    img2 = image_to_pil_with_lut(sample, column2, dataset_alias)
                else:
                    img2 = flux_to_pil(data2)
                if img2 is None:
                    continue
                all_images2.append(img2)
            else:
                if data2 is None:
                    continue
                all_images2.append(np.array(data2))
            
            all_indices.append(idx)
            processed_pairs += 1
            
        except Exception as e:
            logging.warning(f"Skipping pair at index {idx} due to error: {e}")
    
    logging.info(f"Phase 1 complete: {len(all_images1)} pairs preprocessed into RAM")
    
    # Phase 2: Bulk GPU embedding with larger batches
    logging.info("Phase 2: Bulk GPU embedding...")
    
    # Use larger batch sizes for bulk processing (like friend's script: 96-128)
    bulk_batch_size = max(batch_size * 4, 96)  # Scale up batch size for bulk mode
    
    embeddings1, embeddings2 = [], []
    lookup1, lookup2 = [], []
    
    # Process in bulk batches
    for start_idx in tqdm(range(0, len(all_images1), bulk_batch_size), desc="Bulk Embedding"):
        end_idx = min(start_idx + bulk_batch_size, len(all_images1))
        
        batch_imgs1 = all_images1[start_idx:end_idx]
        batch_imgs2 = all_images2[start_idx:end_idx]
        batch_indices = all_indices[start_idx:end_idx]
        
        # Embed batch1
        if needs_embedding1:
            emb_batch1 = embedder.embed_batch(batch_imgs1)
        else:
            emb_batch1 = np.array(batch_imgs1)
        
        # Embed batch2
        if needs_embedding2:
            emb_batch2 = embedder.embed_batch(batch_imgs2)
        else:
            emb_batch2 = np.array(batch_imgs2)
        
        # Store results
        for i in range(len(emb_batch1)):
            lookup1.append({
                'dataset_index': batch_indices[i], 
                'type': label1, 
                'embedding_index': len(embeddings1)
            })
            embeddings1.append(emb_batch1[i].flatten())
            
            lookup2.append({
                'dataset_index': batch_indices[i], 
                'type': label2, 
                'embedding_index': len(embeddings2)
            })
            embeddings2.append(emb_batch2[i].flatten())
        
        # Clean GPU cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logging.info(f"Bulk processing complete: {len(embeddings1)} aligned pairs embedded")
    return np.array(embeddings1), lookup1, np.array(embeddings2), lookup2