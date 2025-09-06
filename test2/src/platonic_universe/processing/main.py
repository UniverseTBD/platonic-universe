import torch
import logging
import numpy as np
from tqdm.auto import tqdm
from datasets import Dataset
from ..models.embedders import BaseEmbedder
from ..preprocessing.preparation import flux_to_pil, image_to_pil_with_lut

def process_paired_dataset(
    dataset,  # Can be Dataset or tuple of datasets for streaming DESI-HSC
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
    # Handle special case for streaming datasets that return tuples
    if isinstance(dataset, tuple) and len(dataset) == 2:
        logging.info(f"🔍 TUPLE DETECTED: dataset_alias='{dataset_alias}', columns=({column1}, {column2})")
        
        # Check if this is HSC-SDSS first (more specific)
        if dataset_alias == "hsc-sdss":
            logging.info("✅ Processing HSC-SDSS streaming tuple...")
            return _process_streaming_hsc_sdss(
                dataset[0], dataset[1], embedder, column1, label1, column2, label2, 
                batch_size, max_samples, dataset_alias, use_lut_normalization
            )
        # Then check DESI-HSC variants
        elif dataset_alias.startswith("desi-hsc"):
            logging.info("✅ Processing DESI-HSC streaming tuple...")
            return _process_streaming_desi_hsc(
                dataset[0], dataset[1], embedder, column1, label1, column2, label2, 
                batch_size, max_samples, dataset_alias, use_lut_normalization
            )
        else:
            logging.warning(f"Unknown tuple dataset format for '{dataset_alias}', attempting standard processing...")
    
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
                # Handle different formats of pre-computed embeddings
                if isinstance(data1, (list, tuple)):
                    emb_array = np.array(data1, dtype=np.float32)
                else:
                    emb_array = np.array(data1, dtype=np.float32)
                batch1_images.append(emb_array)
            
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
                # Handle different formats of pre-computed embeddings
                if isinstance(data2, (list, tuple)):
                    emb_array = np.array(data2, dtype=np.float32)
                else:
                    emb_array = np.array(data2, dtype=np.float32)
                batch2_images.append(emb_array)

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
    
    # Detect AstroPT models for special preprocessing
    is_astropt = hasattr(embedder.model, 'generate_embeddings') and hasattr(embedder.model, 'modality_registry')
    if is_astropt:
        return _process_astropt_dataset_bulk(
            dataset, embedder, column1, label1, column2, label2,
            batch_size, max_samples, dataset_alias, use_lut_normalization,
            needs_embedding1, needs_embedding2
        )
    
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
            processed_data1 = None
            if needs_embedding1:
                if use_lut_normalization and dataset_alias:
                    img1 = image_to_pil_with_lut(sample, column1, dataset_alias)
                else:
                    img1 = flux_to_pil(data1)
                if img1 is None:
                    continue  # Skip this pair entirely
                processed_data1 = img1
            else:
                if data1 is None:
                    continue  # Skip this pair entirely
                # Handle different formats of pre-computed embeddings
                if isinstance(data1, (list, tuple)):
                    emb_array = np.array(data1, dtype=np.float32)
                else:
                    emb_array = np.array(data1, dtype=np.float32)
                processed_data1 = emb_array
            
            # Process column2
            processed_data2 = None
            if needs_embedding2:
                if use_lut_normalization and dataset_alias:
                    img2 = image_to_pil_with_lut(sample, column2, dataset_alias)
                else:
                    img2 = flux_to_pil(data2)
                if img2 is None:
                    continue  # Skip this pair entirely
                processed_data2 = img2
            else:
                if data2 is None:
                    continue  # Skip this pair entirely
                # Handle different formats of pre-computed embeddings
                if isinstance(data2, (list, tuple)):
                    emb_array = np.array(data2, dtype=np.float32)
                else:
                    emb_array = np.array(data2, dtype=np.float32)
                processed_data2 = emb_array
            
            # Only add to lists if both data items are valid
            if processed_data1 is not None and processed_data2 is not None:
                all_images1.append(processed_data1)
                all_images2.append(processed_data2)
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
        
        # Store results - ensure indices align properly
        actual_batch_size = min(len(emb_batch1), len(emb_batch2), len(batch_indices))
        for i in range(actual_batch_size):
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


def _process_streaming_hsc_sdss(
    hsc_sdss_dataset,
    sdss_embeddings_dataset, 
    embedder: BaseEmbedder,
    column1: str,
    label1: str,
    column2: str,
    label2: str,
    batch_size: int,
    max_samples: int,
    dataset_alias: str,
    use_lut_normalization: bool
) -> tuple[np.ndarray, list, np.ndarray, list]:
    """
    Process streaming HSC-SDSS datasets separately and zip them together.
    """
    logging.info("Processing streaming HSC-SDSS datasets...")
    
    embeddings1, embeddings2 = [], []
    lookup1, lookup2 = [], []
    
    batch1_images, batch2_data = [], []
    batch_indices = []
    
    processed_pairs = 0
    
    # Zip the two streaming datasets together
    dataset_zip = zip(hsc_sdss_dataset, sdss_embeddings_dataset)
    for idx, (hsc_sample, sdss_sample) in enumerate(tqdm(dataset_zip, desc="Processing HSC-SDSS pairs")):
        if max_samples and processed_pairs >= max_samples:
            break
            
        try:
            # Process HSC image (column1 = "hsc_image" but actual key is "image")
            if column1 == "hsc_image":
                if "image" in hsc_sample:
                    data1 = hsc_sample["image"]
                else:
                    logging.warning(f"'image' key not found in HSC sample. Available keys: {list(hsc_sample.keys())}")
                    continue
                if use_lut_normalization and dataset_alias:
                    img1 = image_to_pil_with_lut(hsc_sample, column1, dataset_alias)
                else:
                    img1 = flux_to_pil(data1)
                if img1 is None:
                    continue
                batch1_images.append(img1)
            
            # Process SDSS embedding (column2 = "embedding") 
            if column2 == "embedding":
                data2 = sdss_sample["embedding"]
                if data2 is None:
                    continue
                # Handle different formats of pre-computed embeddings
                if isinstance(data2, (list, tuple)):
                    emb_array = np.array(data2, dtype=np.float32)
                else:
                    emb_array = np.array(data2, dtype=np.float32)
                batch2_data.append(emb_array)
                
            batch_indices.append(idx)
            processed_pairs += 1
            
            # Process batch when full
            if len(batch1_images) >= batch_size:
                # Embed HSC images
                if batch1_images:
                    batch_embeddings1 = embedder.embed_batch(batch1_images)
                    embeddings1.extend(batch_embeddings1)
                    
                    # Create lookup entries for HSC
                    for i, img_idx in enumerate(batch_indices[:len(batch1_images)]):
                        lookup1.append({
                            'index': img_idx, 
                            'source': label1,
                            'original_data': f'{label1}_image_{img_idx}'
                        })
                
                # Process SDSS embeddings (already computed)
                if batch2_data:
                    embeddings2.extend(batch2_data)
                    
                    # Create lookup entries for SDSS
                    for i, emb_idx in enumerate(batch_indices[:len(batch2_data)]):
                        lookup2.append({
                            'index': emb_idx,
                            'source': label2, 
                            'original_data': f'{label2}_embedding_{emb_idx}'
                        })
                
                # Clear batches
                batch1_images, batch2_data = [], []
                batch_indices = []
                
        except Exception as e:
            logging.warning(f"Failed to process HSC-SDSS pair {idx}: {e}")
            continue
    
    # Process final batch
    if batch1_images and batch2_data:
        # Embed remaining HSC images
        batch_embeddings1 = embedder.embed_batch(batch1_images)
        embeddings1.extend(batch_embeddings1)
        
        for i, img_idx in enumerate(batch_indices[:len(batch1_images)]):
            lookup1.append({
                'index': img_idx,
                'source': label1,
                'original_data': f'{label1}_image_{img_idx}'
            })
        
        # Process remaining SDSS embeddings
        embeddings2.extend(batch2_data)
        
        for i, emb_idx in enumerate(batch_indices[:len(batch2_data)]):
            lookup2.append({
                'index': emb_idx,
                'source': label2,
                'original_data': f'{label2}_embedding_{emb_idx}'
            })
    
    # Convert to numpy arrays
    final_embeddings1 = np.array(embeddings1, dtype=np.float32) if embeddings1 else np.array([])
    final_embeddings2 = np.array(embeddings2, dtype=np.float32) if embeddings2 else np.array([])
    
    logging.info(f"Successfully processed {len(final_embeddings1)} aligned pairs from streaming HSC-SDSS.")
    return final_embeddings1, lookup1, final_embeddings2, lookup2


def _process_streaming_desi_hsc(
    desi_hsc_dataset,
    desi_embeddings_dataset, 
    embedder: BaseEmbedder,
    column1: str,
    label1: str,
    column2: str,
    label2: str,
    batch_size: int,
    max_samples: int,
    dataset_alias: str,
    use_lut_normalization: bool
) -> tuple[np.ndarray, list, np.ndarray, list]:
    """
    Process streaming DESI-HSC datasets separately and zip them together.
    """
    logging.info("Processing streaming DESI-HSC datasets...")
    
    embeddings1, embeddings2 = [], []
    lookup1, lookup2 = [], []
    
    batch1_images, batch2_data = [], []
    batch_indices = []
    
    processed_pairs = 0
    
    # Zip the two streaming datasets together
    dataset_zip = zip(desi_hsc_dataset, desi_embeddings_dataset)
    for idx, (desi_sample, emb_sample) in enumerate(tqdm(dataset_zip, desc="Processing DESI-HSC pairs")):
        if max_samples and processed_pairs >= max_samples:
            break
            
        try:
            # Process HSC image (column1 = "image")
            if column1 == "image":
                data1 = desi_sample["image"]
                if use_lut_normalization and dataset_alias:
                    img1 = image_to_pil_with_lut(desi_sample, column1, dataset_alias)
                else:
                    img1 = flux_to_pil(data1)
                if img1 is None:
                    continue
                batch1_images.append(img1)
            
            # Process DESI embedding (column2 = "embedding") 
            if column2 == "embedding":
                data2 = emb_sample["embedding"]
                if data2 is None:
                    continue
                # Handle different formats of pre-computed embeddings
                if isinstance(data2, (list, tuple)):
                    emb_array = np.array(data2, dtype=np.float32)
                else:
                    emb_array = np.array(data2, dtype=np.float32)
                batch2_data.append(emb_array)
                
            batch_indices.append(idx)
            processed_pairs += 1
            
            # Process batch when full
            if len(batch1_images) >= batch_size:
                # Embed HSC images
                emb_batch1 = embedder.embed_batch(batch1_images)
                # DESI embeddings are already computed
                emb_batch2 = np.array(batch2_data)
                
                for i in range(len(emb_batch1)):
                    lookup1.append({'dataset_index': batch_indices[i], 'type': label1, 'embedding_index': len(embeddings1)})
                    embeddings1.append(emb_batch1[i].flatten())
                    
                    lookup2.append({'dataset_index': batch_indices[i], 'type': label2, 'embedding_index': len(embeddings2)})
                    embeddings2.append(emb_batch2[i].flatten())
                
                batch1_images, batch2_data, batch_indices = [], [], []
                if torch.cuda.is_available(): 
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            logging.warning(f"Skipping pair at index {idx} due to error: {e}")
    
    # Process final batch
    if batch1_images:
        emb_batch1 = embedder.embed_batch(batch1_images)
        emb_batch2 = np.array(batch2_data)
        
        for i in range(len(emb_batch1)):
            lookup1.append({'dataset_index': batch_indices[i], 'type': label1, 'embedding_index': len(embeddings1)})
            embeddings1.append(emb_batch1[i].flatten())
            
            lookup2.append({'dataset_index': batch_indices[i], 'type': label2, 'embedding_index': len(embeddings2)})
            embeddings2.append(emb_batch2[i].flatten())
    
    logging.info(f"Successfully processed {len(embeddings1)} aligned pairs from streaming DESI-HSC.")
    return np.array(embeddings1), lookup1, np.array(embeddings2), lookup2


def _process_astropt_dataset_bulk(
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
    AstroPT-specific processing: applies PreprocessAstropt directly to raw flux data.
    This matches the workflow from the original script.
    """
    logging.info("Using AstroPT-specific preprocessing pipeline")
    
    # Import AstroPT preprocessor
    from ..preprocessing.astropt import PreprocessAstropt
    
    # Determine modes from column names
    modes = []
    if column1.endswith('_image') or column1 == 'image':
        mode1 = column1.replace('_image', '') if column1.endswith('_image') else 'hsc'
        modes.append(mode1)
    if column2.endswith('_image') or column2 == 'image':
        mode2 = column2.replace('_image', '') if column2.endswith('_image') else column2.replace('_image', '')
        modes.append(mode2)
    
    # Initialize AstroPT preprocessor
    preprocessor = PreprocessAstropt(embedder.model.modality_registry, modes, resize=False, use_lut_normalization=use_lut_normalization)
    
    # Process the dataset using the same pattern as original script
    logging.info("Phase 1: Applying AstroPT preprocessing to astronomical data...")
    
    all_samples = []
    processed_pairs = 0
    
    # Collect and preprocess samples
    for idx, sample in enumerate(tqdm(dataset, desc="AstroPT Preprocessing")):
        if max_samples and processed_pairs >= max_samples:
            break
            
        try:
            # Apply AstroPT preprocessing to the sample
            processed_sample = preprocessor(sample)
            all_samples.append((idx, processed_sample))
            processed_pairs += 1
        except Exception as e:
            logging.debug(f"Skipped sample {idx}: {e}")
            continue
    
    logging.info(f"Phase 1 complete: {len(all_samples)} samples preprocessed")
    
    # Phase 2: Bulk embedding using processed data
    logging.info("Phase 2: Bulk GPU embedding with AstroPT...")
    
    embeddings1, embeddings2 = [], []
    lookup1, lookup2 = [], []
    
    # Process in batches
    for i in tqdm(range(0, len(all_samples), batch_size), desc="Bulk Embedding"):
        batch_samples = all_samples[i:i + batch_size]
        batch_data1, batch_data2 = [], []
        batch_indices = []
        
        for orig_idx, processed_sample in batch_samples:
            batch_indices.append(orig_idx)
            
            # Prepare data for each mode
            if needs_embedding1 and modes:
                mode1_key = f"{modes[0]}_images"
                mode1_pos_key = f"{modes[0]}_positions"
                if mode1_key in processed_sample and mode1_pos_key in processed_sample:
                    batch_data1.append({
                        'images': processed_sample[mode1_key],
                        'images_positions': processed_sample[mode1_pos_key]
                    })
            
            if needs_embedding2 and len(modes) > 1:
                mode2_key = f"{modes[1]}_images"
                mode2_pos_key = f"{modes[1]}_positions"
                if mode2_key in processed_sample and mode2_pos_key in processed_sample:
                    batch_data2.append({
                        'images': processed_sample[mode2_key],
                        'images_positions': processed_sample[mode2_pos_key]
                    })
        
        # Generate embeddings
        if batch_data1 and needs_embedding1:
            emb_batch1 = embedder.embed_batch(batch_data1)
            for j, emb in enumerate(emb_batch1):
                lookup1.append({'dataset_index': batch_indices[j], 'type': label1, 'embedding_index': len(embeddings1)})
                embeddings1.append(emb.flatten())
        
        if batch_data2 and needs_embedding2:
            emb_batch2 = embedder.embed_batch(batch_data2)
            for j, emb in enumerate(emb_batch2):
                lookup2.append({'dataset_index': batch_indices[j], 'type': label2, 'embedding_index': len(embeddings2)})
                embeddings2.append(emb.flatten())
    
    logging.info(f"AstroPT processing complete: {len(embeddings1)} aligned pairs embedded")
    return np.array(embeddings1), lookup1, np.array(embeddings2), lookup2