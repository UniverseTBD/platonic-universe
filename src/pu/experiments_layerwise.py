"""
Layer-by-layer comparison experiments for testing the Platonic Representation Hypothesis.

This module provides functionality to compare representations across ALL layers
of different models, enabling analysis of:
1. Maximum correlation layer pairs between models
2. Convergence patterns with model size
3. Consistency across different metrics and hyperparameters
4. The "mix-up hypothesis" (do early layers align with early layers?)
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pu.models import get_adapter
from pu.pu_datasets import get_dataset_adapter
from pu.metrics import mknn, cka, compare, METRICS_REGISTRY


@dataclass
class LayerwiseResult:
    """Results from a layer-by-layer comparison between two models."""
    
    # Model info
    model_a_alias: str
    model_a_size: str
    model_a_num_layers: int
    model_b_alias: str
    model_b_size: str
    model_b_num_layers: int
    
    # Dataset info
    mode: str
    n_samples: int
    
    # Alignment matrices: metric_name -> (num_layers_a, num_layers_b) array
    alignment_matrices: Dict[str, List[List[float]]] = field(default_factory=dict)
    
    # Optimal pairs for each metric: metric_name -> (layer_a, layer_b, score)
    optimal_pairs: Dict[str, Tuple[int, int, float]] = field(default_factory=dict)
    
    # Diagonal scores (same layer index): metric_name -> list of scores
    diagonal_scores: Dict[str, List[float]] = field(default_factory=dict)
    
    # Maximum score per metric
    max_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def _collect_layer_embeddings(
    adapter,
    dataloader,
    mode: str,
    max_samples: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """
    Collect embeddings from all layers for all samples.
    
    Returns:
        Dict mapping layer index to (n_samples, hidden_dim) array
    """
    if not adapter.supports_layerwise():
        raise ValueError(f"Adapter {adapter.alias} does not support layer-wise extraction")
    
    num_layers = adapter.get_num_layers()
    layer_embeddings = {i: [] for i in range(num_layers)}
    n_collected = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting {mode} layers"):
            # Get all layer embeddings for this batch
            batch_layer_embs = adapter.embed_all_layers_for_mode(batch, mode)
            
            for layer_idx, emb in batch_layer_embs.items():
                # emb shape: (batch_size, hidden_dim) or (1, hidden_dim)
                if isinstance(emb, torch.Tensor):
                    emb = emb.cpu().numpy()
                layer_embeddings[layer_idx].append(emb)
            
            n_collected += emb.shape[0] if emb.ndim > 1 else 1
            
            if max_samples is not None and n_collected >= max_samples:
                break
    
    # Concatenate all batches
    result = {}
    for layer_idx, emb_list in layer_embeddings.items():
        if emb_list:
            stacked = np.vstack(emb_list)
            if max_samples is not None:
                stacked = stacked[:max_samples]
            result[layer_idx] = stacked
    
    return result


def compute_alignment_matrix(
    embeddings_a: Dict[int, np.ndarray],
    embeddings_b: Dict[int, np.ndarray],
    metric_name: str,
    **metric_kwargs
) -> np.ndarray:
    """
    Compute alignment scores between all layer pairs.
    
    Args:
        embeddings_a: Layer embeddings from model A
        embeddings_b: Layer embeddings from model B
        metric_name: Name of the metric to use
        **metric_kwargs: Additional arguments for the metric
    
    Returns:
        (num_layers_a, num_layers_b) alignment matrix
    """
    num_layers_a = len(embeddings_a)
    num_layers_b = len(embeddings_b)
    
    alignment_matrix = np.zeros((num_layers_a, num_layers_b))
    
    metric_fn = METRICS_REGISTRY.get(metric_name)
    if metric_fn is None:
        raise ValueError(f"Unknown metric: {metric_name}")
    
    for i in range(num_layers_a):
        for j in range(num_layers_b):
            try:
                score = metric_fn(embeddings_a[i], embeddings_b[j], **metric_kwargs)
                alignment_matrix[i, j] = score
            except Exception as e:
                # Some comparisons may fail (dimension mismatch, etc.)
                alignment_matrix[i, j] = np.nan
    
    return alignment_matrix


def find_optimal_pair(alignment_matrix: np.ndarray) -> Tuple[int, int, float]:
    """Find the layer pair with maximum alignment score."""
    # Handle NaN values
    masked = np.where(np.isnan(alignment_matrix), -np.inf, alignment_matrix)
    max_idx = np.unravel_index(np.argmax(masked), alignment_matrix.shape)
    max_score = alignment_matrix[max_idx]
    return int(max_idx[0]), int(max_idx[1]), float(max_score)


def get_diagonal_scores(alignment_matrix: np.ndarray) -> List[float]:
    """Get scores along the diagonal (same layer index comparisons)."""
    min_dim = min(alignment_matrix.shape)
    return [float(alignment_matrix[i, i]) for i in range(min_dim)]


def run_layerwise_comparison(
    model_a_alias: str,
    model_a_size: str,
    model_b_alias: str,
    model_b_size: str,
    mode: str = "jwst",
    metrics: List[str] = None,
    mknn_k_values: List[int] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    output_dir: str = "data/layerwise",
) -> LayerwiseResult:
    """
    Run layer-by-layer comparison between two models.
    
    This function:
    1. Loads both models
    2. Extracts embeddings from ALL layers
    3. Computes alignment matrices for each metric
    4. Finds optimal layer pairs and patterns
    
    Args:
        model_a_alias: Alias of first model (e.g., 'smolvlm')
        model_a_size: Size of first model (e.g., '256M')
        model_b_alias: Alias of second model
        model_b_size: Size of second model
        mode: Dataset mode (e.g., 'jwst', 'legacysurvey')
        metrics: List of metrics to compute (default: ['mknn', 'cka'])
        mknn_k_values: List of k values to test for MKNN (default: [5, 10, 20])
        batch_size: Batch size for data loading
        num_workers: Number of data loader workers
        max_samples: Maximum samples to process (None for all)
        output_dir: Directory to save results
    
    Returns:
        LayerwiseResult with alignment matrices and analysis
    """
    if metrics is None:
        metrics = ["mknn", "cka"]
    
    if mknn_k_values is None:
        mknn_k_values = [5, 10, 20]
    
    # Model configurations
    model_map = {
        "smolvlm": {
            "256M": "HuggingFaceTB/SmolVLM-256M-Instruct",
            "500M": "HuggingFaceTB/SmolVLM-500M-Instruct",
            "2.2B": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        },
    }
    
    # Get model names
    if model_a_alias not in model_map:
        raise ValueError(f"Model '{model_a_alias}' not supported for layerwise comparison")
    if model_a_size not in model_map[model_a_alias]:
        raise ValueError(f"Size '{model_a_size}' not available for {model_a_alias}")
    
    model_a_name = model_map[model_a_alias][model_a_size]
    model_b_name = model_map[model_b_alias][model_b_size]
    
    # Dataset configuration
    hf_ds = f"Smith42/{mode}_hsc_crossmatched"
    modes = ["hsc", mode]
    
    # Filter function for JWST
    def filterfun(idx):
        if "jwst" != mode:
            return True
        im = idx["jwst_image"]["flux"][3]
        v0, v1 = np.nanpercentile(im, 5), np.nanpercentile(im, 99)
        return v0 - v1 != 0
    
    print(f"\n{'='*60}")
    print(f"LAYER-BY-LAYER COMPARISON")
    print(f"{'='*60}")
    print(f"Model A: {model_a_alias}-{model_a_size}")
    print(f"Model B: {model_b_alias}-{model_b_size}")
    print(f"Mode: {mode}")
    print(f"Metrics: {metrics}")
    print(f"MKNN k values: {mknn_k_values}")
    print(f"{'='*60}\n")
    
    # Load Model A
    print(f"[1/5] Loading Model A: {model_a_alias}-{model_a_size}")
    adapter_a_cls = get_adapter(model_a_alias)
    adapter_a = adapter_a_cls(model_a_name, model_a_size, alias=model_a_alias)
    adapter_a.load()
    
    if not adapter_a.supports_layerwise():
        raise ValueError(f"Model {model_a_alias} does not support layer-wise extraction")
    
    num_layers_a = adapter_a.get_num_layers()
    print(f"  -> {num_layers_a} layers")
    
    # Load Model B
    print(f"\n[2/5] Loading Model B: {model_b_alias}-{model_b_size}")
    adapter_b_cls = get_adapter(model_b_alias)
    adapter_b = adapter_b_cls(model_b_name, model_b_size, alias=model_b_alias)
    adapter_b.load()
    
    if not adapter_b.supports_layerwise():
        raise ValueError(f"Model {model_b_alias} does not support layer-wise extraction")
    
    num_layers_b = adapter_b.get_num_layers()
    print(f"  -> {num_layers_b} layers")
    
    # Prepare dataset (use model A's preprocessor for consistency)
    print(f"\n[3/5] Loading dataset: {hf_ds}")
    processor_a = adapter_a.get_preprocessor(modes)
    
    dataset_adapter_cls = get_dataset_adapter(mode)
    dataset_adapter = dataset_adapter_cls(hf_ds, mode)
    dataset_adapter.load()
    ds = dataset_adapter.prepare(processor_a, modes, filterfun)
    
    if max_samples is not None:
        ds = ds.take(max_samples)
        print(f"  -> Limited to {max_samples} samples")
    
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    
    # Extract embeddings from Model A
    print(f"\n[4/5] Extracting embeddings from Model A")
    embeddings_a = _collect_layer_embeddings(adapter_a, dl, modes[0], max_samples)
    n_samples = len(embeddings_a[0])
    print(f"  -> Collected {n_samples} samples, {len(embeddings_a)} layers")
    
    # Need to reload dataset for Model B with its preprocessor
    print(f"\n[4b/5] Extracting embeddings from Model B")
    processor_b = adapter_b.get_preprocessor(modes)
    dataset_adapter_b = dataset_adapter_cls(hf_ds, mode)
    dataset_adapter_b.load()
    ds_b = dataset_adapter_b.prepare(processor_b, modes, filterfun)
    
    if max_samples is not None:
        ds_b = ds_b.take(max_samples)
    
    dl_b = DataLoader(ds_b, batch_size=batch_size, num_workers=num_workers)
    embeddings_b = _collect_layer_embeddings(adapter_b, dl_b, modes[0], max_samples)
    print(f"  -> Collected {len(embeddings_b[0])} samples, {len(embeddings_b)} layers")
    
    # Compute alignment matrices
    print(f"\n[5/5] Computing alignment matrices")
    
    result = LayerwiseResult(
        model_a_alias=model_a_alias,
        model_a_size=model_a_size,
        model_a_num_layers=num_layers_a,
        model_b_alias=model_b_alias,
        model_b_size=model_b_size,
        model_b_num_layers=num_layers_b,
        mode=mode,
        n_samples=n_samples,
    )
    
    for metric_name in metrics:
        if metric_name == "mknn":
            # Test multiple k values
            for k in mknn_k_values:
                metric_key = f"mknn_k{k}"
                print(f"  Computing {metric_key}...")
                
                matrix = compute_alignment_matrix(
                    embeddings_a, embeddings_b, "mknn", k=k
                )
                
                result.alignment_matrices[metric_key] = matrix.tolist()
                result.optimal_pairs[metric_key] = find_optimal_pair(matrix)
                result.diagonal_scores[metric_key] = get_diagonal_scores(matrix)
                result.max_scores[metric_key] = float(np.nanmax(matrix))
        else:
            print(f"  Computing {metric_name}...")
            
            matrix = compute_alignment_matrix(
                embeddings_a, embeddings_b, metric_name
            )
            
            result.alignment_matrices[metric_name] = matrix.tolist()
            result.optimal_pairs[metric_name] = find_optimal_pair(matrix)
            result.diagonal_scores[metric_name] = get_diagonal_scores(matrix)
            result.max_scores[metric_name] = float(np.nanmax(matrix))
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, 
        f"{mode}_{model_a_alias}_{model_a_size}_vs_{model_b_alias}_{model_b_size}.json"
    )
    
    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for metric_key, (layer_a, layer_b, score) in result.optimal_pairs.items():
        print(f"{metric_key}:")
        print(f"  Optimal pair: Layer {layer_a} (A) <-> Layer {layer_b} (B)")
        print(f"  Max score: {score:.4f}")
    
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*60}\n")
    
    return result


def run_size_convergence_study(
    model_alias: str = "smolvlm",
    sizes: List[str] = None,
    mode: str = "jwst",
    metrics: List[str] = None,
    mknn_k_values: List[int] = None,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    output_dir: str = "data/layerwise",
) -> Dict[str, Any]:
    """
    Study convergence of representations with model size.
    
    Compares adjacent size pairs (small->medium, medium->large) to test
    if alignment increases with model scale.
    
    Args:
        model_alias: Model family (e.g., 'smolvlm')
        sizes: List of sizes in order (default: ['256M', '500M', '2.2B'])
        mode: Dataset mode
        metrics: Metrics to compute
        mknn_k_values: k values for MKNN
        batch_size: Batch size
        max_samples: Max samples to process
        output_dir: Output directory
    
    Returns:
        Dictionary with convergence analysis results
    """
    if sizes is None:
        sizes = ["256M", "500M", "2.2B"]
    
    if metrics is None:
        metrics = ["mknn", "cka"]
    
    if mknn_k_values is None:
        mknn_k_values = [5, 10, 20]
    
    print(f"\n{'#'*60}")
    print(f"SIZE CONVERGENCE STUDY")
    print(f"{'#'*60}")
    print(f"Model: {model_alias}")
    print(f"Sizes: {' -> '.join(sizes)}")
    print(f"{'#'*60}\n")
    
    # Compare adjacent pairs
    comparisons = []
    for i in range(len(sizes) - 1):
        size_a, size_b = sizes[i], sizes[i + 1]
        print(f"\n>>> Comparing {size_a} vs {size_b}")
        
        result = run_layerwise_comparison(
            model_a_alias=model_alias,
            model_a_size=size_a,
            model_b_alias=model_alias,
            model_b_size=size_b,
            mode=mode,
            metrics=metrics,
            mknn_k_values=mknn_k_values,
            batch_size=batch_size,
            max_samples=max_samples,
            output_dir=output_dir,
        )
        
        comparisons.append({
            "size_a": size_a,
            "size_b": size_b,
            "result": result.to_dict(),
        })
    
    # Analyze convergence
    convergence_analysis = {
        "model": model_alias,
        "sizes": sizes,
        "mode": mode,
        "comparisons": comparisons,
        "convergence_pattern": {},
    }
    
    # For each metric, check if max score increases with size
    all_metric_keys = set()
    for comp in comparisons:
        all_metric_keys.update(comp["result"]["max_scores"].keys())
    
    for metric_key in all_metric_keys:
        scores = []
        for comp in comparisons:
            score = comp["result"]["max_scores"].get(metric_key)
            scores.append(score)
        
        # Check if monotonically increasing
        is_increasing = all(s1 <= s2 for s1, s2 in zip(scores[:-1], scores[1:]))
        
        convergence_analysis["convergence_pattern"][metric_key] = {
            "scores": scores,
            "is_increasing": is_increasing,
            "comparison_labels": [f"{sizes[i]}->{sizes[i+1]}" for i in range(len(sizes)-1)],
        }
    
    # Check consistency of optimal layer pairs
    consistency_analysis = {}
    for metric_key in all_metric_keys:
        pairs = []
        for comp in comparisons:
            pair = comp["result"]["optimal_pairs"].get(metric_key)
            if pair:
                pairs.append((pair[0], pair[1]))  # (layer_a, layer_b)
        
        # Check if same layers are selected across comparisons
        unique_pairs = set(pairs)
        consistency_analysis[metric_key] = {
            "pairs": pairs,
            "unique_count": len(unique_pairs),
            "is_consistent": len(unique_pairs) == 1,
        }
    
    convergence_analysis["consistency_analysis"] = consistency_analysis
    
    # Save convergence study results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"convergence_study_{model_alias}_{mode}_{'_'.join(sizes)}.json"
    )
    
    with open(output_file, "w") as f:
        json.dump(convergence_analysis, f, indent=2, default=str)
    
    # Print summary
    print(f"\n{'#'*60}")
    print("CONVERGENCE STUDY SUMMARY")
    print(f"{'#'*60}")
    
    print("\n📈 Convergence Pattern (does max alignment increase with size?):")
    for metric_key, pattern in convergence_analysis["convergence_pattern"].items():
        scores_str = " -> ".join([f"{s:.4f}" for s in pattern["scores"]])
        status = "✅ YES" if pattern["is_increasing"] else "❌ NO"
        print(f"  {metric_key}: {scores_str} [{status}]")
    
    print("\n🎯 Optimal Layer Pair Consistency:")
    for metric_key, consistency in consistency_analysis.items():
        pairs_str = ", ".join([f"({p[0]},{p[1]})" for p in consistency["pairs"]])
        status = "✅ CONSISTENT" if consistency["is_consistent"] else f"⚠️  {consistency['unique_count']} different pairs"
        print(f"  {metric_key}: {pairs_str} [{status}]")
    
    print(f"\nResults saved to: {output_file}")
    print(f"{'#'*60}\n")
    
    return convergence_analysis


def run_metric_consistency_study(
    model_a_alias: str,
    model_a_size: str,
    model_b_alias: str,
    model_b_size: str,
    mode: str = "jwst",
    metrics: List[str] = None,
    mknn_k_values: List[int] = None,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    output_dir: str = "data/layerwise",
) -> Dict[str, Any]:
    """
    Study consistency of optimal layer pairs across different metrics.
    
    Tests whether the same layer pairs are selected as optimal regardless
    of which metric is used.
    
    Args:
        model_a_alias, model_a_size: First model
        model_b_alias, model_b_size: Second model
        mode: Dataset mode
        metrics: Metrics to compare (default: all main metrics)
        mknn_k_values: k values for MKNN ablation
        batch_size: Batch size
        max_samples: Max samples
        output_dir: Output directory
    
    Returns:
        Consistency analysis results
    """
    if metrics is None:
        # Use a diverse set of metrics
        metrics = ["mknn", "cka", "procrustes", "cosine_similarity"]
    
    if mknn_k_values is None:
        mknn_k_values = [5, 10, 20, 50]
    
    # Run comparison with all metrics
    result = run_layerwise_comparison(
        model_a_alias=model_a_alias,
        model_a_size=model_a_size,
        model_b_alias=model_b_alias,
        model_b_size=model_b_size,
        mode=mode,
        metrics=metrics,
        mknn_k_values=mknn_k_values,
        batch_size=batch_size,
        max_samples=max_samples,
        output_dir=output_dir,
    )
    
    # Analyze consistency
    consistency = {
        "model_a": f"{model_a_alias}-{model_a_size}",
        "model_b": f"{model_b_alias}-{model_b_size}",
        "mode": mode,
        "optimal_pairs": result.optimal_pairs,
        "max_scores": result.max_scores,
    }
    
    # Group by optimal pair
    pair_to_metrics = {}
    for metric_key, (layer_a, layer_b, score) in result.optimal_pairs.items():
        pair = (layer_a, layer_b)
        if pair not in pair_to_metrics:
            pair_to_metrics[pair] = []
        pair_to_metrics[pair].append((metric_key, score))
    
    consistency["pair_to_metrics"] = {
        str(k): v for k, v in pair_to_metrics.items()
    }
    
    # Check MKNN k-value consistency
    mknn_pairs = {
        k: v for k, v in result.optimal_pairs.items() 
        if k.startswith("mknn_k")
    }
    mknn_layers = [(v[0], v[1]) for v in mknn_pairs.values()]
    mknn_unique = set(mknn_layers)
    
    consistency["mknn_k_consistency"] = {
        "pairs": list(mknn_pairs.items()),
        "unique_pairs": len(mknn_unique),
        "is_consistent": len(mknn_unique) == 1,
    }
    
    # Check cross-metric consistency
    all_pairs = [(v[0], v[1]) for v in result.optimal_pairs.values()]
    unique_pairs = set(all_pairs)
    
    consistency["cross_metric_consistency"] = {
        "unique_pairs": len(unique_pairs),
        "is_consistent": len(unique_pairs) == 1,
        "most_common_pair": max(set(all_pairs), key=all_pairs.count) if all_pairs else None,
    }
    
    # Save
    output_file = os.path.join(
        output_dir,
        f"consistency_{model_a_alias}_{model_a_size}_vs_{model_b_alias}_{model_b_size}_{mode}.json"
    )
    
    with open(output_file, "w") as f:
        json.dump(consistency, f, indent=2, default=str)
    
    # Print summary
    print(f"\n{'='*60}")
    print("METRIC CONSISTENCY ANALYSIS")
    print(f"{'='*60}")
    
    print("\n🎯 Optimal pairs by metric:")
    for metric_key, (layer_a, layer_b, score) in result.optimal_pairs.items():
        print(f"  {metric_key}: Layer {layer_a} <-> Layer {layer_b} (score: {score:.4f})")
    
    print(f"\n📊 MKNN k-value consistency:")
    if consistency["mknn_k_consistency"]["is_consistent"]:
        print(f"  ✅ CONSISTENT across k values")
    else:
        print(f"  ⚠️  {consistency['mknn_k_consistency']['unique_pairs']} different pairs")
    
    print(f"\n📊 Cross-metric consistency:")
    if consistency["cross_metric_consistency"]["is_consistent"]:
        print(f"  ✅ ALL metrics agree on optimal pair")
    else:
        print(f"  ⚠️  {consistency['cross_metric_consistency']['unique_pairs']} different pairs")
        print(f"  Most common: {consistency['cross_metric_consistency']['most_common_pair']}")
    
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*60}\n")
    
    return consistency
