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
import tempfile
import shutil
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from pu.models import get_adapter
from pu.pu_datasets import get_dataset_adapter
from pu.metrics import mknn, cka, compare, METRICS_REGISTRY


# ============================================================
# Plotting aesthetics
# ============================================================
COLOR = "black"
plt.rcParams.update(
    {
        "figure.dpi": 100,
        "figure.figsize": (14, 9),
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.grid": True,
        "legend.fontsize": 14,
        "legend.title_fontsize": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "ytick.labelsize": 12,
        "xtick.labelsize": 12,
        "text.color": COLOR,
        "axes.labelcolor": COLOR,
        "xtick.color": COLOR,
        "ytick.color": COLOR,
        "grid.color": COLOR,
    }
)
plt.rcParams["text.latex.preamble"] = r"\usepackage[version=3]{mhchem}"


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


def _collect_layer_embeddings_to_disk(
    adapter,
    dataloader,
    mode: str,
    temp_dir: str,
    max_samples: Optional[int] = None,
) -> Tuple[Dict[int, str], int]:
    """
    Collect embeddings from all layers and save to disk as memory-mapped files.
    
    This is memory-efficient: embeddings are written to disk incrementally
    and can be loaded on-demand for alignment computation.
    
    Args:
        adapter: Model adapter
        dataloader: DataLoader for the dataset
        mode: Dataset mode (e.g., 'jwst')
        temp_dir: Directory to save temporary embedding files
        max_samples: Maximum samples to collect
    
    Returns:
        Tuple of (dict mapping layer index to file path, number of samples)
    """
    if not adapter.supports_layerwise():
        raise ValueError(f"Adapter {adapter.alias} does not support layer-wise extraction")
    
    num_layers = adapter.get_num_layers()
    
    # First pass: collect embeddings in memory (we need to know final shape)
    layer_embeddings = {i: [] for i in range(num_layers)}
    n_collected = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting {mode} layers"):
            batch_layer_embs = adapter.embed_all_layers_for_mode(batch, mode)
            
            for layer_idx, emb in batch_layer_embs.items():
                if isinstance(emb, torch.Tensor):
                    emb = emb.cpu().numpy()
                layer_embeddings[layer_idx].append(emb)
            
            n_collected += emb.shape[0] if emb.ndim > 1 else 1
            
            # Clear GPU/MPS memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            if max_samples is not None and n_collected >= max_samples:
                break
    
    # Save to disk as memory-mapped files
    file_paths = {}
    for layer_idx, emb_list in layer_embeddings.items():
        if emb_list:
            stacked = np.vstack(emb_list)
            if max_samples is not None:
                stacked = stacked[:max_samples]
            
            # Save to disk
            filepath = os.path.join(temp_dir, f"layer_{layer_idx}.npy")
            np.save(filepath, stacked)
            file_paths[layer_idx] = filepath
            
            # Clear from memory immediately
            del stacked
    
    # Clear the in-memory lists
    del layer_embeddings
    import gc
    gc.collect()
    
    return file_paths, min(n_collected, max_samples) if max_samples else n_collected


def _load_embeddings_from_disk(file_paths: Dict[int, str]) -> Dict[int, np.ndarray]:
    """
    Load embeddings from disk files.
    
    Args:
        file_paths: Dict mapping layer index to file path
    
    Returns:
        Dict mapping layer index to embedding array
    """
    return {layer_idx: np.load(path) for layer_idx, path in file_paths.items()}


def _cleanup_temp_files(temp_dir: str) -> None:
    """Remove temporary embedding files."""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def _collect_layer_embeddings(
    adapter,
    dataloader,
    mode: str,
    max_samples: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """
    Collect embeddings from all layers for all samples (in-memory version).
    
    For memory-efficient processing, use _collect_layer_embeddings_to_disk instead.
    
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
            batch_layer_embs = adapter.embed_all_layers_for_mode(batch, mode)
            
            for layer_idx, emb in batch_layer_embs.items():
                if isinstance(emb, torch.Tensor):
                    emb = emb.cpu().numpy()
                layer_embeddings[layer_idx].append(emb)
            
            n_collected += emb.shape[0] if emb.ndim > 1 else 1
            
            # Clear GPU/MPS memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            if max_samples is not None and n_collected >= max_samples:
                break
    
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


# ============================================================
# Visualization functions
# ============================================================

def plot_alignment_heatmap(
    alignment_matrix: np.ndarray,
    metric_name: str,
    model_a_label: str,
    model_b_label: str,
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "viridis",
    show_optimal: bool = True,
) -> plt.Figure:
    """
    Plot a heatmap of the alignment matrix between two models.
    
    Args:
        alignment_matrix: (num_layers_a, num_layers_b) alignment scores
        metric_name: Name of the metric for the title
        model_a_label: Label for model A (y-axis)
        model_b_label: Label for model B (x-axis)
        output_path: If provided, save figure to this path
        ax: Matplotlib axes to plot on (creates new figure if None)
        cmap: Colormap to use
        show_optimal: If True, mark the optimal pair with a star
    
    Returns:
        matplotlib Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()
    
    # Create heatmap
    im = ax.imshow(alignment_matrix, cmap=cmap, aspect="auto", origin="lower")
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label="Alignment Score")
    
    # Mark optimal pair
    if show_optimal:
        layer_a, layer_b, score = find_optimal_pair(alignment_matrix)
        ax.scatter(layer_b, layer_a, marker="*", s=300, c="red", edgecolors="white", linewidths=2, zorder=10)
        ax.annotate(f"Max: {score:.3f}", (layer_b, layer_a), 
                   xytext=(5, 5), textcoords="offset points", fontsize=10, color="red")
    
    # Labels
    ax.set_xlabel(f"{model_b_label} Layer Index")
    ax.set_ylabel(f"{model_a_label} Layer Index")
    ax.set_title(f"Layer Alignment: {metric_name}")
    
    # Set ticks with smaller font size
    ax.set_xticks(range(alignment_matrix.shape[1]))
    ax.set_yticks(range(alignment_matrix.shape[0]))
    ax.tick_params(axis='both', which='major', labelsize=7)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved heatmap: {output_path}")
    
    return fig


def plot_diagonal_scores(
    diagonal_scores: Dict[str, List[float]],
    model_a_label: str,
    model_b_label: str,
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot diagonal scores (same layer index comparisons) across metrics.
    
    Diagonal scores show how well corresponding layers align between models.
    High diagonal scores suggest that layer i in model A aligns well with 
    layer i in model B, supporting the hypothesis that models develop 
    similar representations at similar depths.
    
    Args:
        diagonal_scores: Dict mapping metric name to list of diagonal scores
        model_a_label: Label for model A
        model_b_label: Label for model B
        output_path: If provided, save figure to this path
        ax: Matplotlib axes to plot on
    
    Returns:
        matplotlib Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()
    
    # Plot each metric
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]
    colors = plt.cm.tab10.colors
    
    for idx, (metric_name, scores) in enumerate(diagonal_scores.items()):
        layer_indices = list(range(len(scores)))
        ax.plot(layer_indices, scores, 
               marker=markers[idx % len(markers)], 
               color=colors[idx % len(colors)],
               label=metric_name, 
               linewidth=2, 
               markersize=8)
    
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Alignment Score")
    ax.set_title(f"Diagonal Alignment: {model_a_label} vs {model_b_label}\n(Layer i ↔ Layer i comparisons)")
    ax.legend(loc="best", title="Metric")
    ax.set_xticks(range(len(list(diagonal_scores.values())[0])))
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved diagonal plot: {output_path}")
    
    return fig


def plot_all_alignment_matrices(
    result: "LayerwiseResult",
    output_dir: str,
    cmap: str = "viridis",
) -> List[plt.Figure]:
    """
    Generate heatmap plots for all alignment matrices in a LayerwiseResult.
    
    Args:
        result: LayerwiseResult object with alignment matrices
        output_dir: Directory to save plots
        cmap: Colormap to use
    
    Returns:
        List of matplotlib Figure objects
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_a_label = f"{result.model_a_alias}-{result.model_a_size}"
    model_b_label = f"{result.model_b_alias}-{result.model_b_size}"
    
    figures = []
    
    print("\n[Plots] Generating alignment heatmaps...")
    
    for metric_name, matrix_list in result.alignment_matrices.items():
        matrix = np.array(matrix_list)
        
        output_path = os.path.join(
            output_dir,
            f"heatmap_{result.mode}_{metric_name}_{model_a_label}_vs_{model_b_label}.png"
        )
        
        fig = plot_alignment_heatmap(
            matrix, metric_name, model_a_label, model_b_label,
            output_path=output_path, cmap=cmap
        )
        figures.append(fig)
        plt.close(fig)
    
    # Also plot diagonal scores
    print("[Plots] Generating diagonal scores plot...")
    diag_output_path = os.path.join(
        output_dir,
        f"diagonal_{result.mode}_{model_a_label}_vs_{model_b_label}.png"
    )
    
    fig = plot_diagonal_scores(
        result.diagonal_scores, model_a_label, model_b_label,
        output_path=diag_output_path
    )
    figures.append(fig)
    plt.close(fig)
    
    return figures


def plot_mknn_k_sensitivity(
    result: "LayerwiseResult",
    output_dir: str,
) -> plt.Figure:
    """
    Plot how MKNN scores vary with different k values.
    
    Shows the sensitivity of the alignment measure to the choice of k,
    both for the optimal pair and along the diagonal.
    
    Args:
        result: LayerwiseResult object
        output_dir: Directory to save plot
    
    Returns:
        matplotlib Figure object
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract MKNN results
    mknn_metrics = {k: v for k, v in result.max_scores.items() if k.startswith("mknn_k")}
    
    if not mknn_metrics:
        print("  No MKNN metrics found, skipping k-sensitivity plot")
        return None
    
    # Parse k values and scores
    k_values = []
    max_scores = []
    for metric_key in sorted(mknn_metrics.keys(), key=lambda x: int(x.replace("mknn_k", ""))):
        k = int(metric_key.replace("mknn_k", ""))
        k_values.append(k)
        max_scores.append(mknn_metrics[metric_key])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_values, max_scores, marker="o", linewidth=2, markersize=10, color="tab:blue")
    
    ax.set_xlabel("k (number of nearest neighbors)")
    ax.set_ylabel("Maximum MKNN Score")
    ax.set_title(f"MKNN Sensitivity to k\n{result.model_a_alias}-{result.model_a_size} vs {result.model_b_alias}-{result.model_b_size}")
    
    # Mark each point with its value
    for k, score in zip(k_values, max_scores):
        ax.annotate(f"{score:.3f}", (k, score), xytext=(0, 10), 
                   textcoords="offset points", ha="center", fontsize=10)
    
    plt.tight_layout()
    
    model_a_label = f"{result.model_a_alias}-{result.model_a_size}"
    model_b_label = f"{result.model_b_alias}-{result.model_b_size}"
    output_path = os.path.join(
        output_dir,
        f"mknn_k_sensitivity_{result.mode}_{model_a_label}_vs_{model_b_label}.png"
    )
    
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved k-sensitivity plot: {output_path}")
    plt.close(fig)
    
    return fig


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
    generate_plots: bool = True,
    force_cpu: bool = False,
    include_llm: bool = False,
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
        mknn_k_values = [3, 9, 12, 15, 18, 21, 30, 40]
    
    # Model configurations
    model_map = {
        "smolvlm": {
            "256M": "HuggingFaceTB/SmolVLM-256M-Instruct",
            "500M": "HuggingFaceTB/SmolVLM-500M-Instruct",
            "2.2B": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        },
   #     "vit": {
   #         "base": "google/vit-base-patch16-224-in21k",
   #         "large": "google/vit-large-patch16-224-in21k",
   #         "huge": "google/vit-huge-patch14-224-in21k",
   #     },
   #     "dino": {
   #         "small": "facebook/dinov2-with-registers-small",
   #         "base": "facebook/dinov2-with-registers-base",
   #         "large": "facebook/dinov2-with-registers-large",
   #         "giant": "facebook/dinov2-with-registers-giant",
   #     },
   #     "dinov3": {
   #         "vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
   #         "vits16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
   #         "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
   #         "vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
   #         "vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
   #         "vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
   #         "convnext-base": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
   #         "convnext-large": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
   #         "convnext-small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
   #         "convnext-tiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
   #         "vitl16-sat493m": "facebook/dinov3-vitl16-pretrain-sat493m",
   #         "vit7b16-sat493m": "facebook/dinov3-vit7b16-pretrain-sat493m",
   #     },
   #     "convnext": {
   #         "nano": "facebook/convnextv2-nano-22k-224",
   #         "tiny": "facebook/convnextv2-tiny-22k-224",
   #         "base": "facebook/convnextv2-base-22k-224",
   #         "large": "facebook/convnextv2-large-22k-224",
   #     },
   #     "ijepa": {
   #         "huge": "facebook/ijepa_vith14_22k",
   #         "giant": "facebook/ijepa_vitg16_22k",
   #     },
   #     "vjepa": {
   #         "large": "facebook/vjepa2-vitl-fpc64-256",
   #         "huge": "facebook/vjepa2-vith-fpc64-256",
   #         "giant": "facebook/vjepa2-vitg-fpc64-256",
   #     },
   #     "astropt": {
   #         "015M": "Smith42/astroPT_v2.0",
   #         "095M": "Smith42/astroPT_v2.0",
   #         "850M": "Smith42/astroPT_v2.0",
   #     },
   #     "sam2": {
   #         "tiny": "facebook/sam2.1-hiera-tiny",
   #         "small": "facebook/sam2.1-hiera-small",
    #        "base-plus": "facebook/sam2.1-hiera-base-plus",
    #        "large": "facebook/sam2.1-hiera-large",
    #    },
    #    "vit-mae": {
    #        "base": "facebook/vit-mae-base",
    #        "large": "facebook/vit-mae-large",
    #        "huge": "facebook/vit-mae-huge",
    #    },
    #    "hiera": {
    #        "tiny": "facebook/hiera-tiny-224-hf",
    #        "small": "facebook/hiera-small-224-hf",
    #        "base-plus": "facebook/hiera-base-plus-224-hf",
    #        "large": "facebook/hiera-large-224-hf",
    #    },
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
    if force_cpu:
        print("  -> Forcing CPU (slower but uses system RAM)")
    adapter_a_cls = get_adapter(model_a_alias)
    adapter_a = adapter_a_cls(model_a_name, model_a_size, alias=model_a_alias)
    adapter_a.load(force_cpu=force_cpu, include_llm=include_llm)
    
    if not adapter_a.supports_layerwise():
        raise ValueError(f"Model {model_a_alias} does not support layer-wise extraction")
    
    num_layers_a = adapter_a.get_num_layers()
    print(f"  -> {num_layers_a} layers")
    
    # Load Model B
    print(f"\n[2/5] Loading Model B: {model_b_alias}-{model_b_size}")
    adapter_b_cls = get_adapter(model_b_alias)
    adapter_b = adapter_b_cls(model_b_name, model_b_size, alias=model_b_alias)
    adapter_b.load(force_cpu=force_cpu, include_llm=include_llm)
    
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
    
    # Create temporary directory for embeddings (memory-efficient disk storage)
    temp_dir = tempfile.mkdtemp(prefix="platonic_embeddings_")
    temp_dir_a = os.path.join(temp_dir, "model_a")
    temp_dir_b = os.path.join(temp_dir, "model_b")
    os.makedirs(temp_dir_a, exist_ok=True)
    os.makedirs(temp_dir_b, exist_ok=True)
    
    try:
        # Extract embeddings from Model A and save to disk
        print("\n[4/5] Extracting embeddings from Model A (saving to disk)")
        file_paths_a, n_samples = _collect_layer_embeddings_to_disk(
            adapter_a, dl, modes[0], temp_dir_a, max_samples
        )
        print(f"  -> Collected {n_samples} samples, {len(file_paths_a)} layers")
        print(f"  -> Saved to: {temp_dir_a}")
        
        # Unload Model A to free memory
        del adapter_a
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print("  -> Model A unloaded to free memory")
        
        # Need to reload dataset for Model B with its preprocessor
        print("\n[4b/5] Extracting embeddings from Model B (saving to disk)")
        processor_b = adapter_b.get_preprocessor(modes)
        dataset_adapter_b = dataset_adapter_cls(hf_ds, mode)
        dataset_adapter_b.load()
        ds_b = dataset_adapter_b.prepare(processor_b, modes, filterfun)
        
        if max_samples is not None:
            ds_b = ds_b.take(max_samples)
        
        dl_b = DataLoader(ds_b, batch_size=batch_size, num_workers=num_workers)
        file_paths_b, _ = _collect_layer_embeddings_to_disk(
            adapter_b, dl_b, modes[0], temp_dir_b, max_samples
        )
        print(f"  -> Collected {len(file_paths_b)} layers")
        print(f"  -> Saved to: {temp_dir_b}")
        
        # Unload Model B to free memory
        del adapter_b
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print("  -> Model B unloaded to free memory")
        
        # Load embeddings from disk for alignment computation
        print("\n[5/5] Computing alignment matrices (loading from disk)")
        embeddings_a = _load_embeddings_from_disk(file_paths_a)
        embeddings_b = _load_embeddings_from_disk(file_paths_b)
        
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
        
        # Generate plots if requested
        if generate_plots:
            plots_dir = os.path.join(output_dir, "plots")
            plot_all_alignment_matrices(result, plots_dir)
            plot_mknn_k_sensitivity(result, plots_dir)
        
        print(f"{'='*60}\n")
        
    finally:
        # Clean up temporary files
        print("\nCleaning up temporary embedding files...")
        _cleanup_temp_files(temp_dir)
        print(f"  -> Removed: {temp_dir}")
    
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
