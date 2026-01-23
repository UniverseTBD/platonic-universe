import os
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pu.models import get_adapter
from pu.pu_datasets import get_dataset_adapter
from pu.metrics import mknn, compute_cka_mmap, wass_distance
from pu.utils import write_bin


# Centralized model configurations
MODEL_MAP = {
    "vit": (
        ["base", "large", "huge"],
        [
            "google/vit-base-patch16-224-in21k",
            "google/vit-large-patch16-224-in21k",
            "google/vit-huge-patch14-224-in21k",
        ],
    ),
    "dino": (
        ["small", "base", "large", "giant"],
        [f"facebook/dinov2-with-registers-{s}" for s in ["small", "base", "large", "giant"]],
    ),
    "dinov3": (
        [
            "vits16", "vits16plus", "vitb16", "vitl16", "vith16plus", "vit7b16",
            "convnext-base", "convnext-large", "convnext-small", "convnext-tiny",
            "vitl16-sat493m", "vit7b16-sat493m",
        ],
        [
            "facebook/dinov3-vits16-pretrain-lvd1689m",
            "facebook/dinov3-vits16plus-pretrain-lvd1689m",
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "facebook/dinov3-vith16plus-pretrain-lvd1689m",
            "facebook/dinov3-vit7b16-pretrain-lvd1689m",
            "facebook/dinov3-convnext-base-pretrain-lvd1689m",
            "facebook/dinov3-convnext-large-pretrain-lvd1689m",
            "facebook/dinov3-convnext-small-pretrain-lvd1689m",
            "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
            "facebook/dinov3-vitl16-pretrain-sat493m",
            "facebook/dinov3-vit7b16-pretrain-sat493m",
        ],
    ),
    "convnext": (
        ["nano", "tiny", "base", "large"],
        [f"facebook/convnextv2-{s}-22k-224" for s in ["nano", "tiny", "base", "large"]],
    ),
    "ijepa": (
        ["huge", "giant"],
        ["facebook/ijepa_vith14_22k", "facebook/ijepa_vitg16_22k"],
    ),
    "vjepa": (
        ["large", "huge", "giant"],
        [
            "facebook/vjepa2-vitl-fpc64-256",
            "facebook/vjepa2-vith-fpc64-256",
            "facebook/vjepa2-vitg-fpc64-256",
        ],
    ),
    "astropt": (
        ["015M", "095M", "850M"],
        ["Smith42/astroPT_v2.0" for _ in range(3)],
    ),
    "sam2": (
        ["tiny", "small", "base-plus", "large"],
        [
            "facebook/sam2.1-hiera-tiny",
            "facebook/sam2.1-hiera-small",
            "facebook/sam2.1-hiera-base-plus",
            "facebook/sam2.1-hiera-large",
        ],
    ),
    "vit-mae": (
        ["base", "large", "huge"],
        [f"facebook/vit-mae-{s}" for s in ["base", "large", "huge"]],
    ),
    "hiera": (
        ["tiny", "small", "base-plus", "large"],
        [f"facebook/hiera-{s}-224-hf" for s in ["tiny", "small", "base-plus", "large"]],
    ),
}


def get_model_config(model_alias: str):
    """Get model sizes and names for a given alias."""
    if model_alias not in MODEL_MAP:
        raise NotImplementedError(
            f"Model '{model_alias}' not implemented. "
            f"Available models: {list(MODEL_MAP.keys())}"
        )
    return MODEL_MAP[model_alias]


def list_models() -> List[str]:
    """Return list of available model aliases."""
    return list(MODEL_MAP.keys())


def _create_filter_function(comp_mode: str):
    """Create the appropriate filter function for a given mode."""
    def filterfun(idx):
        if comp_mode == "physical":
            return True
        if comp_mode != "jwst":
            return True
        im = idx["jwst_image"]["flux"][3]
        v0, v1 = np.nanpercentile(im, 5), np.nanpercentile(im, 99)
        return v0 - v1 != 0
    return filterfun


def _generate_embeddings_for_size(
    model_alias: str,
    model_name: str,
    size: str,
    modes: List[str],
    comp_mode: str,
    hf_ds: str,
    batch_size: int,
    num_workers: int,
    physical_params: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate embeddings for a single model size.
    
    Returns dict with embeddings per mode and optionally physical parameters.
    """
    adapter_cls = get_adapter(model_alias)
    adapter = adapter_cls(model_name, size, alias=model_alias)
    adapter.load()
    processor = adapter.get_preprocessor(modes)

    dataset_adapter_cls = get_dataset_adapter(comp_mode)
    dataset_adapter = dataset_adapter_cls(hf_ds, comp_mode)
    dataset_adapter.load()
    
    filterfun = _create_filter_function(comp_mode)
    ds = dataset_adapter.prepare(processor, modes, filterfun, physical_params=physical_params)
    if max_samples is not None:
        ds = ds.take(max_samples)

    dl = iter(DataLoader(ds, batch_size=batch_size, num_workers=num_workers))

    zs = {mode: [] for mode in modes}
    params_collected = {p: [] for p in (physical_params or [])}
    
    with torch.no_grad():
        for B in tqdm(dl, desc=f"Processing {model_alias}/{size}"):
            for mode in modes:
                if mode == "sdss":
                    zs[mode].append(torch.tensor(np.array(B["embedding"])).T)
                elif mode == "desi":
                    zs[mode].append(torch.tensor(np.array(B["embeddings"])).T)
                else:
                    outputs = adapter.embed_for_mode(B, mode)
                    zs[mode].append(outputs)
            
            # Collect physical parameters if requested
            for param in (physical_params or []):
                if param in B:
                    params_collected[param].append(np.array(B[param]))

    result = {
        "embeddings": {mode: torch.cat(embs) for mode, embs in zs.items()},
    }
    
    if physical_params:
        result["params"] = {
            param: np.concatenate(vals) for param, vals in params_collected.items() if vals
        }
    
    return result


def _compute_cross_modal_metrics(
    zs: Dict[str, torch.Tensor],
    modes: List[str],
    knn_k: int,
    physical_params: Optional[List[str]],
    params: Optional[Dict[str, np.ndarray]],
    n_samples: Optional[int],
) -> Dict[str, Any]:
    """Compute metrics comparing two different modalities."""
    
    # Compute mKNN
    mknn_score = mknn(
        zs[modes[0]].cpu().numpy(), 
        zs[modes[1]].cpu().numpy(), 
        knn_k
    )

    # Compute CKA using memory-mapped files for large matrices
    temp1 = tempfile.NamedTemporaryFile(delete=False)
    temp2 = tempfile.NamedTemporaryFile(delete=False)
    temp1.close()
    temp2.close()

    k1 = zs[modes[0]].cpu().numpy() @ zs[modes[0]].cpu().numpy().T
    k2 = zs[modes[1]].cpu().numpy() @ zs[modes[1]].cpu().numpy().T

    write_bin(k1, str(temp1.name))
    write_bin(k2, str(temp2.name))

    cka_score = compute_cka_mmap(str(temp1.name), str(temp2.name), k1.shape[0], k1.shape[1])
    
    # Clean up temp files
    Path(temp1.name).unlink(missing_ok=True)
    Path(temp2.name).unlink(missing_ok=True)

    metrics = {
        "mknn": float(mknn_score),
        "cka": float(cka_score),
    }
    
    # Compute Wasserstein distances if physical params provided
    if physical_params and params:
        z1 = zs[modes[0]].cpu().numpy()
        z2 = zs[modes[1]].cpu().numpy()
        
        if n_samples:
            n = min(n_samples, len(z1), len(z2))
            z1, z2 = z1[:n], z2[:n]
            params = {k: v[:n] for k, v in params.items()}
        
        wass_results = wass_distance(z1, z2, k=knn_k, params=params)
        metrics["wasserstein"] = {k: float(v) for k, v in wass_results.items()}

    return metrics


def run_experiment(
    model_alias: str,
    mode: str,
    output_dataset: Optional[str] = None,
    batch_size: int = 128,
    num_workers: int = 0,
    knn_k: int = 10,
    sizes: Optional[List[str]] = None,
    physical_params: Optional[List[str]] = None,
    n_samples: Optional[int] = None,
    intramodal: bool = False,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run embedding experiment.
    
    Supports two experiment types:
    - Cross-modal (default): Compare embeddings between HSC and another modality
    - Intra-modal (--intramodal): Compare embeddings across model sizes within same modality
    
    Args:
        model_alias: Model family to use (e.g., 'vit', 'dino', 'convnext')
        mode: Dataset to compare to HSC ('jwst', 'legacysurvey', 'sdss', 'desi', 'physical')
        output_dataset: Optional HuggingFace dataset to push results to
        batch_size: Batch size for processing
        num_workers: Number of data loader workers
        knn_k: K value for mutual KNN calculation
        sizes: Optional list of sizes to run (defaults to all available)
        physical_params: Optional list of physical parameters for Wasserstein analysis
        n_samples: Number of samples for physical parameter analysis (default: all)
        intramodal: If True, compare across model sizes; if False, compare across modalities
    
    Returns:
        Dictionary containing experiment results
    """
    comp_mode = mode

    if mode == "galaxies":
        modes = ["galaxies"]
        hf_ds = "Smith42/galaxies"
    elif mode == "physical":
        modes = ["hsc", "jwst"]
        hf_ds = "Ashodkh/hsc-jwst-images-high-snr"
    else:
        modes = ["hsc", comp_mode]
        hf_ds = f"Smith42/{comp_mode}_hsc_crossmatched"

    available_sizes, model_names = get_model_config(model_alias)
    
    # Filter to requested sizes if specified
    if sizes:
        size_indices = [i for i, s in enumerate(available_sizes) if s in sizes]
        if not size_indices:
            raise ValueError(f"None of the requested sizes {sizes} are available. Available: {available_sizes}")
        available_sizes = [available_sizes[i] for i in size_indices]
        model_names = [model_names[i] for i in size_indices]

    os.makedirs("data", exist_ok=True)
    
    if intramodal:
        return _run_intramodal(
            model_alias=model_alias,
            available_sizes=available_sizes,
            model_names=model_names,
            modes=modes,
            comp_mode=comp_mode,
            hf_ds=hf_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            knn_k=knn_k,
            physical_params=physical_params,
            n_samples=n_samples,
	    max_samples=max_samples,
        )
    else:
        return _run_cross_modal(
            model_alias=model_alias,
            available_sizes=available_sizes,
            model_names=model_names,
            modes=modes,
            comp_mode=comp_mode,
            hf_ds=hf_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            knn_k=knn_k,
            physical_params=physical_params,
            n_samples=n_samples,
	    max_samples=max_samples,
        )


def _run_cross_modal(
    model_alias: str,
    available_sizes: List[str],
    model_names: List[str],
    modes: List[str],
    comp_mode: str,
    hf_ds: str,
    batch_size: int,
    num_workers: int,
    knn_k: int,
    physical_params: Optional[List[str]],
    n_samples: Optional[int],
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Run cross-modal experiment comparing HSC to another modality."""
    
    df = pl.DataFrame()
    all_results = []
    
    for size, model_name in zip(available_sizes, model_names):
        print(f"\n{'='*50}")
        print(f"Processing {model_alias}/{size}")
        print(f"{'='*50}")
        
        result = _generate_embeddings_for_size(
            model_alias=model_alias,
            model_name=model_name,
            size=size,
            modes=modes,
            comp_mode=comp_mode,
            hf_ds=hf_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            physical_params=physical_params,
	    max_samples=max_samples,
        )
        
        zs = result["embeddings"]
        params = result.get("params")
        
        metrics = _compute_cross_modal_metrics(
            zs=zs,
            modes=modes,
            knn_k=knn_k,
            physical_params=physical_params,
            params=params,
            n_samples=n_samples,
        )

        size_result = {"model": model_alias, "size": size, **metrics}

        print(f"\nCKA {model_alias}/{size}: {metrics['cka']:.8f}")
        print(f"mKNN {model_alias}/{size}: {metrics['mknn']:.8f}")
        
        if "wasserstein" in metrics:
            print("Wasserstein distances:")
            for param, dist in metrics["wasserstein"].items():
                print(f"  {param}: {dist:.6f}")

        # Save scores to file
        with open(f"data/{comp_mode}_{model_alias}_scores.txt", "a") as fi:
            fi.write(f"{model_alias} {size}, mknn: {metrics['mknn']:.8f}, cka: {metrics['cka']:.8f}\n")

        # Add embeddings to dataframe
        df = df.with_columns(
            [
                pl.Series(
                    f"{model_alias}_{size.lstrip('0')}_{m}".lower(),
                    embs.cpu().numpy(),
                )
                for m, embs in zs.items()
            ]
        )

        df.write_parquet(f"data/{comp_mode}_{model_alias}_{size}.parquet")
        all_results.append(size_result)

    return {
        "experiment": "cross_modal",
        "model": model_alias,
        "mode": comp_mode,
        "results": all_results,
    }


def _run_intramodal(
    model_alias: str,
    available_sizes: List[str],
    model_names: List[str],
    modes: List[str],
    comp_mode: str,
    hf_ds: str,
    batch_size: int,
    num_workers: int,
    knn_k: int,
    physical_params: Optional[List[str]],
    n_samples: int,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Run intra-modal experiment comparing embeddings across model sizes."""
    
    print(f"\nRunning intra-modal experiment for {model_alias}")
    print(f"Sizes: {available_sizes}")
    if physical_params:
        print(f"Physical params: {physical_params}")
    
    # Generate embeddings for all sizes
    all_embeddings = {}
    all_params = None
    
    for size, model_name in zip(available_sizes, model_names):
        print(f"\n{'='*50}")
        print(f"Generating embeddings for {model_alias}/{size}")
        print(f"{'='*50}")
        
        result = _generate_embeddings_for_size(
            model_alias=model_alias,
            model_name=model_name,
            size=size,
            modes=modes,
            comp_mode=comp_mode,
            hf_ds=hf_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            physical_params=physical_params,
            max_samples=max_samples,
        )
        
        # Use HSC embeddings for intra-modal comparison
        all_embeddings[size] = result["embeddings"]["hsc"].cpu().numpy()
        
        # Store params from first size (they're the same across sizes)
        if all_params is None and "params" in result:
            all_params = result["params"]
    
    # Compare consecutive sizes
    comparisons = []
    
    for i in range(len(available_sizes) - 1):
        size1, size2 = available_sizes[i], available_sizes[i + 1]
        z1, z2 = all_embeddings[size1], all_embeddings[size2]
        
        # Limit samples
        n = min(n_samples, len(z1), len(z2))
        z1_sample, z2_sample = z1[:n], z2[:n]
        
        comparison = {
            "sizes": f"{size1}_vs_{size2}",
            "mknn": float(mknn(z1_sample, z2_sample, k=knn_k)),
        }
        
        # Compute Wasserstein distances if physical params available
        if all_params:
            params_sample = {k: v[:n] for k, v in all_params.items()}
            wass_results = wass_distance(z1_sample, z2_sample, k=knn_k, params=params_sample)
            comparison["wasserstein"] = {k: float(v) for k, v in wass_results.items()}
        
        comparisons.append(comparison)
        
        print(f"\n{size1} vs {size2}:")
        print(f"  mKNN: {comparison['mknn']:.4f}")
        if "wasserstein" in comparison:
            for param, dist in comparison["wasserstein"].items():
                print(f"  Wasserstein({param}): {dist:.6f}")
    
    return {
        "experiment": "intramodal",
        "model": model_alias,
        "mode": comp_mode,
        "sizes": available_sizes,
        "comparisons": comparisons,
    }
