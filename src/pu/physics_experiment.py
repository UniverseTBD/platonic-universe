"""
Physics test runner.

Generates embeddings from Smith42/galaxies (v2.0) and evaluates how well
they predict physical galaxy properties using linear probes, neighbour
consistency, and distance correlation.

This complements the cross-modal experiments (experiments.py) by testing
whether embeddings encode real astrophysics, not just statistical structure.
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pu.models import get_adapter
from pu.pu_datasets import get_dataset_adapter
from pu.metrics.physics import (
    ALL_PROPERTIES,
    DEFAULT_PROPERTIES,
    run_physics_tests,
)


# Model map mirroring experiments.py so all models can be physics-tested.
PHYSICS_MODEL_MAP = {
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
        ["vits16", "vits16plus", "vitb16", "vitl16", "vith16plus", "vit7b16"],
        [
            "facebook/dinov3-vits16-pretrain-lvd1689m",
            "facebook/dinov3-vits16plus-pretrain-lvd1689m",
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "facebook/dinov3-vith16plus-pretrain-lvd1689m",
            "facebook/dinov3-vit7b16-pretrain-lvd1689m",
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
        [f"Smith42/astroPT_v2.0" for _ in range(3)],
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


def _make_galaxies_preprocessor(adapter, model_alias):
    """Build a preprocessor that works with Smith42/galaxies image column.

    For HF-based models we wrap the raw autoprocessor directly around the
    PIL image.  For astropt / sam2 we delegate to the adapter's own
    get_preprocessor but remap the column name so the existing preprocessor
    sees image as if it were an HSC flux image.

    In all cases the resulting dataset column used for embedding is called
    ``"galaxies"`` so the inference loop can use ``adapter.embed_for_mode(B, "galaxies")``.
    """
    import numpy as np

    # ---- HF-style models (vit, dino, dinov3, convnext, ijepa, vjepa, vit-mae, hiera) ----
    if hasattr(adapter, "processor") and adapter.processor is not None:
        proc = adapter.processor

        def hf_wrapper(example):
            img = example.get("image")
            if img is None:
                raise KeyError("No 'image' column")

            proc_out = proc(img, return_tensors="pt")
            if "pixel_values" in proc_out:
                result = {"galaxies": proc_out["pixel_values"].squeeze()}
            elif "pixel_values_videos" in proc_out:
                result = {
                    "galaxies": proc_out["pixel_values_videos"]
                    .repeat(1, 16, 1, 1, 1)
                    .squeeze()
                }
            else:
                raise KeyError("Processor output missing pixel_values")
            return result

        return hf_wrapper

    # ---- SAM2 ----
    if model_alias == "sam2":
        sam2_transforms = adapter.predictor._transforms

        def sam2_wrapper(example):
            img = example.get("image")
            if img is None:
                raise KeyError("No 'image' column")
            arr = np.asarray(img)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            transformed = sam2_transforms(arr)
            return {"galaxies": transformed}

        return sam2_wrapper

    # ---- AstroPT ----
    if model_alias == "astropt":
        import torch as _torch
        from astropt.local_datasets import GalaxyImageDataset
        from torchvision import transforms

        def _normalise(x):
            std, mean = _torch.std_mean(x, dim=1, keepdim=True)
            return (x - mean) / (std + 1e-8)

        data_tf = transforms.Compose([transforms.Lambda(_normalise)])
        galproc = GalaxyImageDataset(
            None,
            spiral=True,
            transform={"images": data_tf},
            modality_registry=adapter.model.modality_registry,
        )

        def astropt_wrapper(example):
            img = example.get("image")
            if img is None:
                raise KeyError("No 'image' column")
            arr = np.asarray(img, dtype=np.float32)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            # (H,W,C) -> (C,H,W) for astropt
            arr = arr.swapaxes(0, 2)
            im = galproc.process_galaxy(
                _torch.from_numpy(arr).to(_torch.float)
            ).to(_torch.float)
            return {
                "galaxies_images": im,
                "galaxies_positions": _torch.arange(0, len(im), dtype=_torch.long),
            }

        return astropt_wrapper

    # Fallback: shouldn't happen if model map is correct
    raise ValueError(f"Don't know how to build a galaxies preprocessor for '{model_alias}'")


def run_physics_experiment(
    model_alias: str,
    split: str = "test",
    max_samples: int | None = 5000,
    batch_size: int = 128,
    num_workers: int = 0,
    knn_k: int = 10,
    cv: int = 5,
    properties: list[str] | None = None,
    output_dir: str = "data",
) -> dict[str, Any]:
    """
    Run physics validation tests for a model against Smith42/galaxies.

    Args:
        model_alias: Model to test (e.g., 'vit', 'dino', 'convnext')
        split: Dataset split to use ('test' or 'validation')
        max_samples: Cap on number of galaxies to process (None = all)
        batch_size: Batch size for inference
        num_workers: DataLoader workers
        knn_k: k for neighbour consistency metric
        cv: Cross-validation folds for linear probe
        properties: Which physical properties to test (None = defaults)
        output_dir: Directory for result JSON files

    Returns:
        Nested dict with results per model size and property
    """
    if model_alias not in PHYSICS_MODEL_MAP:
        raise ValueError(
            f"Model '{model_alias}' not in PHYSICS_MODEL_MAP. "
            f"Available: {list(PHYSICS_MODEL_MAP.keys())}"
        )

    sizes, model_names = PHYSICS_MODEL_MAP[model_alias]
    adapter_cls = get_adapter(model_alias)

    hf_ds = "Smith42/galaxies"
    property_keys = properties or DEFAULT_PROPERTIES

    all_results: dict[str, Any] = {
        "model": model_alias,
        "split": split,
        "max_samples": max_samples,
        "property_keys": property_keys,
        "sizes": {},
    }

    for size, model_name in zip(sizes, model_names):
        print(f"\n{'='*60}")
        print(f"Physics test: {model_alias} {size}")
        print(f"{'='*60}")

        # --- Load model ---
        adapter = adapter_cls(model_name, size, alias=model_alias)
        adapter.load()

        # Build a preprocessor that maps image -> model input
        proc_fn = _make_galaxies_preprocessor(adapter, model_alias)

        # --- Load dataset ---
        dataset_adapter_cls = get_dataset_adapter("galaxies")
        dataset_adapter = dataset_adapter_cls(hf_ds, "galaxies")
        dataset_adapter.load()

        ds = dataset_adapter.prepare(
            processor=proc_fn,
            modes=["galaxies"],
            filterfun=lambda x: True,
            split=split,
            max_samples=max_samples,
        )

        dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

        # --- Inference ---
        embeddings = []
        metadata_accum: dict[str, list] = {key: [] for key in property_keys}

        with torch.no_grad():
            for B in tqdm(dl, desc=f"{model_alias}-{size}"):
                # Use the adapter's own embed_for_mode — same as experiments.py
                emb = adapter.embed_for_mode(B, "galaxies")
                embeddings.append(emb.float().cpu())

                # Accumulate metadata from this batch
                for key in property_keys:
                    col_name = ALL_PROPERTIES.get(key, key)
                    if col_name in B:
                        metadata_accum[key].extend(
                            B[col_name].numpy().tolist()
                            if hasattr(B[col_name], "numpy")
                            else list(B[col_name])
                        )

        Z = torch.cat(embeddings).numpy()
        n_samples = len(Z)
        print(f"  Embedded {n_samples} galaxies, shape: {Z.shape}")

        # --- Build property arrays ---
        prop_arrays: dict[str, np.ndarray] = {}
        for key in property_keys:
            if metadata_accum[key]:
                arr = np.array(metadata_accum[key], dtype=np.float64)
                # Only include if we have matching lengths and non-trivial data
                if len(arr) == n_samples and np.any(np.isfinite(arr)):
                    prop_arrays[key] = arr
                else:
                    print(f"  Warning: skipping {key} (length mismatch or all NaN)")
            else:
                print(f"  Warning: column for '{key}' not found in dataset")

        # --- Run physics tests ---
        print(f"\n  Running physics tests on {len(prop_arrays)} properties...")
        size_results = run_physics_tests(
            Z, prop_arrays, property_keys=list(prop_arrays.keys()), k=knn_k, cv=cv
        )

        # Print results
        print(f"\n  {'Property':<25} {'Lin R²':<12} {'Neighbor':<12} {'Dist Corr':<12}")
        print(f"  {'-'*60}")
        for prop_key, metrics in size_results.items():
            lr2 = metrics.get("linear_probe_r2", float("nan"))
            nc = metrics.get("neighbor_consistency", float("nan"))
            dc = metrics.get("distance_correlation", float("nan"))
            print(f"  {prop_key:<25} {lr2:<12.4f} {nc:<12.4f} {dc:<12.4f}")

        all_results["sizes"][size] = {
            "n_samples": n_samples,
            "embedding_dim": Z.shape[1],
            "properties": {
                k: {mk: float(mv) if np.isfinite(mv) else None for mk, mv in v.items()}
                for k, v in size_results.items()
            },
        }

    # --- Save results ---
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"physics_{model_alias}_{split}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return all_results
