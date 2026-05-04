"""
Physics test runner.

Generates embeddings from Smith42/galaxies (v2.0) and evaluates how well
they predict physical galaxy properties using linear probes, neighbour
consistency, and distance correlation.

This complements the cross-modal experiments (experiments.py) by testing
whether embeddings encode real astrophysics, not just statistical structure.
"""

import polars as pl
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    _clean_inputs,
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
    "clip": (
        ["base", "large"],
        [
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-large-patch14",
        ],
    ),
    "dino": (
        ["small", "base", "large", "giant"],
        [f"facebook/dinov2-with-registers-{s}" for s in ["small", "base", "large", "giant"]],
    ),
    "aion": (
        ["300M"],                                                      
        ["polymathic-ai/aion-base"],
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
    "paligemma": (
        ["3b", "10b", "28b"],
        [
            "google/paligemma2-3b-mix-224",
            "google/paligemma2-10b-mix-224",
            "google/paligemma2-28b-mix-224",
        ],
    ),
    "llava_15": (
        ["7b", "13b"],
        [
            "llava-hf/llava-1.5-7b-hf",
            "llava-hf/llava-1.5-13b-hf",
        ],
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

            if adapter.alias == "clip":
                proc_out = proc(images=img, return_tensors="pt")
            elif hasattr(adapter, "_PROMPTS"):
                # VLM adapters (PaliGemma, LLaVA, etc.) need text + images
                prompt = adapter._PROMPTS.get(adapter.alias, "<image> ")
                proc_out = proc(text=prompt, images=img, return_tensors="pt")
            else:
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

    # ---- AION ----
    if model_alias == "aion":
        import torch as _torch
        from aion.codecs.manager import CodecManager
        from aion.modalities import LegacySurveyImage

        codec_mgr = CodecManager(device="cpu")

        def aion_wrapper(example):
            img = example.get("image")
            if img is None:
                raise KeyError("No 'image' column")
            flux = np.asarray(img, dtype=np.float32)
            if flux.ndim == 2:
                flux = np.stack([flux, flux, flux], axis=-1)
            # (H, W, C) -> (C, H, W) and add batch dim
            flux_t = _torch.from_numpy(flux).permute(2, 0, 1).unsqueeze(0)
            bands = ["DES-G", "DES-R", "DES-Z"]
            modality = LegacySurveyImage(flux=flux_t, bands=bands)
            tokens = codec_mgr.encode(modality)
            # tokens is {"tok_image": (1, N)} — squeeze batch dim
            return {"galaxies": tokens["tok_image"].squeeze(0)}

        return aion_wrapper

    # Fallback: shouldn't happen if model map is correct
    raise ValueError(f"Don't know how to build a galaxies preprocessor for '{model_alias}'")


def _compute_2d_projection(
    Z: np.ndarray,
    method: str = "pca",
    random_state: int = 42,
) -> np.ndarray:
    """Reduce embeddings to 2D for visualisation.

    Args:
        Z: (n_samples, d) embedding matrix
        method: 'pca' or 'umap'
        random_state: Seed for reproducibility

    Returns:
        (n_samples, 2) projected coordinates
    """
    if method == "umap":
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=random_state)
            return reducer.fit_transform(Z)
        except ImportError:
            print("  Warning: umap-learn not installed, falling back to PCA")
            method = "pca"

    if method == "pca":
        # pca is bettterrr
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=random_state).fit_transform(Z)

    raise ValueError(f"Unknown projection method: {method}")


def plot_physics_embeddings(
    Z: np.ndarray,
    prop_arrays: dict[str, np.ndarray],
    size_results: dict[str, dict[str, float]],
    model_alias: str,
    size: str,
    output_dir: str,
    split: str = "test",
    method: str = "pca",
    max_plot_samples: int = 10_000,
) -> str:
    """Generate a grid of 2-D scatter plots, one per physical property.

    Each panel colours points by the property value and annotates the
    linear-probe R², neighbour consistency and distance correlation.

    Args:
        Z: (n_samples, d) embedding matrix
        prop_arrays: {property_key: (n_samples,) array}
        size_results: output of run_physics_tests for this size
        model_alias: e.g. 'vit'
        size: e.g. 'base'
        output_dir: directory to write the PNG into
        split: dataset split name (for the filename)
        method: 'pca' or 'umap'
        max_plot_samples: subsample for speed / readability

    Returns:
        Path to the saved figure.
    """
    n_props = len(prop_arrays)
    if n_props == 0:
        return ""

    # Subsample if needed
    n = len(Z)
    if n > max_plot_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_plot_samples, replace=False)
        Z_sub = Z[idx]
        prop_sub = {k: v[idx] for k, v in prop_arrays.items()}
    else:
        Z_sub = Z
        prop_sub = prop_arrays

    coords = _compute_2d_projection(Z_sub, method=method)

    ncols = min(n_props, 4)
    nrows = (n_props + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                             squeeze=False)

    method_label = method.upper()

    for ax, (prop_key, y) in zip(axes.flat, prop_sub.items()):
        try:
            Z_clean, y_clean = _clean_inputs(coords, y)
        except ValueError:
            ax.set_title(f"{prop_key} (no valid data)", fontsize=10)
            continue
        sc = ax.scatter(
            Z_clean[:, 0], Z_clean[:, 1],
            c=y_clean, s=1, alpha=0.6, cmap="viridis", rasterized=True,
        )
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel(f"{method_label}1")
        ax.set_ylabel(f"{method_label}2")
        ax.set_title(prop_key, fontsize=10)

        # Annotate metrics
        metrics = size_results.get(prop_key, {})
        lr2 = metrics.get("linear_probe_r2", float("nan"))
        ax.text(
            0.02, 0.98,
            f"R²={lr2:.3f}",
            transform=ax.transAxes, fontsize=7,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    # Hide unused axes
    for ax in axes.flat[n_props:]:
        ax.set_visible(False)

    fig.suptitle(f"{model_alias}-{size}  ({method_label} projection)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(
        output_dir, f"physics_{model_alias}_{size}_{split}_{method}.png"
    )
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {fig_path}")
    return fig_path


def run_physics_experiment(
    model_alias: str,
    split: str = "test",
    max_samples: int | None = None,
    batch_size: int = 128,
    num_workers: int = 0,
    knn_k: int = 10,
    cv: int = 5,
    properties: list[str] | None = None,
    output_dir: str = "data",
    projection: str = "pca",
    pca_components: int | None = None,
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
        projection: Dimensionality reduction method for plots ('pca' or 'umap')

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

        os.makedirs(output_dir, exist_ok=True)
        emb_df = pl.DataFrame(
            {
                f"{model_alias}_{size}_galaxies": list(Z),
            }
        )
        parquet_path = os.path.join(
            output_dir, f"physics_{model_alias}_{size}_{split}.parquet"
        )
        emb_df.write_parquet(parquet_path)
        print(f"  Embeddings saved to {parquet_path}")

        # --- Build property arrays ---
        prop_arrays: dict[str, np.ndarray] = {}
        for key in property_keys:
            if metadata_accum[key]:
                arr = np.array(metadata_accum[key], dtype=np.float64)
                if key in ("sfr", "ssfr"):
                    arr[arr <= -99] = np.nan
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
            Z, prop_arrays, property_keys=list(prop_arrays.keys()), k=knn_k, cv=cv,
            pca_components=pca_components,
        )

        # Print results
        print(f"\n  {'Property':<25} {'Lin R²':<12} {'±std':<10} {'Neighbor':<12} {'Dist Corr':<12} {'MKNN prop':<12}")
        print(f"  {'-'*80}")
        for prop_key, metrics in size_results.items():
            if prop_key.startswith("_"):
                continue
            lr2 = metrics.get("linear_probe_r2", float("nan"))
            lr2_std = metrics.get("linear_probe_r2_std", float("nan"))
            print(f"  {prop_key:<25} {lr2:<12.4f} {lr2_std:<10.4f}")

        # Print mean R² summary
        summary = size_results.get("_summary", {})
        r2_mean = summary.get("r2_mean", float("nan"))
        r2_se = summary.get("r2_se", float("nan"))
        print(f"  {'-'*80}")
        print(f"  {'MEAN R²':<25} {r2_mean:<12.4f} ±{r2_se:<9.4f} (SE, {summary.get('n_properties', 0)} properties)")

        size_props = {}
        for k, v in size_results.items():
            if k.startswith("_"):
                continue
            prop_dict = {}
            for mk, mv in v.items():
                if isinstance(mv, list):
                    prop_dict[mk] = mv
                elif isinstance(mv, (int, float)) and np.isfinite(mv):
                    prop_dict[mk] = float(mv)
                else:
                    prop_dict[mk] = None
            size_props[k] = prop_dict

        all_results["sizes"][size] = {
            "n_samples": n_samples,
            "embedding_dim": Z.shape[1],
            "r2_mean": summary.get("r2_mean"),
            "r2_se": summary.get("r2_se"),
            "r2_std": summary.get("r2_std"),
            "r2_per_property": summary.get("r2_per_property"),
            "n_properties": summary.get("n_properties"),
            "properties": size_props,
        }

        # --- Generate visualisation
        if prop_arrays:
            plot_physics_embeddings(
                Z, prop_arrays, size_results,
                model_alias=model_alias,
                size=size,
                output_dir=output_dir,
                split=split,
                method=projection,
            )

    # --- Save results ---
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"physics_{model_alias}_{split}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return all_results


def rerun_physics_from_parquet(
    model_alias: str,
    split: str = "test",
    max_samples: int | None = None,
    knn_k: int = 10,
    cv: int = 5,
    properties: list[str] | None = None,
    input_dir: str = "data",
    output_dir: str = "data",
    pca_components: int | None = None,
) -> dict[str, Any]:
    """
    Re-run physics tests from saved parquet embeddings — no inference needed.

    Loads embeddings from parquet files written by ``run_physics_experiment``
    and re-streams the Smith42/galaxies dataset for property labels only.

    Args:
        model_alias: Model to test (e.g., 'vit', 'dino')
        split: Dataset split ('test' or 'validation')
        max_samples: Must match the value used during the original run
        knn_k: k for neighbour consistency metric
        cv: Cross-validation folds for linear probe
        properties: Which physical properties to test (None = defaults)
        input_dir: Directory containing the parquet files
        output_dir: Directory for result JSON files

    Returns:
        Nested dict with results per model size and property
    """
    if model_alias not in PHYSICS_MODEL_MAP:
        raise ValueError(
            f"Model '{model_alias}' not in PHYSICS_MODEL_MAP. "
            f"Available: {list(PHYSICS_MODEL_MAP.keys())}"
        )

    from datasets import load_dataset

    sizes, _ = PHYSICS_MODEL_MAP[model_alias]
    property_keys = properties or DEFAULT_PROPERTIES

    all_results: dict[str, Any] = {
        "model": model_alias,
        "split": split,
        "max_samples": max_samples,
        "property_keys": property_keys,
        "sizes": {},
    }

    for size in sizes:
        parquet_path = os.path.join(
            input_dir, f"physics_{model_alias}_{size}_{split}.parquet"
        )
        if not os.path.exists(parquet_path):
            print(f"  Skipping {model_alias}-{size}: {parquet_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Physics rerun: {model_alias} {size}")
        print(f"{'='*60}")

        # Load embeddings from parquet
        df = pl.read_parquet(parquet_path)
        emb_col = f"{model_alias}_{size}_galaxies"
        Z = np.stack(df[emb_col].to_list())
        n_samples = len(Z)
        print(f"  Loaded {n_samples} embeddings, shape: {Z.shape}")

        # Stream dataset for property labels (no inference)
        ds = load_dataset(
            "Smith42/galaxies", revision="v2.0", split=split, streaming=True,
        )
        if max_samples is not None:
            ds = ds.take(max_samples)

        prop_arrays: dict[str, list] = {key: [] for key in property_keys}
        for i, example in enumerate(tqdm(ds, total=n_samples, desc=f"Loading properties")):
            if i >= n_samples:
                break
            for key in property_keys:
                col_name = ALL_PROPERTIES.get(key, key)
                val = example.get(col_name)
                prop_arrays[key].append(float("nan") if val is None else val)

        # Convert to numpy and validate
        prop_np: dict[str, np.ndarray] = {}
        for key, vals in prop_arrays.items():
            if len(vals) == n_samples:
                arr = np.array(vals, dtype=np.float64)
                if key in ("sfr", "ssfr"):
                    arr[arr <= -99] = np.nan
                if np.any(np.isfinite(arr)):
                    prop_np[key] = arr
                else:
                    print(f"  Warning: skipping {key} (all NaN)")
            else:
                print(f"  Warning: skipping {key} (got {len(vals)} vs {n_samples} embeddings)")

        # Run physics tests
        print(f"\n  Running physics tests on {len(prop_np)} properties...")
        size_results = run_physics_tests(
            Z, prop_np, property_keys=list(prop_np.keys()), k=knn_k, cv=cv,
            pca_components=pca_components,
        )

        # Print results
        print(f"\n  {'Property':<25} {'Lin R²':<12} {'±std':<10}")
        print(f"  {'-'*50}")
        for prop_key, metrics in size_results.items():
            if prop_key.startswith("_"):
                continue
            lr2 = metrics.get("linear_probe_r2", float("nan"))
            lr2_std = metrics.get("linear_probe_r2_std", float("nan"))
            print(f"  {prop_key:<25} {lr2:<12.4f} {lr2_std:<10.4f}")

        summary = size_results.get("_summary", {})
        r2_mean = summary.get("r2_mean", float("nan"))
        r2_se = summary.get("r2_se", float("nan"))
        print(f"  {'-'*50}")
        print(f"  {'MEAN R²':<25} {r2_mean:<12.4f} ±{r2_se:<9.4f} (SE, {summary.get('n_properties', 0)} properties)")

        size_props = {}
        for k, v in size_results.items():
            if k.startswith("_"):
                continue
            prop_dict = {}
            for mk, mv in v.items():
                if isinstance(mv, list):
                    prop_dict[mk] = mv
                elif isinstance(mv, (int, float)) and np.isfinite(mv):
                    prop_dict[mk] = float(mv)
                else:
                    prop_dict[mk] = None
            size_props[k] = prop_dict

        all_results["sizes"][size] = {
            "n_samples": n_samples,
            "embedding_dim": Z.shape[1],
            "r2_mean": summary.get("r2_mean"),
            "r2_se": summary.get("r2_se"),
            "r2_std": summary.get("r2_std"),
            "r2_per_property": summary.get("r2_per_property"),
            "n_properties": summary.get("n_properties"),
            "properties": size_props,
        }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"physics_{model_alias}_{split}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return all_results
