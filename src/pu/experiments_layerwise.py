"""
Layer-wise embedding extraction pipeline.

Extracts embeddings from ALL layers of each model, saves to parquet,
and optionally uploads to HuggingFace Hub.
"""

import os

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pu.models import get_adapter
from pu.pu_datasets import get_dataset_adapter

# Shared model map — same as experiments.py
MODEL_MAP = {
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
    "paligemma": (
        ["3b", "10b", "28b"],
        [
            "google/paligemma2-3b-mix-224",
            "google/paligemma2-10b-mix-224",
            "google/paligemma2-28b-mix-224",
        ],
    ),
    "paligemma_3b": (["3b"], ["google/paligemma2-3b-mix-224"]),
    "paligemma_10b": (["10b"], ["google/paligemma2-10b-mix-224"]),
    "paligemma_28b": (["28b"], ["google/paligemma2-28b-mix-224"]),
    "llava_15": (
        ["7b", "13b"],
        ["llava-hf/llava-1.5-7b-hf", "llava-hf/llava-1.5-13b-hf"],
    ),
    "llava_ov": (["7b"], ["llava-hf/llava-onevision-qwen2-7b-ov-hf"]),
}


def extract_all_layers(
    model_alias,
    mode,
    batch_size=64,
    num_workers=0,
    max_samples=None,
    resize=True,
    resize_mode="match",
    output_dir="data",
    hf_repo=None,
    hf_token=None,
    upload=True,
    delete_after_upload=False,
):
    """Extract embeddings from all layers of a model across a dataset.

    For each model size, extracts per-layer embeddings for the HSC baseline
    and (for JWST/LegacySurvey) the comparison mode. For DESI/SDSS the
    comparison embeddings are pre-computed and stored alongside.

    Saves one parquet per (model, size, dataset) to output_dir.
    Optionally uploads to HuggingFace Hub.
    """
    comp_mode = mode
    is_spectral_model = model_alias == "specformer"
    if is_spectral_model:
        print("[skip] SpecFormer does not support layerwise extraction")
        return

    modes = ["hsc", comp_mode]
    hf_ds = f"Smith42/{comp_mode}_hsc_crossmatched"

    def filterfun(idx):
        if "jwst" != comp_mode:
            return True
        im = idx["jwst_image"]["flux"][3]
        v0, v1 = np.nanpercentile(im, 5), np.nanpercentile(im, 99)
        return v0 - v1 != 0

    try:
        sizes, model_names = MODEL_MAP[model_alias]
    except KeyError:
        raise NotImplementedError(f"Model '{model_alias}' not in MODEL_MAP.")

    adapter_cls = get_adapter(model_alias)

    for size, model_name in zip(sizes, model_names):
        print(f"\n{'='*60}")
        print(f"[{model_alias} {size}] Loading model...")
        adapter = adapter_cls(model_name, size, alias=model_alias)
        adapter.load()

        if not adapter.supports_layerwise():
            print(f"[skip] {model_alias} {size} does not support layerwise extraction")
            continue

        num_layers = adapter.get_num_layers()
        print(f"[{model_alias} {size}] {num_layers} hookable modules, extracting on {comp_mode}...")

        processor = adapter.get_preprocessor(modes, resize=resize, resize_mode=resize_mode)

        ds_alias = comp_mode
        dataset_cls = get_dataset_adapter(ds_alias)
        ds_adapter = dataset_cls(hf_ds, comp_mode)
        ds_adapter.load()
        ds = ds_adapter.prepare(processor, modes, filterfun)

        if max_samples is not None:
            ds = ds.take(max_samples)

        dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

        # Accumulate per-layer embeddings for each mode
        # For modes where the model runs (HSC, JWST, LegacySurvey):
        #   layer_embs[mode] = {layer_idx: [tensor, ...]}
        # For pre-computed modes (DESI, SDSS):
        #   precomputed[mode] = [tensor, ...]
        layer_embs = {}
        precomputed = {}

        with torch.no_grad():
            for batch in tqdm(dl, desc=f"{model_alias} {size}"):
                for m in modes:
                    if m == "desi":
                        precomputed.setdefault(m, []).append(
                            torch.tensor(np.array(batch["embeddings"])).T
                        )
                    elif m == "sdss":
                        precomputed.setdefault(m, []).append(
                            torch.tensor(np.array(batch["embedding"])).T
                        )
                    else:
                        batch_layers = adapter.embed_all_layers_for_mode(batch, m)
                        if m not in layer_embs:
                            layer_embs[m] = {k: [] for k in batch_layers}
                        for k, emb in batch_layers.items():
                            layer_embs[m][k].append(emb.cpu())

        # Build polars DataFrame — use first batch's key order (DFS module order)
        columns = {}

        for m, layers_dict in layer_embs.items():
            for layer_key in layers_dict:
                col_name = f"{layer_key}_{m}"
                cat = torch.cat(layers_dict[layer_key]).numpy()
                columns[col_name] = cat

        for m, tensors in precomputed.items():
            col_name = f"{m}_embedding"
            cat = torch.cat(tensors).numpy()
            columns[col_name] = cat

        # Verify all columns have the same row count
        row_counts = {k: v.shape[0] for k, v in columns.items()}
        unique_counts = set(row_counts.values())
        if len(unique_counts) > 1:
            print(f"[ERROR] Row count mismatch: {row_counts}")
            continue

        n_samples = list(row_counts.values())[0] if row_counts else 0
        print(f"[{model_alias} {size}] {n_samples} samples, {len(columns)} columns")

        df = pl.DataFrame({
            k: pl.Series(k, v) for k, v in columns.items()
        })

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{comp_mode}_{model_alias}_{size}_layerwise.parquet")
        df.write_parquet(out_path)
        print(f"[{model_alias} {size}] Saved to {out_path}")

        # Upload to HuggingFace Hub and optionally delete local file
        if upload and hf_repo:
            from pu.hub import push_parquet
            push_parquet(out_path, hf_repo, token=hf_token)
            if delete_after_upload:
                os.remove(out_path)
                print(f"[{model_alias} {size}] Deleted local file after upload")

        # Free memory before next size
        del layer_embs, precomputed, columns, df
        del adapter
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
