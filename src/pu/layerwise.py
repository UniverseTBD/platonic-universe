"""Layer-wise CKA and MKNN analysis between spectral models."""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pu.models import get_adapter
from pu.pu_datasets import get_dataset_adapter
from pu.metrics import cka, mknn


# Model-specific preprocessor batch keys
_SPECTRA_KEY_MAP = {
    "specformer": ("spectra",),
    "specclip": ("spectra",),
    "aion": ("flux", "ivar", "mask", "wavelength"),
}

# Size -> model name mapping per alias
_SIZE_MAP = {
    "specformer": {"43M": "polymathic-ai/specformer"},
    "specclip": {"43M": "astroshawn/SpecCLIP"},
    "aion": {
        "base": "polymathic-ai/aion-base",
        "large": "polymathic-ai/aion-large",
        "xlarge": "polymathic-ai/aion-xlarge",
    },
}

_DEFAULT_SIZES = {
    "specformer": "43M",
    "specclip": "43M",
    "aion": "base",
}


def _resolve_model(alias, size=None):
    """Resolve alias + optional size to (size, model_name)."""
    if size is None:
        size = _DEFAULT_SIZES[alias]
    return size, _SIZE_MAP[alias][size]


class DualPreprocessor:
    """Wraps two preprocessors, prefixing output keys with a_/b_."""

    def __init__(self, preproc_a, preproc_b):
        self.preproc_a = preproc_a
        self.preproc_b = preproc_b

    def __call__(self, idx):
        result_a = self.preproc_a(idx)
        result_b = self.preproc_b(idx)
        out = {}
        for k, v in result_a.items():
            out[f"a_{k}"] = v
        for k, v in result_b.items():
            out[f"b_{k}"] = v
        return out


def _make_batch(raw_batch, prefix, keys):
    """Extract prefixed keys from a dataloader batch into a model batch."""
    return {k: raw_batch[f"{prefix}_{k}"] for k in keys}


def run_layerwise_analysis(
    model_a_alias,
    model_b_alias,
    comp_mode="desi",
    batch_size=128,
    num_workers=0,
    knn_k=10,
    max_samples=None,
    output_dir="data",
    size_a=None,
    size_b=None,
):
    """Compute layer-by-layer CKA and MKNN between two spectral models.

    Streams DESI spectra, extracts per-layer embeddings from both models,
    then computes (n_layers_a, n_layers_b) matrices of CKA and MKNN scores.
    Saves results as .npy files and heatmap PNGs.
    """
    hf_ds = f"Smith42/{comp_mode}_hsc_crossmatched"

    size_a, model_name_a = _resolve_model(model_a_alias, size_a)
    size_b, model_name_b = _resolve_model(model_b_alias, size_b)

    # Load both adapters
    adapter_a_cls = get_adapter(model_a_alias)
    adapter_a = adapter_a_cls(model_name_a, size_a, alias=model_a_alias)
    adapter_a.load()

    adapter_b_cls = get_adapter(model_b_alias)
    adapter_b = adapter_b_cls(model_name_b, size_b, alias=model_b_alias)
    adapter_b.load()

    # Build dual preprocessor
    modes = [comp_mode]
    preproc_a = adapter_a.get_preprocessor(modes)
    preproc_b = adapter_b.get_preprocessor(modes)
    dual_preproc = DualPreprocessor(preproc_a, preproc_b)

    # Load dataset
    ds_alias = f"{comp_mode}_spectra"
    dataset_adapter_cls = get_dataset_adapter(ds_alias)
    dataset_adapter = dataset_adapter_cls(hf_ds, comp_mode)
    dataset_adapter.load()
    ds = dataset_adapter.prepare(dual_preproc, modes, lambda idx: True)

    if max_samples is not None:
        ds = ds.take(max_samples)

    dl = iter(DataLoader(ds, batch_size=batch_size, num_workers=num_workers))

    keys_a = _SPECTRA_KEY_MAP[model_a_alias]
    keys_b = _SPECTRA_KEY_MAP[model_b_alias]

    # Collect per-layer embeddings
    all_layers_a = None
    all_layers_b = None

    with torch.no_grad():
        for batch in tqdm(dl, desc="Embedding"):
            batch_a = _make_batch(batch, "a", keys_a)
            batch_b = _make_batch(batch, "b", keys_b)

            layers_a = adapter_a.embed_layerwise(batch_a, comp_mode)
            layers_b = adapter_b.embed_layerwise(batch_b, comp_mode)

            if all_layers_a is None:
                all_layers_a = [[] for _ in layers_a]
                all_layers_b = [[] for _ in layers_b]

            for i, emb in enumerate(layers_a):
                all_layers_a[i].append(emb)
            for i, emb in enumerate(layers_b):
                all_layers_b[i].append(emb)

    # Concatenate across batches
    all_layers_a = [torch.cat(embs).numpy() for embs in all_layers_a]
    all_layers_b = [torch.cat(embs).numpy() for embs in all_layers_b]

    n_a = len(all_layers_a)
    n_b = len(all_layers_b)
    n_samples = all_layers_a[0].shape[0]
    print(f"\n{n_samples} samples, {n_a} layers ({model_a_alias}), {n_b} layers ({model_b_alias})")

    # Compute CKA and MKNN matrices
    cka_matrix = np.zeros((n_a, n_b))
    mknn_matrix = np.zeros((n_a, n_b))

    total = n_a * n_b
    with tqdm(total=total, desc="Computing metrics") as pbar:
        for i in range(n_a):
            for j in range(n_b):
                Za, Zb = all_layers_a[i], all_layers_b[j]
                if np.isnan(Za).any() or np.isnan(Zb).any():
                    cka_matrix[i, j] = np.nan
                    mknn_matrix[i, j] = np.nan
                else:
                    cka_matrix[i, j] = cka(Za, Zb)
                    mknn_matrix[i, j] = mknn(Za, Zb, k=knn_k)
                pbar.update(1)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    label_a = f"{model_a_alias}_{size_a}"
    label_b = f"{model_b_alias}_{size_b}"
    prefix = f"{label_a}_vs_{label_b}_{comp_mode}"

    np.save(os.path.join(output_dir, f"{prefix}_cka.npy"), cka_matrix)
    np.save(os.path.join(output_dir, f"{prefix}_mknn.npy"), mknn_matrix)

    # Plot heatmaps
    plot_layerwise_heatmap(
        cka_matrix, "CKA", label_a, label_b,
        os.path.join(output_dir, f"{prefix}_cka.png"),
    )
    plot_layerwise_heatmap(
        mknn_matrix, "MKNN", label_a, label_b,
        os.path.join(output_dir, f"{prefix}_mknn.png"),
    )

    print(f"\nSaved to {output_dir}/{prefix}_*.{{npy,png}}")
    return cka_matrix, mknn_matrix


def plot_layerwise_heatmap(scores, metric_name, model_a_name, model_b_name, output_path):
    """Plot a layer-by-layer metric heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(scores, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xlabel(f"{model_b_name} layer")
    ax.set_ylabel(f"{model_a_name} layer")
    ax.set_title(f"Layer-wise {metric_name}: {model_a_name} vs {model_b_name}")
    ax.set_xticks(range(scores.shape[1]))
    ax.set_yticks(range(scores.shape[0]))
    fig.colorbar(im, ax=ax, label=metric_name)

    # Annotate cells
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            ax.text(
                j, i, f"{scores[i, j]:.3f}",
                ha="center", va="center", fontsize=6,
                color="white" if scores[i, j] < (scores.max() + scores.min()) / 2 else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
