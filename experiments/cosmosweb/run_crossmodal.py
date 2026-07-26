"""
CosmosWeb physics experiment — pu adapter/registry version.

Uses the the pu adapter/registry system for both model loading and dataset streaming.

Pipeline:
  1. Catalog pass  — stream the raw dataset once (no model) to collect physical
                     parameters.
  2. Embedding     — for each model size, use get_dataset_adapter("cosmosweb")
                     + DataLoader to embed images, caching results to .npy.
  3. Analysis      — Wasserstein distance, MKNN overlap, linear probe R² scaling.
  4. Plots         — PDF output identical to the original script.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import login
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from pu.metrics.neighbors import mknn_neighbor_input
from pu.metrics.physics import wass_distance
from pu.models import get_adapter
from pu.pu_datasets import get_dataset_adapter
from pu.pu_datasets.cosmosweb import CATALOG_COLUMNS


# ── Config ─────────────────────────────────────────────────────────────────────
model_map = {
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
    "vit-mae": (
        ["base", "large", "huge"],
        [f"facebook/vit-mae-{s}" for s in ["base", "large", "huge"]],
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

model_params_M = {
    "vit":       {"base": 86,    "large": 307,   "huge": 632},
    "clip":      {"base": 86,    "large": 307},
    "dinov3":    {"vits16": 22,  "vits16plus": 26, "vitb16": 86,
                  "vitl16": 307, "vith16plus": 650, "vit7b16": 7000},
    "convnext":  {"nano": 16,    "tiny": 29,     "base": 89,     "large": 198},
    "ijepa":     {"huge": 632,   "giant": 1011},
    "vjepa":     {"large": 307,  "huge": 632,    "giant": 1011},
    "astropt":   {"015M": 15,    "095M": 95,     "850M": 850},
    "vit-mae":   {"base": 86,    "large": 307,   "huge": 632},
    "paligemma": {"3b": 3000,    "10b": 10000,   "28b": 28000},
    "llava_15":  {"7b": 7000,    "13b": 13000},
}

DATASET         = ""
telescopes      = ["hsc", "jwst"]
N_USE           = 45000
BATCH_SIZE      = 128
OUT_DIR         = ""
n_neighbors     = 10
N_PROBE_RUNS    = 10
PROBE_TEST_SIZE = 5000

DS_TAG          = DATASET.split("/")[-1]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/embeddings", exist_ok=True)


# ── 1. Catalog pass (no model) ─────────────────────────────────────────────────
# Stream the raw dataset once to collect physical parameters.  No image
# preprocessing happens here — we only read the catalog columns.
print(f"Loading catalog from {N_USE} galaxies in {DATASET}...")
ds_raw = load_dataset(DATASET, split="train", streaming=True)

params_raw = {k: [] for k in CATALOG_COLUMNS}
for row in tqdm(ds_raw, total=N_USE, desc="Catalog pass"):
    for param, col in CATALOG_COLUMNS.items():
        params_raw[param].append(row[col])
    if len(params_raw["redshift"]) >= N_USE:
        break

params = {k: np.array(v, dtype=np.float32) for k, v in params_raw.items()}
params["g-r"] = params["mag_g"] - params["mag_r"]
n_valid = len(params["redshift"])
print(f"Loaded {n_valid} galaxies.")


# ── 2. Preprocessor builder ────────────────────────────────────────────────────
def _make_cosmosweb_preprocessor(adapter, telescope):
    """Return a dataset.map-compatible callable for cosmosweb PIL images.

    Output keys match embed_for_mode expectations per adapter type:
      - HF models  → {telescope: pixel_values_tensor}
      - SAM2       → {telescope: transformed_tensor}
      - AstroPT    → {telescope}_images and {telescope}_positions tensors
    """
    image_col = f"{telescope}_images"

    # ---- HF-style models (vit, clip, dino, dinov3, convnext, ijepa, vjepa, vit-mae, hiera) ----
    if hasattr(adapter, "processor") and adapter.processor is not None:
        proc = adapter.processor

        def _hf_wrapper(example):
            img = example[image_col]
            if adapter.alias == "clip":
                proc_out = proc(images=img, return_tensors="pt")
            elif hasattr(adapter, "_PROMPTS"):
                prompt = adapter._PROMPTS.get(adapter.alias, "<image> ")
                proc_out = proc(text=prompt, images=img, return_tensors="pt")
            else:
                proc_out = proc(img, return_tensors="pt")
            if "pixel_values" in proc_out:
                return {telescope: proc_out["pixel_values"].squeeze(0)}
            if "pixel_values_videos" in proc_out:
                # vjepa outputs a video tensor; replicate single frame to 16
                return {
                    telescope: proc_out["pixel_values_videos"]
                    .repeat(1, 16, 1, 1, 1)
                    .squeeze(0)
                }
            raise KeyError("Processor output missing pixel_values")

        return _hf_wrapper

    # ---- SAM2 ----
    if adapter.alias == "sam2":
        sam2_transforms = adapter.predictor._transforms

        def _sam2_wrapper(example):
            img = example[image_col]
            arr = np.asarray(img)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            return {telescope: sam2_transforms(arr)}

        return _sam2_wrapper

    # ---- AstroPT ----
    if adapter.alias == "astropt":
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

        def _astropt_wrapper(example):
            img = example[image_col]
            arr = np.asarray(img, dtype=np.float32)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            arr = arr.swapaxes(0, 2)  # (H,W,C) -> (C,H,W)
            t = _torch.from_numpy(arr).to(_torch.float).unsqueeze(0)  # (1,C,H,W)
            t = _torch.nn.functional.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
            arr = t.squeeze(0).numpy()  # back to (C,H,W)
            im = galproc.process_galaxy(
                _torch.from_numpy(arr).to(_torch.float)  # arr already float (C,H,W)
            ).to(_torch.float)
            return {
                f"{telescope}_images": im,
                f"{telescope}_positions": _torch.arange(0, len(im), dtype=_torch.long),
            }

        return _astropt_wrapper

    raise ValueError(
        f"Cannot build cosmosweb preprocessor for '{adapter.alias}': "
        "adapter has no .processor and is not sam2 or astropt"
    )


# ── 3. Compute embeddings + neighbors ─────────────────────────────────────────
# For each model size we use the dataset adapter to stream and preprocess images,
# then call adapter.embed_for_mode — exactly as physics_experiment.py does.
#
# The filterfun passed to prepare() mirrors the valid_idx filter above so the
# resulting embeddings are row-aligned with the params arrays.

embeddings_all = {}
neighbors_all  = {}

for telescope in telescopes:
    embeddings_all[telescope] = {}
    neighbors_all[telescope] = {}
    for model_alias, (sizes, model_names) in model_map.items():
        adapter_cls = get_adapter(model_alias)
        embeddings_all[telescope][model_alias] = {}
        neighbors_all[telescope][model_alias]  = {}
    
        for size, model_name in zip(sizes, model_names):
            print(f"\n{'='*60}")
            print(f"Model: {model_alias} {size}  ({model_name})")
            print(f"{'='*60}")
    
            cache_path = (
                f"{OUT_DIR}/embeddings/"
                f"{telescope}_embeddings_{DS_TAG}_{model_alias}_{size}_{n_valid}_upsampled.npy"
            )
            nn_cache_path = (
                f"{OUT_DIR}/embeddings/"
                f"{telescope}_nn{n_neighbors}_{DS_TAG}_{model_alias}_{size}_{n_valid}_upsampled.npy"
            )

            print(f"  Loading cached embeddings from {cache_path}")
            embeddings = np.load(cache_path)

            print(f"  Embeddings shape: {embeddings.shape}")
            embeddings_all[telescope][model_alias][size] = embeddings

            if os.path.exists(nn_cache_path):
                print(f"  Loading cached NN matrix from {nn_cache_path}")
                nn = np.load(nn_cache_path)
            else:
                nn = (
                    NearestNeighbors(n_neighbors=n_neighbors, metric="minkowski")
                    .fit(embeddings)
                    .kneighbors(return_distance=False)
                )
                np.save(nn_cache_path, nn)
                print(f"  Saved NN matrix to {nn_cache_path}")
            neighbors_all[telescope][model_alias][size] = nn


# ── 4. Cross-modal Wasserstein + MKNN (HSC vs JWST, per size) ─────────────────
results_w_d  = {}
results_mknn = {}

for model_alias, (sizes, _) in model_map.items():
    results_w_d[model_alias]  = {}
    results_mknn[model_alias] = {}
    for size in sizes:
        w_ds = {}
        for param_name, param_vals in params.items():
            scaler = StandardScaler()
            scaled = scaler.fit_transform(param_vals.reshape(-1, 1)).ravel()
            w_ds[param_name] = wass_distance(
                neighbors_all["hsc"][model_alias][size],
                neighbors_all["jwst"][model_alias][size],
                scaled,
            )
        results_w_d[model_alias][size] = w_ds
        results_mknn[model_alias][size] = mknn_neighbor_input(
            neighbors_all["hsc"][model_alias][size],
            neighbors_all["jwst"][model_alias][size],
        )


# ── 5. Load linear-probe R² results (produced by cosmosweb_physics_experiment_pu.py) ──
import json

r2_hsc_path = f"{OUT_DIR}/hsc_linear_probe_r2_{n_valid}galaxies_upsampled.json"
r2_jwst_path = f"{OUT_DIR}/jwst_linear_probe_r2_{n_valid}galaxies_upsampled.json"

with open(r2_hsc_path) as f:
    r2_hsc = json.load(f)
print(f"Loaded HSC R² from {r2_hsc_path}")

with open(r2_jwst_path) as f:
    r2_jwst = json.load(f)
print(f"Loaded JWST R² from {r2_jwst_path}")


# ── 6. Plots ──────────────────────────────────────────────────────────────────
param_names  = list(params.keys())
model_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_map    = {alias: c for alias, c in zip(model_map.keys(), model_colors)}


def _fmt_params(val, _):
    if val >= 1000:
        return f"{val/1000:.4g}B"
    return f"{val:.4g}M"


# ── 6a. Wasserstein distance as a function of model size ──────────────────────
wass_pdf_path = f"{OUT_DIR}/cross_modal_wasserstein_vs_size_{n_neighbors}neighbors.pdf"
with PdfPages(wass_pdf_path) as pdf:
    for param in param_names:
        fig, ax = plt.subplots(figsize=(9, 5))
        for model_alias, (sizes, _) in model_map.items():
            params_M = model_params_M.get(model_alias, {})
            valid_sizes = [s for s in sizes if s in results_w_d[model_alias] and s in params_M]
            if not valid_sizes:
                continue
            x = np.array([params_M[s] for s in valid_sizes], dtype=float)
            y = [np.mean(results_w_d[model_alias][s][param]) for s in valid_sizes]
            ax.plot(
                x, y, marker="o", lw=1.8, ms=6,
                color=color_map[model_alias], label=model_alias,
            )
            for xi, size, yi in zip(x, valid_sizes, y):
                ax.annotate(
                    size, (xi, yi),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=6, color=color_map[model_alias], alpha=0.8,
                )
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_params))
        ax.set_xlabel("Number of parameters", fontsize=11)
        ax.set_ylabel("Mean Wasserstein distance (HSC vs JWST)", fontsize=11)
        ax.set_title(
            f"Cross-modal Wasserstein distance — {param}\n"
            f"{n_valid} galaxies, {n_neighbors} neighbors",
            fontsize=12,
        )
        ax.legend(fontsize=8, loc="best", ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print(f"Saved cross-modal Wasserstein plots to {wass_pdf_path}")


# ── 6b. MKNN overlap as a function of model size ──────────────────────────────
mknn_pdf_path = f"{OUT_DIR}/cross_modal_mknn_vs_size_{n_neighbors}neighbors.pdf"
with PdfPages(mknn_pdf_path) as pdf:
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_alias, (sizes, _) in model_map.items():
        params_M = model_params_M.get(model_alias, {})
        valid_sizes = [s for s in sizes if s in results_mknn[model_alias] and s in params_M]
        if not valid_sizes:
            continue
        x = np.array([params_M[s] for s in valid_sizes], dtype=float)
        y = [results_mknn[model_alias][s] for s in valid_sizes]
        ax.plot(
            x, y, marker="o", lw=1.8, ms=6,
            color=color_map[model_alias], label=model_alias,
        )
        for xi, size, yi in zip(x, valid_sizes, y):
            ax.annotate(
                size, (xi, yi),
                textcoords="offset points", xytext=(0, 6),
                ha="center", fontsize=6, color=color_map[model_alias], alpha=0.8,
            )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_params))
    ax.set_xlabel("Number of parameters", fontsize=11)
    ax.set_ylabel("MKNN overlap (HSC vs JWST)", fontsize=11)
    ax.set_title(
        f"Cross-modal MKNN overlap vs model size\n"
        f"{n_valid} galaxies, {n_neighbors} neighbors",
        fontsize=12,
    )
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"Saved cross-modal MKNN vs size plots to {mknn_pdf_path}")


# ── 6c. MKNN overlap vs linear-probe R² ──────────────────────────────────────
# One scatter per physical parameter; each point = (model_alias, size).
# Two versions: R² from HSC embeddings, and R² from JWST embeddings.

def _plot_mknn_vs_r2(r2_dict, telescope_label, pdf_path):
    with PdfPages(pdf_path) as pdf:
        for param in param_names:
            fig, ax = plt.subplots(figsize=(8, 5))
            for model_alias, (sizes, _) in model_map.items():
                xs, ys, labels = [], [], []
                for size in sizes:
                    r2_entry = r2_dict.get(model_alias, {}).get(size, {}).get(param)
                    if r2_entry is None:
                        continue
                    r2_mean = r2_entry["r2_mean"]
                    r2_std  = r2_entry["r2_std"]
                    mknn    = results_mknn[model_alias][size]
                    xs.append(r2_mean)
                    ys.append(mknn)
                    labels.append(size)
                    ax.errorbar(
                        r2_mean, mknn,
                        xerr=r2_std,
                        fmt="o", color=color_map[model_alias],
                        ms=7, capsize=3, capthick=1, elinewidth=1,
                    )
                    ax.annotate(
                        f"{model_alias}\n{size}", (r2_mean, mknn),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=5, color=color_map[model_alias], alpha=0.85,
                    )
                if xs:
                    # Connect sizes within the same model family with a line
                    order = np.argsort(xs)
                    ax.plot(
                        [xs[i] for i in order], [ys[i] for i in order],
                        lw=1, ls="--", color=color_map[model_alias],
                        alpha=0.5, label=model_alias,
                    )
            #ax.set_xlim(0,1)
            ax.set_xlabel(f"Linear probe R² ({telescope_label})", fontsize=11)
            ax.set_ylabel("MKNN overlap (HSC vs JWST)", fontsize=11)
            ax.set_title(
                f"Cross-modal MKNN vs R² ({telescope_label}) — {param}\n"
                f"{n_valid} galaxies, {n_neighbors} neighbors",
                fontsize=12,
            )
            ax.legend(fontsize=8, loc="best", ncol=2)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved MKNN vs R² ({telescope_label}) plots to {pdf_path}")


_plot_mknn_vs_r2(
    r2_hsc,
    "HSC",
    f"{OUT_DIR}/cross_modal_mknn_vs_r2_hsc_{n_neighbors}neighbors.pdf",
)

_plot_mknn_vs_r2(
    r2_jwst,
    "JWST",
    f"{OUT_DIR}/cross_modal_mknn_vs_r2_jwst_{n_neighbors}neighbors.pdf",
)
