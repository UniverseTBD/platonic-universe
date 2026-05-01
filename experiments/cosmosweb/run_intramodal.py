"""
CosmosWeb physics experiment — pu adapter/registry version.

Uses the pu adapter/registry system for both model loading and dataset streaming.

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
from PIL import Image
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

# Parameter counts in millions for each (model_alias, size).
# Sources: model cards / published papers.
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

DATASET          = ""
telescope        = "jwst"
N_USE            = 45000
BATCH_SIZE       = 16
OUT_DIR          = ""
n_neighbors      = 10
N_PROBE_RUNS     = 10
PROBE_TEST_SIZE  = 5000
upsample_images  = True   # bilinear-upsample all images to 224×224 before any model preprocessing

UPSAMPLE_SUFFIX  = "_upsampled" if upsample_images else ""
DS_TAG           = DATASET.split("/")[-1]

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
def _make_cosmosweb_preprocessor(adapter, telescope, upsample_images=True):
    """Return a dataset.map-compatible callable for cosmosweb PIL images.

    Output keys match embed_for_mode expectations per adapter type:
      - HF models  → {telescope: pixel_values_tensor}
      - SAM2       → {telescope: transformed_tensor}
      - AstroPT    → {telescope}_images and {telescope}_positions tensors

    If upsample_images is True, every PIL image is bilinear-resized to
    224×224 before being passed to the model-specific preprocessor.
    """
    image_col = f"{telescope}_images"
    _TARGET = (224, 224)

    def _maybe_upsample(img):
        if upsample_images:
            img = img.resize(_TARGET, Image.Resampling.BILINEAR)
        return img

    # ---- HF-style models (vit, clip, dino, dinov3, convnext, ijepa, vjepa, vit-mae, hiera) ----
    if hasattr(adapter, "processor") and adapter.processor is not None:
        proc = adapter.processor

        def _hf_wrapper(example):
            img = _maybe_upsample(example[image_col])
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
            img = _maybe_upsample(example[image_col])
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
            img = _maybe_upsample(example[image_col])
            arr = np.asarray(img, dtype=np.float32)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            arr = arr.swapaxes(0, 2)  # (H,W,C) -> (C,H,W)
            t = _torch.from_numpy(arr).to(_torch.float).unsqueeze(0)  # (1,C,H,W)
            # Only torch-interpolate when we haven't already upsampled via PIL
            if not upsample_images:
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

for model_alias, (sizes, model_names) in model_map.items():
    adapter_cls = get_adapter(model_alias)
    embeddings_all[model_alias] = {}
    neighbors_all[model_alias]  = {}

    for size, model_name in zip(sizes, model_names):
        print(f"\n{'='*60}")
        print(f"Model: {model_alias} {size}  ({model_name})")
        print(f"{'='*60}")

        cache_path = (
            f"{OUT_DIR}/embeddings/"
            f"{telescope}_embeddings_{DS_TAG}_{model_alias}_{size}_{n_valid}{UPSAMPLE_SUFFIX}.npy"
        )
        nn_cache_path = (
            f"{OUT_DIR}/embeddings/"
            f"{telescope}_nn{n_neighbors}_{DS_TAG}_{model_alias}_{size}_{n_valid}{UPSAMPLE_SUFFIX}.npy"
        )

        if os.path.exists(cache_path):
            print(f"  Loading cached embeddings from {cache_path}")
            embeddings = np.load(cache_path)
        else:
            # Load model via registry
            adapter = adapter_cls(model_name, size, alias=model_alias)
            adapter.load()

            # Build preprocessor and stream dataset via the dataset adapter
            proc_fn = _make_cosmosweb_preprocessor(adapter, telescope, upsample_images=upsample_images)

            dataset_adapter = get_dataset_adapter("cosmosweb")(DATASET, telescope)
            dataset_adapter.load()
            ds = dataset_adapter.prepare(
                processor=proc_fn,
                modes=[telescope],
                filterfun=lambda row: True,
                telescope=telescope,
                n_galaxies=N_USE,
                # AstroPT's preprocessor overwrites {telescope}_images in-place,
                # so we must NOT remove it after mapping.
                remove_image_col=(model_alias != "astropt")
            )
            if model_alias == "astropt":
                ds = ds.select_columns([f"{telescope}_images", f"{telescope}_positions"])
            else:
                ds = ds.select_columns([telescope])

            dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0)

            all_embs = []
            for batch in tqdm(dl, desc="  Embedding"):
                emb = adapter.embed_for_mode(batch, telescope).float().cpu()
                all_embs.append(emb.numpy())
                if sum(len(e) for e in all_embs) >= n_valid:
                    break

            embeddings = np.concatenate(all_embs, axis=0)[:n_valid]
            np.save(cache_path, embeddings)
            print(f"  Saved embeddings to {cache_path}")

            del adapter
            torch.cuda.empty_cache()

        print(f"  Embeddings shape: {embeddings.shape}")
        embeddings_all[model_alias][size] = embeddings

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
        neighbors_all[model_alias][size] = nn


# ── 4. Linear probes ──────────────────────────────────────────────────────────
import json

r2_path = f"{OUT_DIR}/{telescope}_linear_probe_r2_{n_valid}galaxies{UPSAMPLE_SUFFIX}.json"

def run_probe(embeddings, y, test_size, random_state):
    valid = np.isfinite(y)
    X_v, y_v = embeddings[valid], y[valid]
    lo, hi = np.quantile(y_v, [0.01, 0.99])
    y_clipped = np.clip(y_v, lo, hi)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_v, y_clipped, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    probe = LinearRegression().fit(X_tr, y_tr)
    y_pred = probe.predict(X_te)
    return dict(
        y_test=y_te, y_pred=y_pred,
        bias=float(np.mean(y_pred - y_te)),
        r2=r2_score(y_te, y_pred),
        lo=lo, hi=hi,
        n_train=len(X_tr), n_test=len(X_te),
    )


def run_probe_averaged(embeddings, y, test_size, n_runs):
    r2_list, last = [], None
    for seed in range(n_runs):
        pr = run_probe(embeddings, y, test_size=test_size, random_state=seed)
        r2_list.append(pr["r2"])
        last = pr
    last["r2_all"]  = r2_list
    last["r2_mean"] = float(np.mean(r2_list))
    last["r2_std"]  = float(np.std(r2_list))
    last["r2"]      = last["r2_mean"]
    return last


if os.path.exists(r2_path):
    print(f"Loading cached R² results from {r2_path}")
    with open(r2_path) as f:
        r2_summary = json.load(f)
    # Reconstruct results_probe in the shape expected by the plotting code
    results_probe = {}
    for model_alias, (sizes, _) in model_map.items():
        results_probe[model_alias] = {}
        for size in sizes:
            results_probe[model_alias][size] = {
                param_name: {
                    "r2_mean": r2_summary[model_alias][size][param_name]["r2_mean"],
                    "r2_std":  r2_summary[model_alias][size][param_name]["r2_std"],
                }
                for param_name in params
            }
else:
    results_probe = {}
    for model_alias, (sizes, _) in model_map.items():
        results_probe[model_alias] = {}
        for size in sizes:
            print(f"\n  Linear probes: {model_alias} {size}  ({N_PROBE_RUNS} runs each)...")
            embeddings = embeddings_all[model_alias][size]
            param_results = {}
            for param_name, param_vals in params.items():
                pr = run_probe_averaged(embeddings, param_vals,
                                        test_size=PROBE_TEST_SIZE, n_runs=N_PROBE_RUNS)
                param_results[param_name] = pr
                print(f"    {param_name:12s}  R²={pr['r2_mean']:.4f} ± {pr['r2_std']:.4f}")
            results_probe[model_alias][size] = param_results

    # ── Save linear probe R² results ──────────────────────────────────────────
    r2_summary = {}
    for model_alias, (sizes, _) in model_map.items():
        r2_summary[model_alias] = {}
        for size in sizes:
            r2_summary[model_alias][size] = {
                param_name: {
                    "r2_mean": results_probe[model_alias][size][param_name]["r2_mean"],
                    "r2_std":  results_probe[model_alias][size][param_name]["r2_std"],
                }
                for param_name in params
            }

    with open(r2_path, "w") as f:
        json.dump(r2_summary, f, indent=2)
    print(f"Saved R² results to {r2_path}")


# ── 5. Wasserstein + MKNN ─────────────────────────────────────────────────────
results_w_d  = {}
results_mknn = {}

for model_alias, (sizes, _) in model_map.items():
    results_w_d[model_alias]  = {}
    results_mknn[model_alias] = {}
    for m in range(len(sizes) - 1):
        key = f"{sizes[m]} vs {sizes[m+1]}"
        w_ds = {}
        for param_name, param_vals in params.items():
            scaler = StandardScaler()
            scaled = scaler.fit_transform(param_vals.reshape(-1, 1)).ravel()
            w_ds[param_name] = wass_distance(
                neighbors_all[model_alias][sizes[m]],
                neighbors_all[model_alias][sizes[m+1]],
                scaled,
            )
        results_w_d[model_alias][key] = w_ds
        results_mknn[model_alias][key] = mknn_neighbor_input(
            neighbors_all[model_alias][sizes[m]],
            neighbors_all[model_alias][sizes[m+1]],
        )


# ── 6. Plots ──────────────────────────────────────────────────────────────────
param_names = list(params.keys())

# Wasserstein scaling
pdf_path = f"{OUT_DIR}/{telescope}_wasserstein_scaling_{n_neighbors}neighbors{UPSAMPLE_SUFFIX}.pdf"
with PdfPages(pdf_path) as pdf:
    for param in param_names:
        fig, ax = plt.subplots(figsize=(7, 4))
        for model_alias, (sizes, _) in model_map.items():
            comparisons = [f"{sizes[m]} vs {sizes[m+1]}" for m in range(len(sizes) - 1)]
            y = [np.mean(results_w_d[model_alias][c][param]) for c in comparisons]
            ax.plot(range(len(comparisons)), y, marker="o", label=model_alias)
            ax.set_xticks(range(len(comparisons)))
            ax.set_xticklabels(comparisons, rotation=15, ha="right")
        ax.set_ylabel("Mean Wasserstein distance")
        ax.set_xlabel("Model size comparison")
        ax.set_title(f"Wasserstein scaling — {param}")
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        for model_alias, (sizes, _) in model_map.items():
            fig, ax = plt.subplots(figsize=(7, 4))
            comparisons = [f"{sizes[m]} vs {sizes[m+1]}" for m in range(len(sizes) - 1)]
            for c in comparisons:
                ax.hist(results_w_d[model_alias][c][param], histtype="step", bins=100, label=c)
                ax.axvline(np.mean(results_w_d[model_alias][c][param]), c="k", ls="--", label="mean")
                ax.axvline(np.median(results_w_d[model_alias][c][param]), c="m", ls="--", label="median")
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)

print(f"Saved Wasserstein plots to {pdf_path}")

# MKNN scaling
mknn_pdf_path = f"{OUT_DIR}/{telescope}_mknn_scaling_{n_neighbors}neighbors{UPSAMPLE_SUFFIX}.pdf"
with PdfPages(mknn_pdf_path) as pdf:
    fig, ax = plt.subplots(figsize=(7, 4))
    for model_alias, (sizes, _) in model_map.items():
        comparisons = [f"{sizes[m]} vs {sizes[m+1]}" for m in range(len(sizes) - 1)]
        y = [results_mknn[model_alias][c] for c in comparisons]
        ax.plot(range(len(comparisons)), y, marker="o", label=model_alias)
        ax.set_xticks(range(len(comparisons)))
        ax.set_xticklabels(comparisons, rotation=15, ha="right")
    ax.set_ylabel("MKNN overlap")
    ax.set_xlabel("Model size comparison")
    ax.set_title("MKNN scaling — adjacent model sizes")
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"Saved MKNN plots to {mknn_pdf_path}")

# Linear probe R² scaling — one plot per parameter, all models overlaid
probe_pdf_path = f"{OUT_DIR}/{telescope}_linear_probe_scaling_{n_neighbors}neighbors{UPSAMPLE_SUFFIX}.pdf"
model_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

with PdfPages(probe_pdf_path) as pdf:
    for param_name in params:
        fig, ax = plt.subplots(figsize=(9, 5))
        for color, (model_alias, (sizes, _)) in zip(model_colors, model_map.items()):
            # if model_alias == "dinov3":
            #     sizes = sizes[:-1]

            params_M = model_params_M.get(model_alias, {})
            sizes = [s for s in sizes if s in params_M]
            if not sizes:
                continue

            x = np.array([params_M[s] for s in sizes], dtype=float)
            r2_mean = np.array([results_probe[model_alias][s][param_name]["r2_mean"] for s in sizes])
            r2_std  = np.array([results_probe[model_alias][s][param_name]["r2_std"]  for s in sizes])

            ax.errorbar(
                x, r2_mean, yerr=r2_std,
                fmt="o-", color=color,
                lw=1.8, ms=6, capsize=4, capthick=1.2, elinewidth=1.2,
                label=model_alias,
            )
            # Annotate each point with its size label
            for xi, size, r2m in zip(x, sizes, r2_mean):
                ax.annotate(
                    size, (xi, r2m),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=6, color=color, alpha=0.8,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Number of parameters", fontsize=11)

        def _fmt_params(val, _):
            if val >= 1000:
                return f"{val/1000:.4g}B"
            return f"{val:.4g}M"
        ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_params))
        ax.set_ylabel("R²", fontsize=11)
        ax.set_title(
            f"Linear probe R² — {param_name}\n"
            f"{n_valid} {telescope.upper()} galaxies  (mean ± std, {N_PROBE_RUNS} runs)",
            fontsize=12,
        )
        ax.legend(fontsize=8, loc="best", ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print(f"Saved linear probe plots to {probe_pdf_path}")
