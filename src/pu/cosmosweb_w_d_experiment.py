import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm
from pu.metrics.physics import wass_distance
from pu.metrics.neighbors import mknn_neighbor_input
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

model_map = {
    "convnext": (
        ["nano", "tiny", "base", "large"],
        [
            "facebook/convnextv2-nano-22k-224",
            "facebook/convnextv2-tiny-22k-224",
            "facebook/convnextv2-base-22k-224",
            "facebook/convnextv2-large-22k-224",
        ],
    ),
    "dinov2": (
        ["small", "base", "large", "giant"],
        [
            "facebook/dinov2-small",
            "facebook/dinov2-base",
            "facebook/dinov2-large",
            "facebook/dinov2-giant",
        ],
    ) 
}

N_GALAXIES = 45000
DATASET    = "Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2"
BATCH_SIZE = 128
OUT_DIR    = "/pscratch/sd/a/ashodkh/platonic_universe"
N_USE = 44000 # use a smaller number of galaxies to make testing/debugging quicker
n_neighbors = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/embeddings", exist_ok=True)

# ── 1. Load dataset ────────────────────────────────────────────────────────────
print(f"Loading {N_GALAXIES} galaxies from {DATASET}...")
ds = load_dataset(DATASET, split="train", streaming=True)

catalog_names = {
    "redshift": "lephare_photozs",
    "mag_g": "mag_model_hsc-g",
    "mag_r": "mag_model_hsc-r",
    "mass": "lp_mass",
    "sSFR": "lp_ssfr"
}

images   = []
params = {
    "redshift": [],
    "mag_g": [],
    "mag_r": [],
    "mass": [],
    "sSFR": []
}

for row in tqdm(ds, total=N_GALAXIES, desc="Fetching"):
    images.append(row["hsc_images"])
    for param in params:
        params[param].append(row[catalog_names[param]])
    if len(images) >= N_GALAXIES:
        break

for param in params:
    params[param] = np.array(params[param], dtype=np.float32)

# Keep only positive redshifts, clipped to [0, 4], and keep only N_USE data points
valid_idx = np.where(params["redshift"] > 0)[0][:N_USE]
images    = [images[i] for i in valid_idx]
for param in params:
    if param == "redshift":
        params["redshift"] = np.clip(params["redshift"][valid_idx], 0, 4)
    else:
        params[param] = params[param][valid_idx]

params["g-r"] = params["mag_g"] - params["mag_r"]

print(f"After filtering: {len(images)} galaxies.  z range: [{params['redshift'].min():.3f}, {params['redshift'].max():.3f}]")

# physics space matrix
physics_matrix = np.zeros((len(images), len(params.keys())), dtype=np.float32)
for i, param in enumerate(params):
    scaler = StandardScaler()
    physics_matrix[:,i] = scaler.fit_transform(params[param].reshape(-1, 1)).ravel()

physics_neighbors = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="minkowski"
    ).fit(physics_matrix).kneighbors(return_distance=False)

# ── 2. Embedding function ──────────────────────────────────────────────────────
def compute_embeddings(images, model_name, model_alias, batch_size, device):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(device).eval()
    all_embs  = []
    for i in tqdm(range(0, len(images), batch_size), desc="  Embedding"):
        batch  = images[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            if model_alias == "dinov2":
                emb = model(inputs).last_hidden_state[:, 0, :].float()
            elif model_alias == "convnext":
                feats = model(inputs).last_hidden_state   # (B, C, H, W)
                emb   = feats.mean(dim=(2, 3)).float()    # spatial mean → (B, C)
        all_embs.append(emb.cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(all_embs, axis=0)


def compute_neighbors(z, n_neighbors):
    return NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="minkowski"
    ).fit(z).kneighbors(return_distance=False)

def run_experiment(nn1, nn2, params):
    w_ds = {}
    for param in params:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(params[param].reshape(-1, 1)).ravel()
        w_ds[param] = wass_distance(nn1, nn2, scaled)

    return w_ds

# compute all embeddings

embeddings_all = {}
neighbors_all = {}
for model_alias in model_map:
    embeddings_all[model_alias] = {}
    neighbors_all[model_alias] = {}
    for size, model_name in zip(*model_map[model_alias]):
        print(f"\n{'='*60}")
        print(f"Model: {model_alias} {size.capitalize()}  ({model_name})")
        print(f"{'='*60}")
        ds_tag = DATASET.split("/")[-1]
        cache_path = f"{OUT_DIR}/embeddings/embeddings_{ds_tag}_{model_alias}_{size}_{N_GALAXIES}.npy"
        if os.path.exists(cache_path):
            print(f"  Loading cached embeddings from {cache_path}")
            embeddings = np.load(cache_path)
            if len(embeddings) != len(images):
                embeddings = embeddings[valid_idx]
        else:
            embeddings = compute_embeddings(images, model_name, model_alias, BATCH_SIZE, DEVICE)
            np.save(cache_path, embeddings)
            print(f"  Saved embeddings to {cache_path}")
        print(f"  Embeddings shape: {embeddings.shape}")
        embeddings_all[model_alias][size] = embeddings
        neighbors_all[model_alias][size] = compute_neighbors(embeddings, n_neighbors)

# calculate results
results_w_d = {}
results_mknn = {}
results_mknn_to_physical = {}

for model_alias in model_map:
    results_w_d[model_alias] = {}
    results_mknn[model_alias] = {}
    results_mknn_to_physical[model_alias] = {}
    sizes = model_map[model_alias][0]
    for m in range(len(sizes)-1):
        results_w_d[model_alias][f"{sizes[m]} vs {sizes[m+1]}"] = \
            run_experiment(
                neighbors_all[model_alias][sizes[m]],
                neighbors_all[model_alias][sizes[m+1]],
                params=params,
            )
        results_mknn[model_alias][f"{sizes[m]} vs {sizes[m+1]}"] = \
            mknn_neighbor_input(
                neighbors_all[model_alias][sizes[m]],
                neighbors_all[model_alias][sizes[m+1]],
            )
    for m in range(len(sizes)):
        results_mknn_to_physical[model_alias][sizes[m]] = \
            mknn_neighbor_input(
                neighbors_all[model_alias][sizes[m]],
                physics_neighbors,
            )

# ── 4. Plotting ────────────────────────────────────────────────────────────────
pdf_path = f"{OUT_DIR}/wasserstein_scaling_{n_neighbors}neighbors.pdf"
param_names = list(params.keys())

with PdfPages(pdf_path) as pdf:
    for param in param_names:
        fig, ax = plt.subplots(figsize=(7, 4))
        for model_alias in model_map:
            sizes = model_map[model_alias][0]
            comparisons = [f"{sizes[m]} vs {sizes[m+1]}" for m in range(len(sizes) - 1)]
            y = [np.mean(results_w_d[model_alias][c][param]) for c in comparisons]
            x = range(len(comparisons))
            ax.plot(x, y, marker="o", label=model_alias)
            ax.set_xticks(list(x))
            ax.set_xticklabels(comparisons, rotation=15, ha="right")
        ax.set_ylabel("Mean Wasserstein distance")
        ax.set_xlabel("Model size comparison")
        ax.set_title(f"Wasserstein scaling — {param}")
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        for model_alias in model_map:
            fig, ax = plt.subplots(figsize=(7,4))
            sizes = model_map[model_alias][0]
            comparisons = [f"{sizes[m]} vs {sizes[m+1]}" for m in range(len(sizes) - 1)]
            for c in comparisons:
                ax.hist(results_w_d[model_alias][c][param], histtype='step', bins=100, label=c)
                ax.axvline(np.mean(results_w_d[model_alias][c][param]), label='mean', c='k', ls='--')
                ax.axvline(np.median(results_w_d[model_alias][c][param]), label='median', c='m', ls='--')
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)
print(f"Saved plots to {pdf_path}")

# ── 5. MKNN scaling plot ───────────────────────────────────────────────────────
mknn_pdf_path = f"{OUT_DIR}/mknn_scaling_{n_neighbors}neighbors.pdf"

with PdfPages(mknn_pdf_path) as pdf:
    fig, ax = plt.subplots(figsize=(7, 4))
    for model_alias in model_map:
        sizes = model_map[model_alias][0]
        comparisons = [f"{sizes[m]} vs {sizes[m+1]}" for m in range(len(sizes) - 1)]
        y = [results_mknn[model_alias][c] for c in comparisons]
        x = range(len(comparisons))
        ax.plot(x, y, marker="o", label=model_alias)
        ax.set_xticks(list(x))
        ax.set_xticklabels(comparisons, rotation=15, ha="right")
    ax.set_ylabel("MKNN overlap")
    ax.set_xlabel("Model size comparison")
    ax.set_title("MKNN scaling — adjacent model sizes")
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"Saved MKNN plots to {mknn_pdf_path}")

# ── 6. MKNN-to-physical scaling plot ──────────────────────────────────────────
mknn_phys_pdf_path = f"{OUT_DIR}/mknn_to_physical_{n_neighbors}neighbors.pdf"

with PdfPages(mknn_phys_pdf_path) as pdf:
    fig, ax = plt.subplots(figsize=(7, 4))
    for model_alias in model_map:
        sizes = model_map[model_alias][0]
        y = [results_mknn_to_physical[model_alias][s] for s in sizes]
        x = range(len(sizes))
        ax.plot(x, y, marker="o", label=model_alias)
        ax.set_xticks(list(x))
        ax.set_xticklabels(sizes, rotation=15, ha="right")
    ax.set_ylabel("MKNN overlap with physical space")
    ax.set_xlabel("Model size")
    ax.set_title("MKNN to physical space — model size scaling")
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"Saved MKNN-to-physical plots to {mknn_phys_pdf_path}")

