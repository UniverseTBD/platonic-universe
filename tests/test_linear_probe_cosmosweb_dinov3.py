"""
DINOv2 (vits14 → vitg14) embeddings of HSC galaxy images from the CosmosWeb dataset.

  - Generates / loads cached embeddings for each model size
  - Linear probe (OLS) on train / test split, with standardized embeddings
  - R² scaling plot for redshift, g-r color, stellar mass, and sSFR

Usage:
    python tests/test_linear_probe_cosmosweb_dinov3.py

Key variables:
    N_GALAXIES  - number of galaxies to use (default: 45000)
    OUT_DIR     - directory for all outputs (embeddings, plots)
    MODELS      - dict of {size_label: HuggingFace model ID}
    BATCH_SIZE  - inference batch size
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ── Configuration ─────────────────────────────────────────────────────────────
N_GALAXIES = 45000
DATASET    = "Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2"
BATCH_SIZE = 128
OUT_DIR    = "/pscratch/sd/a/ashodkh/platonic_universe"
MODELS     = {
    "small": "facebook/dinov2-small",
    "base": "facebook/dinov2-base",
    "large": "facebook/dinov2-large",
    "giant": "facebook/dinov2-giant",
}
N_PROBE_RUNS = 10   # number of random train/test splits for averaging
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/embeddings", exist_ok=True)


# ── 1. Load dataset ────────────────────────────────────────────────────────────
print(f"Loading {N_GALAXIES} galaxies from {DATASET}...")
ds = load_dataset(DATASET, split="train", streaming=True)

images   = []
redshift = []
mag_g    = []
mag_r    = []
lp_mass  = []
lp_ssfr  = []

for row in tqdm(ds, total=N_GALAXIES, desc="Fetching"):
    images.append(row["hsc_images"])
    redshift.append(row["lephare_photozs"])
    mag_g.append(row["mag_model_hsc-g"])
    mag_r.append(row["mag_model_hsc-r"])
    lp_mass.append(row["lp_mass"])
    lp_ssfr.append(row["lp_ssfr"])
    if len(images) >= N_GALAXIES:
        break

redshift = np.array(redshift, dtype=np.float32)
mag_g    = np.array(mag_g,    dtype=np.float32)
mag_r    = np.array(mag_r,    dtype=np.float32)
lp_mass  = np.array(lp_mass,  dtype=np.float32)
lp_ssfr  = np.array(lp_ssfr,  dtype=np.float32)

# Keep only positive redshifts, clipped to [0, 4]
valid_idx = np.where(redshift > 0)[0]
images    = [images[i] for i in valid_idx]
redshift  = np.clip(redshift[valid_idx], 0, 4)
mag_g     = mag_g[valid_idx]
mag_r     = mag_r[valid_idx]
lp_mass   = lp_mass[valid_idx]
lp_ssfr   = lp_ssfr[valid_idx]
gr_color  = mag_g - mag_r

print(f"After filtering: {len(images)} galaxies.  z range: [{redshift.min():.3f}, {redshift.max():.3f}]")

PHYS_PARAMS = {
    "redshift":  redshift,
    "g-r color": gr_color,
    "mag g":     mag_g,
    "mag r":     mag_r,
    "log mass":  lp_mass,
    "log sSFR":  lp_ssfr,
}


# ── 2. Embedding function ──────────────────────────────────────────────────────
def compute_embeddings(images, model_name, batch_size, device):
    """Extract DINOv3 CLS token embeddings."""
    processor = AutoImageProcessor.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(device).eval()
    all_embs  = []
    for i in tqdm(range(0, len(images), batch_size), desc="  Embedding"):
        batch  = images[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            # DINOv3: CLS token is index 0
            emb = model(inputs).last_hidden_state[:, 0, :].float()
        all_embs.append(emb.cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(all_embs, axis=0)


# ── 3. Probe helpers ───────────────────────────────────────────────────────────
def run_probe(embeddings, y, test_size=2000, random_state=42):
    """Linear regression probe; handles NaN by masking, clips 1–99 percentile."""
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
    probe = LinearRegression()
    probe.fit(X_tr, y_tr)
    y_pred = probe.predict(X_te)
    bias = float(np.mean(y_pred - y_te))
    r2   = r2_score(y_te, y_pred)
    return dict(y_test=y_te, y_pred=y_pred, bias=bias, r2=r2, lo=lo, hi=hi,
                n_train=len(X_tr), n_test=len(X_te))


def run_probe_averaged(embeddings, y, test_size=2000, n_runs=N_PROBE_RUNS):
    """Run probe n_runs times with different seeds; return mean/std R²."""
    r2_list = []
    last = None
    for seed in range(n_runs):
        pr = run_probe(embeddings, y, test_size=test_size, random_state=seed)
        r2_list.append(pr["r2"])
        last = pr
    last["r2_all"]  = r2_list
    last["r2_mean"] = float(np.mean(r2_list))
    last["r2_std"]  = float(np.std(r2_list))
    last["r2"]      = last["r2_mean"]
    return last


# ── 4. Compute embeddings & probes for all model sizes ────────────────────────
results = {}
ds_tag  = DATASET.split("/")[-1]

for size, model_name in MODELS.items():
    print(f"\n{'='*60}")
    print(f"Model: DINOv2 {size}  ({model_name})")
    print(f"{'='*60}")

    cache_path = f"{OUT_DIR}/embeddings/embeddings_{ds_tag}_dinov2_{size}_{N_GALAXIES}.npy"
    if os.path.exists(cache_path):
        print(f"  Loading cached embeddings from {cache_path}")
        embeddings = np.load(cache_path)
        if len(embeddings) != len(images):
            embeddings = embeddings[valid_idx]
    else:
        embeddings = compute_embeddings(images, model_name, BATCH_SIZE, DEVICE)
        np.save(cache_path, embeddings)
        print(f"  Saved embeddings to {cache_path}")
    print(f"  Embeddings shape: {embeddings.shape}")

    print(f"  Running probes ({N_PROBE_RUNS} runs each)...")
    param_results = {}
    for param_name, param_vals in PHYS_PARAMS.items():
        pr = run_probe_averaged(embeddings, param_vals, test_size=5000)
        param_results[param_name] = pr
        print(f"    {param_name:12s}  R²={pr['r2_mean']:.4f} ± {pr['r2_std']:.4f}")

    results[size] = dict(embeddings=embeddings, params=param_results)

sizes = list(results.keys())


# ── 5. Redshift linear probe figure (pred vs true + residuals) ────────────────
fig, axes = plt.subplots(2, len(sizes), figsize=(7 * len(sizes), 11))

for col, size in enumerate(sizes):
    pr        = results[size]["params"]["redshift"]
    z_test    = pr["y_test"]
    z_pred    = pr["y_pred"]
    bias      = pr["bias"]
    r2        = pr["r2"]
    residuals = z_pred - z_test

    ax = axes[0, col]
    sc = ax.scatter(z_test, z_pred, c=z_test, s=15, cmap="inferno",
                    vmin=0, vmax=4, alpha=0.7, linewidths=0)
    fig.colorbar(sc, ax=ax, pad=0.02, label="True redshift")
    lo_plot = min(z_test.min(), z_pred.min())
    hi_plot = max(z_test.max(), z_pred.max())
    ax.plot([lo_plot, hi_plot], [lo_plot, hi_plot], "w--", lw=1, label="1:1")
    ax.set_xlabel("True redshift $z$", fontsize=11)
    ax.set_ylabel("Predicted redshift $\\hat{z}$", fontsize=11)
    ax.set_title(
        f"DINOv2 {size}\n"
        f"R²={r2:.3f}  bias={bias:+.4f}  "
        f"(z clipped to [{pr['lo']:.3f}, {pr['hi']:.3f}])",
        fontsize=10,
    )
    ax.legend(fontsize=9)

    ax = axes[1, col]
    ax.hist(residuals, bins=50, color="steelblue", edgecolor="none", alpha=0.85)
    ax.axvline(0,    color="white",  lw=1.5, linestyle="--", label="zero")
    ax.axvline(bias, color="tomato", lw=1.5, linestyle="-",
               label=f"mean bias = {bias:+.4f}")
    ax.set_xlabel("Residual $\\hat{z} - z$", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Residual distribution", fontsize=10)
    ax.legend(fontsize=9)

n_train = results[sizes[0]]["params"]["redshift"]["n_train"]
n_test  = results[sizes[0]]["params"]["redshift"]["n_test"]
fig.suptitle(
    f"Redshift linear probe — {N_GALAXIES} HSC galaxies  "
    f"({n_train} train / {n_test} test)",
    fontsize=13,
)
plt.tight_layout()
probe_path = f"{OUT_DIR}/linear_probe_cosmosweb_dinov2_{'_'.join(sizes)}_{N_GALAXIES}.png"
fig.savefig(probe_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved linear probe figure to {probe_path}")


# ── 6. R² vs model size — one page per physical parameter in a single PDF ─────
param_names   = list(PHYS_PARAMS.keys())
x             = np.arange(len(sizes))
N_INDIVIDUAL  = min(5, N_PROBE_RUNS)
indiv_colors  = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3"]

scaling_path = f"{OUT_DIR}/scaling_linear_cosmosweb_dinov2_{N_GALAXIES}.pdf"
with PdfPages(scaling_path) as pdf:
    for param_name in param_names:
        r2_mean = [results[s]["params"][param_name]["r2_mean"] for s in sizes]
        r2_std  = [results[s]["params"][param_name]["r2_std"]  for s in sizes]
        r2_all  = [results[s]["params"][param_name]["r2_all"]  for s in sizes]

        run_slopes, run_intercepts = [], []
        for run_i in range(N_PROBE_RUNS):
            y_vals = np.array([r2_all[si][run_i] for si in range(len(sizes))])
            m, b = np.polyfit(x, y_vals, 1)
            run_slopes.append(m)
            run_intercepts.append(b)
        mean_slope     = float(np.mean(run_slopes))
        std_slope      = float(np.std(run_slopes))
        mean_intercept = float(np.mean(run_intercepts))
        std_intercept  = float(np.std(run_intercepts))

        fig, ax = plt.subplots(figsize=(7, 4))

        for run_i in range(N_INDIVIDUAL):
            y_vals = [r2_all[si][run_i] for si in range(len(sizes))]
            ax.plot(x, y_vals, "o--", color=indiv_colors[run_i],
                    alpha=0.55, lw=1, ms=5, label=f"run {run_i}")

        ax.errorbar(x, r2_mean, yerr=r2_std, fmt="o-", color="gray",
                    lw=2.5, ms=9, capsize=5, capthick=1.5, elinewidth=1.5,
                    zorder=5, label="mean ± std")

        x_fit = np.array([x[0] - 0.1, x[-1] + 0.1])
        ax.plot(x_fit, mean_slope * x_fit + mean_intercept, "k-", lw=1.5, zorder=6,
                label=f"fit: slope={mean_slope:.4f}±{std_slope:.4f}\n"
                      f"     intercept={mean_intercept:.4f}±{std_intercept:.4f}")

        ax.set_xticks(x)
        ax.set_xticklabels(sizes, fontsize=9, rotation=15)
        ax.set_xlabel("Model size", fontsize=11)
        ax.set_ylabel("R²", fontsize=11)
        ax.set_title(
            f"Linear probe R² — {param_name}\n"
            f"{N_GALAXIES} HSC galaxies  (mean ± std, {N_PROBE_RUNS} runs)",
            fontsize=12,
        )
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print(f"Saved scaling figures to {scaling_path}")
print("\nDone.")
