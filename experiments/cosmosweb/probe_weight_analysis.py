"""
Probe weight analysis: cosine similarity within models, Procrustes disparity across pairs.

For each telescope × model × size:
  - Fits one linear probe per property (redshift, mass, sSFR); seed=42, 5000 test objects.
  - Computes the 3×3 cosine-similarity matrix of the probe weight vectors.

Cross-model (per telescope):
  - For each property, computes pairwise Procrustes disparity of test predictions
    across all model pairs, then plots a histogram.
"""

import os
import pickle
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.spatial import procrustes as scipy_procrustes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from pu.pu_datasets.cosmosweb import CATALOG_COLUMNS

# ── Config ──────────────────────────────────────────────────────────────────────
DATASET         = ""
OUT_DIR         = ""
EMB_DIR         = f"{OUT_DIR}/embeddings"
DS_TAG          = DATASET.split("/")[-1]
N_USE           = 45000
UPSAMPLE_SUFFIX = "_upsampled"
PROBE_TEST_SIZE = 5000
SEED            = 42
TELESCOPES      = ["hsc", "jwst"]
TARGET_PROPS    = ["redshift", "mass", "sSFR"]

model_map = {
    "vit":       ["base", "large", "huge"],
    "clip":      ["base", "large"],
    "dinov3":    ["vits16", "vits16plus", "vitb16", "vitl16", "vith16plus", "vit7b16"],
    "convnext":  ["nano", "tiny", "base", "large"],
    "ijepa":     ["huge", "giant"],
    "vjepa":     ["large", "huge", "giant"],
    "astropt":   ["015M", "095M", "850M"],
    "vit-mae":   ["base", "large", "huge"],
    "paligemma": ["3b", "10b", "28b"],
    "llava_15":  ["7b", "13b"],
}


# ── 1. Catalog ──────────────────────────────────────────────────────────────────
print(f"Streaming catalog from {DATASET} ({N_USE} galaxies)…")
ds_raw = load_dataset(DATASET, split="train", streaming=True)

raw = {k: [] for k in CATALOG_COLUMNS}
for row in tqdm(ds_raw, total=N_USE, desc="Catalog"):
    for param, col in CATALOG_COLUMNS.items():
        raw[param].append(row[col])
    if len(raw["redshift"]) >= N_USE:
        break

all_params = {k: np.array(v, dtype=np.float32) for k, v in raw.items()}
n_valid = len(all_params["redshift"])
print(f"Loaded {n_valid} galaxies.")


# ── 2. Clip targets & build consistent train/test split ─────────────────────────
props_clipped = {}
for prop in TARGET_PROPS:
    v = all_params[prop]
    finite = np.isfinite(v)
    lo, hi = np.quantile(v[finite], [0.01, 0.99])
    vc = np.clip(v, lo, hi).astype(np.float32)
    vc[~finite] = np.nan
    props_clipped[prop] = vc

all_valid = np.stack(
    [np.isfinite(props_clipped[p]) for p in TARGET_PROPS], axis=1
).all(axis=1)
valid_indices = np.where(all_valid)[0]
print(f"Rows with all 3 props finite: {len(valid_indices)}")

idx_train, idx_test = train_test_split(
    valid_indices, test_size=PROBE_TEST_SIZE, random_state=SEED
)
y_test = {prop: props_clipped[prop][idx_test] for prop in TARGET_PROPS}


# ── 3. Helpers ──────────────────────────────────────────────────────────────────
def emb_path(telescope, alias, size):
    return (
        f"{EMB_DIR}/{telescope}_embeddings_{DS_TAG}"
        f"_{alias}_{size}_{n_valid}{UPSAMPLE_SUFFIX}.npy"
    )


CACHE_DIR = f"{OUT_DIR}/probe_weight_cache_{n_valid}{UPSAMPLE_SUFFIX}"
os.makedirs(CACHE_DIR, exist_ok=True)


def probe_cache_path(telescope, alias, size):
    return f"{CACHE_DIR}/{telescope}_{alias}_{size}.pkl"


def run_probes(telescope, alias, size):
    """Load embeddings, fit one linear probe per property, return results dict."""
    cache = probe_cache_path(telescope, alias, size)
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)

    path = emb_path(telescope, alias, size)
    if not os.path.exists(path):
        return None

    emb = np.load(path)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(emb[idx_train])
    X_test  = scaler.transform(emb[idx_test])

    result = {}
    for prop in TARGET_PROPS:
        y = props_clipped[prop]
        reg = LinearRegression().fit(X_train, y[idx_train])
        pred = reg.predict(X_test)
        result[prop] = {
            "w":    reg.coef_.astype(np.float32),   # shape (d,)
            "pred": pred.astype(np.float32),          # shape (n_test,)
            "r2":   float(r2_score(y[idx_test], pred)),
        }

    with open(cache, "wb") as f:
        pickle.dump(result, f)
    return result


# ── 4. Run probes for all models ────────────────────────────────────────────────
all_results = {}  # (telescope, alias, size) → probe dict

for telescope in TELESCOPES:
    print(f"\n{'─'*60}  {telescope.upper()}")
    for alias, sizes in model_map.items():
        for size in sizes:
            res = run_probes(telescope, alias, size)
            if res is None:
                continue
            key = (telescope, alias, size)
            all_results[key] = res
            r2_str = "  ".join(f"{p}={res[p]['r2']:.3f}" for p in TARGET_PROPS)
            print(f"  {alias:12s} {size:12s}  {r2_str}")


# ── 5. Cosine similarity between probe weight vectors (within each model) ────────
# For each model, W is (3, d); cosine_mats[key] is the (3, 3) Gram matrix.
cosine_mats = {}

for key, res in all_results.items():
    W = np.stack([res[p]["w"] for p in TARGET_PROPS], axis=0)   # (3, d)
    norms = np.linalg.norm(W, axis=1, keepdims=True).clip(min=1e-12)
    W_norm = W / norms
    cosine_mats[key] = (W_norm @ W_norm.T).clip(-1, 1)           # (3, 3)


# ── 6. Procrustes disparity — compute once for all model pairs ───────────────────
# Covers within-telescope and cross-telescope (HSC↔JWST same model) pairs.
# pred vectors are (n_test, 1) and share the same row indices, so comparison is valid.
MODEL_PARAMS = {
    ("vit",       "base"):        86e6,
    ("vit",       "large"):      307e6,
    ("vit",       "huge"):       632e6,
    ("clip",      "base"):        86e6,
    ("clip",      "large"):      307e6,
    ("dinov3",    "vits16"):      21e6,
    ("dinov3",    "vits16plus"):  22e6,
    ("dinov3",    "vitb16"):      86e6,
    ("dinov3",    "vitl16"):     307e6,
    ("dinov3",    "vith16plus"): 650e6,
    ("dinov3",    "vit7b16"):   7000e6,
    ("convnext",  "nano"):        15e6,
    ("convnext",  "tiny"):        28e6,
    ("convnext",  "base"):        89e6,
    ("convnext",  "large"):      198e6,
    ("ijepa",     "huge"):       632e6,
    ("ijepa",     "giant"):     1100e6,
    ("vjepa",     "large"):      307e6,
    ("vjepa",     "huge"):       632e6,
    ("vjepa",     "giant"):     1100e6,
    ("astropt",   "015M"):        15e6,
    ("astropt",   "095M"):        95e6,
    ("astropt",   "850M"):       850e6,
    ("vit-mae",   "base"):        86e6,
    ("vit-mae",   "large"):      307e6,
    ("vit-mae",   "huge"):       632e6,
    ("paligemma", "3b"):        3000e6,
    ("paligemma", "10b"):      10000e6,
    ("paligemma", "28b"):      28000e6,
    ("llava_15",  "7b"):        7000e6,
    ("llava_15",  "13b"):      13000e6,
}

all_keys = list(all_results.keys())
all_pairs = list(combinations(all_keys, 2))

PROC_CACHE_PATH = f"{CACHE_DIR}/proc_cache.pkl"

if os.path.exists(PROC_CACHE_PATH):
    print(f"Loading Procrustes cache from {PROC_CACHE_PATH}…")
    with open(PROC_CACHE_PATH, "rb") as f:
        proc_cache = pickle.load(f)
else:
    proc_cache = {}  # (ki, kj, prop) → disparity; stored symmetrically for easy lookup
    for ki, kj in tqdm(all_pairs, desc="Procrustes (all pairs)"):
        for prop in TARGET_PROPS:
            pred_i = all_results[ki][prop]["pred"].reshape(-1, 1).astype(float)
            pred_j = all_results[kj][prop]["pred"].reshape(-1, 1).astype(float)
            try:
                _, _, disp = scipy_procrustes(pred_i, pred_j)
            except Exception:
                disp = np.nan
            proc_cache[(ki, kj, prop)] = disp
            proc_cache[(kj, ki, prop)] = disp  # symmetric
    with open(PROC_CACHE_PATH, "wb") as f:
        pickle.dump(proc_cache, f)
    print(f"Saved Procrustes cache → {PROC_CACHE_PATH}")

# ── 6a. Within-telescope cross-model distribution (for histogram) ────────────────
proc_disp = {tel: {prop: [] for prop in TARGET_PROPS} for tel in TELESCOPES}
for (ki, kj, prop), disp in proc_cache.items():
    if ki[0] == kj[0] and ki < kj:  # same telescope, count each pair once
        proc_disp[ki[0]][prop].append(disp)

# ── 6b. Intra-modal adjacent-size Procrustes (lookup from cache) ─────────────────
adj_disp = {tel: {} for tel in TELESCOPES}
for telescope in TELESCOPES:
    for alias, sizes in model_map.items():
        alias_data = {prop: [] for prop in TARGET_PROPS}
        for s1, s2 in zip(sizes[:-1], sizes[1:]):
            k1 = (telescope, alias, s1)
            k2 = (telescope, alias, s2)
            if k1 not in all_results or k2 not in all_results:
                continue
            pair_label = f"{s1}→{s2}"
            for prop in TARGET_PROPS:
                alias_data[prop].append((pair_label, proc_cache[(k1, k2, prop)]))
        if any(alias_data[p] for p in TARGET_PROPS):
            adj_disp[telescope][alias] = alias_data

# ── 6c. Cross-modal (HSC vs JWST) Procrustes per model (lookup from cache) ───────
cross_modal_disp = {prop: [] for prop in TARGET_PROPS}
for alias, sizes in model_map.items():
    for size in sizes:
        k_hsc  = ("hsc",  alias, size)
        k_jwst = ("jwst", alias, size)
        if k_hsc not in all_results or k_jwst not in all_results:
            continue
        n_params = MODEL_PARAMS.get((alias, size), np.nan)
        for prop in TARGET_PROPS:
            cross_modal_disp[prop].append(
                (alias, size, n_params, proc_cache[(k_hsc, k_jwst, prop)])
            )


# ── 7. Plots ────────────────────────────────────────────────────────────────────
PROP_LABEL  = {"redshift": "z", "mass": "M★", "sSFR": "sSFR"}
PROP_COLOR  = {"redshift": "#2196F3", "mass": "#4CAF50", "sSFR": "#FF5722"}
TEL_COLOR   = {"hsc": "steelblue", "jwst": "tomato"}
_palette    = plt.rcParams["axes.prop_cycle"].by_key()["color"]
ALIAS_COLOR = {alias: _palette[i % len(_palette)] for i, alias in enumerate(model_map)}

PDF_VARIANTS = [
    ("all",    set(model_map)),
    ("nodinov3", set(model_map) - {"dinov3"}),
]

for variant_tag, active_aliases in PDF_VARIANTS:
    pdf_path = (
        f"{OUT_DIR}/probe_weight_analysis_{n_valid}galaxies"
        f"{UPSAMPLE_SUFFIX}_{variant_tag}.pdf"
    )

    # Re-derive proc_disp restricted to this variant's aliases.
    v_proc_disp = {tel: {prop: [] for prop in TARGET_PROPS} for tel in TELESCOPES}
    for (ki, kj, prop), disp in proc_cache.items():
        if ki[0] == kj[0] and ki < kj and ki[1] in active_aliases and kj[1] in active_aliases:
            v_proc_disp[ki[0]][prop].append(disp)

    with PdfPages(pdf_path) as pdf:

        # ── 7a: cosine-similarity heatmaps (one page per telescope) ─────────────
        for telescope in TELESCOPES:
            tel_keys = sorted(
                [k for k in all_results if k[0] == telescope and k[1] in active_aliases],
                key=lambda k: (k[1], k[2]),
            )
            n = len(tel_keys)
            if n == 0:
                continue

            ncols = min(6, n)
            nrows = (n + ncols - 1) // ncols

            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(ncols * 2.5, nrows * 2.5 + 0.7),
                squeeze=False,
            )
            fig.suptitle(
                f"Cosine similarity between probe weight vectors — {telescope.upper()}\n"
                f"Axes: {' / '.join(PROP_LABEL[p] for p in TARGET_PROPS)}",
                fontsize=12,
            )

            last_im = None
            lbls = [PROP_LABEL[p] for p in TARGET_PROPS]
            for idx, key in enumerate(tel_keys):
                r, c = divmod(idx, ncols)
                ax = axes[r][c]
                mat = cosine_mats[key]
                last_im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
                ax.set_xticks(range(3))
                ax.set_yticks(range(3))
                ax.set_xticklabels(lbls, fontsize=6, rotation=45)
                ax.set_yticklabels(lbls, fontsize=6)
                ax.set_title(f"{key[1]}\n{key[2]}", fontsize=7)
                for i in range(3):
                    for j in range(3):
                        val = mat[i, j]
                        fc = "white" if abs(val) > 0.75 else "black"
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=5.5, color=fc)

            for idx in range(n, nrows * ncols):
                r, c = divmod(idx, ncols)
                axes[r][c].set_visible(False)

            if last_im is not None:
                fig.colorbar(
                    last_im, ax=axes.flatten().tolist(),
                    fraction=0.015, pad=0.04, label="cosine similarity",
                )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # ── 7b: Procrustes disparity histograms (one panel per property) ─────────
        fig, axes = plt.subplots(1, len(TARGET_PROPS), figsize=(5 * len(TARGET_PROPS), 4.5))
        fig.suptitle(
            f"Procrustes disparity of test predictions across model pairs\n"
            f"(n_test={PROBE_TEST_SIZE}, seed={SEED}; each count = one model pair)",
            fontsize=12,
        )
        for ax, prop in zip(axes, TARGET_PROPS):
            for telescope in TELESCOPES:
                d = [x for x in v_proc_disp[telescope][prop] if np.isfinite(x)]
                if d:
                    ax.hist(
                        d, bins=40, alpha=0.6,
                        color=TEL_COLOR[telescope],
                        label=f"{telescope.upper()} (n={len(d)} pairs)",
                    )
            ax.set_xlabel("Procrustes disparity", fontsize=11)
            ax.set_ylabel("Number of model pairs", fontsize=11)
            ax.set_title(prop, fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── 7c: R² overview (one bar chart per telescope) ────────────────────────
        for telescope in TELESCOPES:
            tel_keys = [k for k in all_results
                        if k[0] == telescope and k[1] in active_aliases]
            if not tel_keys:
                continue

            labels = [f"{k[1]}\n{k[2]}" for k in tel_keys]
            x = np.arange(len(tel_keys))
            width = 0.25

            fig, ax = plt.subplots(figsize=(max(10, len(tel_keys) * 0.65), 4.5))
            for i, prop in enumerate(TARGET_PROPS):
                r2s = [all_results[k][prop]["r2"] for k in tel_keys]
                ax.bar(x + i * width, r2s, width, label=prop,
                       color=PROP_COLOR[prop], alpha=0.85)

            ax.set_xticks(x + width)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
            ax.set_ylabel("R²", fontsize=11)
            ax.set_title(
                f"Linear probe R² — {telescope.upper()} (seed={SEED}, n_test={PROBE_TEST_SIZE})",
                fontsize=11,
            )
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            ax.axhline(0, color="k", lw=0.5)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # ── 7d: Intra-modal adjacent-size Procrustes disparity ──────────────────
        for telescope in TELESCOPES:
            fig, axes = plt.subplots(
                1, len(TARGET_PROPS),
                figsize=(5.5 * len(TARGET_PROPS), 4.5),
                squeeze=False,
            )
            fig.suptitle(
                f"Intra-modal Procrustes disparity (adjacent sizes) — {telescope.upper()}",
                fontsize=12,
            )
            for col, prop in enumerate(TARGET_PROPS):
                ax = axes[0][col]
                for alias in (a for a in model_map if a in active_aliases):
                    alias_data = adj_disp.get(telescope, {}).get(alias)
                    if not alias_data or not alias_data[prop]:
                        continue
                    pair_labels = [lbl for lbl, _ in alias_data[prop]]
                    yvals       = [d   for _, d   in alias_data[prop]]
                    ax.plot(range(len(pair_labels)), yvals, marker="o",
                            color=ALIAS_COLOR[alias], label=alias)
                    ax.set_xticks(range(len(pair_labels)))
                    ax.set_xticklabels(pair_labels, rotation=15, ha="right", fontsize=8)
                ax.set_ylabel("Procrustes disparity", fontsize=10)
                ax.set_xlabel("Adjacent size pair", fontsize=10)
                ax.set_title(PROP_LABEL[prop], fontsize=11)
                ax.legend(fontsize=7, ncol=2)
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # ── 7e: Cross-modal Procrustes disparity vs number of model parameters ───
        fig, axes = plt.subplots(1, len(TARGET_PROPS), figsize=(5 * len(TARGET_PROPS), 4.5))
        if len(TARGET_PROPS) == 1:
            axes = [axes]
        fig.suptitle(
            "Cross-modal Procrustes disparity (HSC vs JWST) vs model size",
            fontsize=12,
        )
        for ax, prop in zip(axes, TARGET_PROPS):
            entries = [
                (alias, size, np_, disp)
                for alias, size, np_, disp in cross_modal_disp[prop]
                if alias in active_aliases and np.isfinite(np_) and np.isfinite(disp)
            ]
            if not entries:
                ax.set_title(prop)
                continue
            xs     = [e[2] for e in entries]
            ys     = [e[3] for e in entries]
            labels = [f"{e[0]}\n{e[1]}" for e in entries]
            ax.scatter(xs, ys, alpha=0.85, zorder=3)
            for xi, yi, lbl in zip(xs, ys, labels):
                ax.annotate(lbl, (xi, yi), fontsize=5.5,
                            ha="left", va="bottom",
                            xytext=(3, 3), textcoords="offset points")
            ax.set_xscale("log")
            ax.set_xlabel("Number of parameters", fontsize=11)
            ax.set_ylabel("Procrustes disparity", fontsize=11)
            ax.set_title(prop, fontsize=12)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── 7f: Probe weight cosine similarity vs number of model parameters ─────
        PAIR_INFO = [
            (0, 1, f"{PROP_LABEL['redshift']} vs {PROP_LABEL['mass']}",   "#2196F3"),
            (0, 2, f"{PROP_LABEL['redshift']} vs {PROP_LABEL['sSFR']}",   "#FF5722"),
            (1, 2, f"{PROP_LABEL['mass']} vs {PROP_LABEL['sSFR']}",       "#4CAF50"),
        ]
        TEL_MARKERS = {"hsc": "o", "jwst": "^"}

        fig, ax = plt.subplots(figsize=(8, 5))
        for pi, pj, pair_label, color in PAIR_INFO:
            for telescope in TELESCOPES:
                xs, ys = [], []
                for key in all_results:
                    tel, alias, size = key
                    if tel != telescope or alias not in active_aliases:
                        continue
                    n_params = MODEL_PARAMS.get((alias, size), np.nan)
                    if not np.isfinite(n_params):
                        continue
                    xs.append(n_params)
                    ys.append(float(cosine_mats[key][pi, pj]))
                ax.scatter(xs, ys, color=color, marker=TEL_MARKERS[telescope],
                           alpha=0.75, zorder=3, s=40)

        ax.set_xscale("log")
        ax.set_ylim(-1, 1)
        ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.5)
        ax.set_xlabel("Number of parameters", fontsize=11)
        ax.set_ylabel("Cosine similarity", fontsize=11)
        ax.set_title(
            "Probe weight vector cosine similarity vs model size",
            fontsize=12,
        )
        ax.grid(True, alpha=0.3)

        pair_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                   markersize=8, label=pair_label)
            for _, _, pair_label, color in PAIR_INFO
        ]
        tel_handles = [
            Line2D([0], [0], marker=TEL_MARKERS[tel], color="gray",
                   markerfacecolor="gray", markersize=8, linestyle="None",
                   label=tel.upper())
            for tel in TELESCOPES
        ]
        ax.legend(handles=pair_handles + tel_handles, fontsize=8, ncol=2)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── 7g: Per-telescope averaged cosine-similarity matrix + pair histograms ─
        bins = np.linspace(-1, 1, 101)
        for telescope in TELESCOPES:
            tel_keys = [
                k for k in all_results
                if k[0] == telescope and k[1] in active_aliases
            ]
            if not tel_keys:
                continue

            mats = np.stack([cosine_mats[k] for k in tel_keys], axis=0)  # (N, 3, 3)
            avg_mat = mats.mean(axis=0)

            fig, (ax_heat, ax_hist) = plt.subplots(1, 2, figsize=(10, 4.5))
            fig.suptitle(
                f"Probe weight cosine similarity — {telescope.upper()} "
                f"(avg over {len(tel_keys)} models)",
                fontsize=12,
            )

            lbls = [PROP_LABEL[p] for p in TARGET_PROPS]
            im = ax_heat.imshow(avg_mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
            ax_heat.set_xticks(range(3))
            ax_heat.set_yticks(range(3))
            ax_heat.set_xticklabels(lbls, fontsize=9, rotation=45)
            ax_heat.set_yticklabels(lbls, fontsize=9)
            ax_heat.set_title("Average cosine similarity", fontsize=10)
            for i in range(3):
                for j in range(3):
                    val = avg_mat[i, j]
                    fc = "white" if abs(val) > 0.75 else "black"
                    ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center",
                                 fontsize=9, color=fc)
            fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04,
                         label="cosine similarity")

            for pi, pj, pair_label, color in PAIR_INFO:
                vals = [cosine_mats[k][pi, pj] for k in tel_keys]
                ax_hist.hist(vals, bins=bins, histtype="step",
                             color=color, label=pair_label, lw=1.5)
            ax_hist.set_xlim(-1, 1)
            ax_hist.set_xlabel("Cosine similarity", fontsize=11)
            ax_hist.set_ylabel("Number of models", fontsize=11)
            ax_hist.set_title("Distribution across models", fontsize=10)
            ax_hist.legend(fontsize=8)
            ax_hist.grid(True, alpha=0.3)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved plots → {pdf_path}")

print(f"Probe cache in {CACHE_DIR}")

# ── 8. Average cosine-similarity matrices (HSC + JWST side by side) ─────────────
avg_cos_pdf = (
    f"{OUT_DIR}/avg_cosine_similarity_{n_valid}galaxies{UPSAMPLE_SUFFIX}.pdf"
)
lbls = [PROP_LABEL[p] for p in TARGET_PROPS]

with PdfPages(avg_cos_pdf) as pdf:
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    #fig.suptitle(
    #    "Average probe weight cosine similarity (mean ± std across models)\n"
    #    f"Axes: {' / '.join(lbls)}",
    #    fontsize=12,
    #)

    last_im = None
    for ax, telescope in zip(axes, TELESCOPES):
        tel_keys = [k for k in all_results if k[0] == telescope]
        if not tel_keys:
            ax.set_visible(False)
            continue

        mats    = np.stack([cosine_mats[k] for k in tel_keys], axis=0)  # (N, 3, 3)
        avg_mat = mats.mean(axis=0)
        std_mat = mats.std(axis=0)

        last_im = ax.imshow(avg_mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(lbls, fontsize=9, rotation=45)
        ax.set_yticklabels(lbls, fontsize=9)
        ax.set_title(f"{telescope.upper()} (n={len(tel_keys)} models)", fontsize=11)

        for i in range(3):
            for j in range(3):
                val = avg_mat[i, j]
                std = std_mat[i, j]
                fc  = "white" if abs(val) > 0.75 else "black"
                ax.text(j, i, f"{val:.2f}\n±{std:.2f}",
                        ha="center", va="center", fontsize=12, color=fc)

    if last_im is not None:
        fig.colorbar(last_im, ax=axes.tolist(), fraction=0.025, pad=0.04,
                     label="cosine similarity", location='top')

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"Saved average cosine similarity PDF → {avg_cos_pdf}")

# ── 9. Save all Procrustes distances ────────────────────────────────────────────
proc_all = {
    "cross_model_pairwise":    proc_disp,
    "intra_modal_adjacent":    adj_disp,
    "cross_modal_hsc_vs_jwst": cross_modal_disp,
}
proc_save_path = f"{OUT_DIR}/procrustes_distances_{n_valid}galaxies{UPSAMPLE_SUFFIX}.pkl"
with open(proc_save_path, "wb") as f:
    pickle.dump(proc_all, f)
print(f"Saved Procrustes distances to {proc_save_path}")
