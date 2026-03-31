#!/usr/bin/env python3
"""Plot R² (linear probe on galaxy physics) vs model parameter count."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIGS_DIR = Path(__file__).resolve().parent.parent / "figs"

# Parameter counts from HuggingFace Hub model_info().safetensors.total
# For models with text+vision (CLIP), we use *vision encoder only* counts.
# For AstroPT, sizes are in the name (15M, 95M, 850M).
PARAM_COUNTS = {
    "vit": {
        "base": 86_389_248,
        "large": 304_351_232,
        "huge": 632_404_480,
    },
    "vit-mae": {
        "base": 86_389_248,
        "large": 304_351_232,
        "huge": 632_404_480,
    },
    "clip": {
        # Vision encoder only (ViT-B/16 and ViT-L/14)
        "base": 86_192_640,
        "large": 303_971_328,
    },
    "convnext": {
        "nano": 15_623_800,
        "tiny": 28_635_496,
        "base": 88_717_800,
        "large": 197_956_840,
    },
    "dinov3": {
        "vits16": 21_596_544,
        "vits16plus": 28_692_864,
        "vitb16": 85_660_416,
        "vitl16": 303_129_600,
        "vith16plus": 840_592_640,
        "vit7b16": 6_716_035_072,
    },
    "ijepa": {
        "huge": 630_762_240,
        "giant": 1_011_368_576,
    },
    "vjepa": {
        "large": 325_971_328,
        "huge": 653_930_880,
        "giant": 1_034_555_264,
    },
    "astropt": {
        "015M": 15_000_000,
        "095M": 95_000_000,
        "850M": 850_000_000,
    },
}

# Display names and colors for each family
FAMILY_STYLE = {
    "vit":      {"label": "ViT",      "color": "#1f77b4", "marker": "o"},
    "vit-mae":  {"label": "ViT-MAE",  "color": "#efcc00", "marker": "<"},
    "clip":     {"label": "CLIP",     "color": "#ff7f0e", "marker": "s"},
    "convnext": {"label": "ConvNeXt", "color": "#2ca02c", "marker": "^"},
    "dinov3":   {"label": "DINOv3",   "color": "#d62728", "marker": "D"},
    "ijepa":    {"label": "I-JEPA",   "color": "#9467bd", "marker": "v"},
    "vjepa":    {"label": "V-JEPA",   "color": "#8c564b", "marker": "P"},
    "astropt":  {"label": "AstroPT",  "color": "#e377c2", "marker": "*"},
}


PROPERTY_LABELS = {
    "smooth_fraction": "Smooth Fraction",
    "disk_fraction": "Disk Fraction",
    "artifact": "Artifact",
    "edge_on": "Edge-on",
    "tight_spiral": "Tight Spiral",
    "mag_r_desi": "Mag r (DESI)",
    "mag_g_desi": "Mag g (DESI)",
    "photo_z": "Photo-z",
    "spec_z": "Spec-z",
    "stellar_mass": "Stellar Mass",
    "sfr": "SFR",
}


def load_all_data():
    """Load all physics test JSON files, return {family: data_dict}."""
    all_data = {}
    for family in PARAM_COUNTS:
        data_file = DATA_DIR / f"physics_{family}_test.json"
        if not data_file.exists():
            print(f"Skipping {family}: {data_file} not found")
            continue
        with open(data_file) as f:
            all_data[family] = json.load(f)
    return all_data


EXCLUDE_PROPERTIES = {"tight_spiral", "sfr"}


def _recompute_mean_r2(size_data):
    """Recompute mean R² and SE excluding EXCLUDE_PROPERTIES."""
    r2_vals = []
    for prop, v in size_data["r2_per_property"].items():
        if prop in EXCLUDE_PROPERTIES:
            continue
        r2_vals.append(v)
    r2_arr = np.array(r2_vals)
    return r2_arr.mean(), r2_arr.std(ddof=1) / np.sqrt(len(r2_arr))


def plot_mean(all_data):
    """Plot mean R² vs model size, excluding unstable properties."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for family, params_by_size in PARAM_COUNTS.items():
        if family not in all_data:
            continue
        sizes_data = all_data[family]["sizes"]
        xs, ys, yerrs = [], [], []
        for size, n_params in params_by_size.items():
            if size not in sizes_data:
                continue
            mean, se = _recompute_mean_r2(sizes_data[size])
            xs.append(n_params)
            ys.append(mean)
            yerrs.append(se)

        style = FAMILY_STYLE[family]
        ax.errorbar(
            xs, ys, yerr=yerrs,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            markersize=7,
            linewidth=1.5,
            capsize=3,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Parameters", fontsize=12)
    ax.set_ylabel("Mean $R^2$ (linear probe)", fontsize=12)
    n_used = len(PROPERTY_LABELS) - len(EXCLUDE_PROPERTIES)
    excluded = ", ".join(sorted(EXCLUDE_PROPERTIES))
    ax.set_title(
        f"Galaxy Physics $R^2$ vs Model Size ({n_used} properties, excl. {excluded})",
        fontsize=11,
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIGS_DIR / "r2_vs_model_size.pdf"
    fig.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.close(fig)


def plot_per_property(all_data):
    """Plot R² vs model size for each physical property in a grid."""
    properties = list(PROPERTY_LABELS.keys())
    n_props = len(properties)
    ncols = 4
    nrows = (n_props + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2))
    axes = axes.flat

    for i, prop in enumerate(properties):
        ax = axes[i]

        for family, params_by_size in PARAM_COUNTS.items():
            if family not in all_data:
                continue
            sizes_data = all_data[family]["sizes"]
            xs, ys, yerrs = [], [], []
            for size, n_params in params_by_size.items():
                if size not in sizes_data:
                    continue
                prop_data = sizes_data[size].get("properties", {}).get(prop, {})
                r2 = prop_data.get("linear_probe_r2")
                r2_std = prop_data.get("linear_probe_r2_std", 0)
                if r2 is None:
                    continue
                xs.append(n_params)
                ys.append(r2)
                yerrs.append(r2_std)

            style = FAMILY_STYLE[family]
            ax.errorbar(
                xs, ys, yerr=yerrs,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                markersize=5,
                linewidth=1.2,
                capsize=2,
            )

        ax.set_xscale("log")
        ax.set_title(PROPERTY_LABELS[prop], fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("$R^2$", fontsize=9)
        ax.set_xlabel("Parameters", fontsize=9)
        ax.tick_params(labelsize=8)

    # Hide unused subplots
    for j in range(n_props, len(axes)):
        axes[j].set_visible(False)

    # Single legend in the empty subplot space
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=10,
               bbox_to_anchor=(0.98, 0.02), ncol=2)

    fig.suptitle("$R^2$ vs Model Size by Physical Property", fontsize=14, y=1.01)
    fig.tight_layout()
    out = FIGS_DIR / "r2_vs_model_size_per_property.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.close(fig)


def main():
    FIGS_DIR.mkdir(exist_ok=True)
    all_data = load_all_data()
    plot_mean(all_data)
    plot_per_property(all_data)


if __name__ == "__main__":
    main()
