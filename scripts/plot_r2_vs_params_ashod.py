#!/usr/bin/env python3
"""
Plot model parameter count against mean physics R² in the same style
as scripts/plot_crossmodal_ashod.py.

R² is the mean across (redshift, mass, sSFR) from Ashod's
``r2_vs_params_45000galaxies_upsampled.json``. One panel per modality
(HSC / JWST). One point per (family, size).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent
FIGS_DIR = ROOT / "figs"

DEFAULT_R2_JSON = ROOT / "r2_vs_params_45000galaxies_upsampled.json"
R2_PROPS = ("redshift", "mass", "sSFR")

MODALITIES = ("hsc", "jwst")
MODALITY_LABEL = {
    "hsc": "HSC",
    "jwst": "JWST",
}

FAMILY_STYLE = {
    "vit":       {"label": "ViT",         "color": "#1f77b4", "marker": "o"},
    "vit-mae":   {"label": "ViT-MAE",     "color": "#efcc00", "marker": "<"},
    "clip":      {"label": "CLIP",        "color": "#ff7f0e", "marker": "s"},
    "convnext":  {"label": "ConvNeXt",    "color": "#2ca02c", "marker": "^"},
    "dinov3":    {"label": "DINOv3",      "color": "#d62728", "marker": "D"},
    "vjepa":     {"label": "V-JEPA",      "color": "#8c564b", "marker": "P"},
    "ijepa":     {"label": "I-JEPA",      "color": "#9467bd", "marker": "v"},
    "astropt":   {"label": "AstroPT",     "color": "#e377c2", "marker": "*"},
    "paligemma": {"label": "PaliGemma 2", "color": "#17becf", "marker": "h"},
    "llava_15":  {"label": "LLaVA 1.5",   "color": "#7f7f7f", "marker": "X"},
}

# Parameter counts (HuggingFace model_info().safetensors.total). For CLIP
# we use the vision-encoder count; for VLMs the full model since the
# vision backbone is shared across sizes.
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
        "15m": 15_000_000,
        "95m": 95_000_000,
        "850m": 850_000_000,
    },
    "paligemma": {
        "3b": 3_032_081_408,
        "10b": 9_670_746_112,
        "28b": 27_227_128_832,
    },
    "llava_15": {
        "7b": 7_062_898_688,
        "13b": 13_015_864_320,
    },
}


def load_r2_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def r2_for_model(r2: dict, modality: str, family: str, size: str) -> float:
    try:
        entry = r2[modality][family][size]
    except KeyError as e:
        raise KeyError(
            f"{family}/{size} missing under modality {modality!r} in R² JSON"
        ) from e
    vals = []
    for prop in R2_PROPS:
        if prop not in entry:
            raise KeyError(
                f"{family}/{size} missing property {prop!r} under {modality!r}"
            )
        vals.append(float(entry[prop]["r2_mean"]))
    return float(np.mean(vals))


def plot_scatter(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    families: list[str],
    sizes: list[str],
    xlabel: str,
    ylabel: str,
) -> None:
    for family in FAMILY_STYLE:
        mask = np.array([f == family for f in families])
        if not mask.any():
            continue
        style = FAMILY_STYLE[family]
        ax.scatter(
            x[mask], y[mask],
            color=style["color"], marker=style["marker"],
            s=70, label=style["label"], edgecolors="black", linewidths=0.4,
        )

    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() >= 3:
        # Spearman is rank-based, so log-x doesn't change it; report as-is.
        rho, p_rho = spearmanr(x[finite], y[finite])
        r, p_r = pearsonr(np.log10(x[finite]), y[finite])
        ax.text(
            0.95, -0.01,
            f"ρ = {rho:.3f}  (p = {p_rho:.1g})\n",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
            bbox=None,
        )
        log_x = np.log10(x[finite])
        m, b = np.polyfit(log_x, y[finite], 1)
        xfit = np.logspace(log_x.min(), log_x.max(), 200)
        ax.plot(xfit, m * np.log10(xfit) + b, color="gray", lw=1, ls="--", zorder=0)

    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)


def make_figure(
    r2: dict,
    exclude_families: set[str],
    modalities: tuple[str, ...],
) -> None:
    unknown = exclude_families - set(FAMILY_STYLE)
    if unknown:
        raise ValueError(
            f"Unknown family names in --exclude-families: {sorted(unknown)}. "
            f"Valid: {sorted(FAMILY_STYLE)}"
        )

    n_panels = len(modalities)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(8, 3.0)#, sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, modality in zip(axes, modalities):
        xs, ys, fams, szs = [], [], [], []
        for family, sizes in PARAM_COUNTS.items():
            if family in exclude_families:
                continue
            for size, n_params in sizes.items():
                try:
                    r2_val = r2_for_model(r2, modality, family, size)
                except KeyError:
                    continue
                xs.append(float(n_params))
                ys.append(r2_val)
                fams.append(family)
                szs.append(size)

        is_first = ax is axes[0]
        plot_scatter(
            ax,
            np.array(xs), np.array(ys),
            fams, szs,
            xlabel=f"{MODALITY_LABEL[modality]} [Parameters]",
            ylabel=("Mean $R^2$" if is_first else ""),
        )

    seen: dict[str, object] = {}
    for ax in axes:
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in seen:
                seen[lab] = h
    fig.legend(
        seen.values(), list(seen.keys()),
        loc="upper center", fontsize=9, ncol=len(seen),
        columnspacing=0.55,
        bbox_to_anchor=(0.52, 1.08),
        handletextpad=0.1,
        frameon=False,
    )

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0)
    out = FIGS_DIR / "r2_vs_params_ashod.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-json", type=Path, default=DEFAULT_R2_JSON,
                        help="Path to Ashod's r2_vs_params JSON")
    parser.add_argument("--modalities", nargs="+", default=list(MODALITIES),
                        choices=MODALITIES,
                        help="Subset of modality panels to plot")
    parser.add_argument("--exclude-families", nargs="*", default=[],
                        metavar="FAMILY",
                        help=f"Family names to drop (valid: "
                             f"{sorted(FAMILY_STYLE)})")
    args = parser.parse_args()

    FIGS_DIR.mkdir(exist_ok=True)

    r2 = load_r2_json(args.r2_json)
    print(f"Loaded R² JSON from {args.r2_json}")
    print(f"Panels: {args.modalities}")
    kept = [
        (f, s) for f, sizes in PARAM_COUNTS.items()
        if f not in set(args.exclude_families)
        for s in sizes
    ]
    print(f"Models: {sum(len(v) for v in PARAM_COUNTS.values())} total, "
          f"{len(kept)} after --exclude-families={args.exclude_families or '[]'}")

    exclude = set(args.exclude_families)
    modalities = tuple(args.modalities)
    make_figure(r2, exclude, modalities)


if __name__ == "__main__":
    main()
