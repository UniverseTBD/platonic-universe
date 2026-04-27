#!/usr/bin/env python3
"""
Plot per-pair intramodal MKNN against the HSC mean physics R² of the
*larger* model in each pair.

MKNN values are transcribed from the manuscript's intramodal table (one
number per model-pair per modality, for JWST / Legacy Survey / HSC). R²
is the HSC mean across (redshift, mass, sSFR) from
``r2_vs_params_45000galaxies_upsampled.json``, looked up for the larger
member of each pair.

Three panels, one per modality. One point per model-pair. Tests whether
model pairs whose representations stay close as a family scales up also
retain physics information as they scale up.
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

MODALITIES = ("jwst", "legacysurvey", "hsc")
MODALITY_LABEL = {
    "jwst": "JWST",
    "legacysurvey": "Legacy Survey",
    "hsc": "HSC",
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

# (family, small, large,
#  mknn_jwst%, mknn_legacysurvey%, mknn_hsc%,
#  cka_jwst%,  cka_legacysurvey%,  cka_hsc%)
PAIRS = [
    ("astropt",   "15m",        "95m",        47.2, 35.5, 35.7, 96.9, 98.4, 98.1),
    ("astropt",   "95m",        "850m",       51.6, 39.4, 39.6, 97.2, 98.3, 98.1),
    ("clip",      "base",       "large",      30.7,  5.9,  6.2, 66.7, 69.4, 68.8),
    ("convnext",  "nano",       "tiny",       29.6,  4.4,  4.8, 66.3, 70.3, 70.4),
    ("convnext",  "tiny",       "base",       29.0,  3.9,  4.2, 72.0, 68.4, 67.7),
    ("convnext",  "base",       "large",      35.8,  5.9,  6.3, 80.5, 77.0, 76.7),
    ("dinov3",    "vits16",     "vits16plus", 43.6, 10.8, 16.5, 88.5, 89.7, 86.7),
    ("dinov3",    "vits16plus", "vitb16",     38.2,  9.6, 14.3, 86.5, 88.5, 86.3),
    ("dinov3",    "vitb16",     "vitl16",     32.7,  7.1,  9.4, 73.6, 73.1, 68.0),
    ("dinov3",    "vitl16",     "vith16plus", 34.0,  6.3,  8.8, 78.5, 68.7, 66.6),
    ("dinov3",    "vith16plus", "vit7b16",    40.5,  7.8, 11.7, 81.8, 62.1, 64.2),
    ("vit-mae",   "base",       "large",      23.5,  4.2,  4.5, 93.0, 95.9, 95.9),
    ("vit-mae",   "large",      "huge",       26.8,  4.3,  4.6, 94.0, 96.2, 96.1),
    ("vit",       "base",       "large",      22.7, 11.7,  9.8, 72.4, 83.5, 92.1),
    ("vit",       "large",      "huge",       25.8, 14.3, 12.3, 66.5, 72.7, 97.8),
    ("paligemma", "3b",         "10b",        27.7, 25.7, 16.9, 75.9, 72.5, 85.2),
    ("paligemma", "10b",        "28b",        29.5, 28.2, 17.9, 80.3, 66.5, 81.3),
    ("vjepa",     "large",      "huge",       31.9, 32.1, 20.0, 89.7, 82.4, 74.2),
    ("vjepa",     "huge",       "giant",      35.4, 36.0, 20.0, 84.7, 80.6, 80.3),
    ("ijepa",     "huge",       "giant",      28.5, 10.8, 12.2, 84.8, 95.0, 84.7),
    ("llava_15",  "7b",         "13b",        28.4, 21.2, 17.2, 64.3, 83.6, 59.3),
]

MKNN_COL = {"jwst": 3, "legacysurvey": 4, "hsc": 5}
CKA_COL  = {"jwst": 6, "legacysurvey": 7, "hsc": 8}


def load_r2_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def r2_for_larger(r2: dict, family: str, large_size: str) -> float:
    try:
        entry = r2["hsc"][family][large_size]
    except KeyError as e:
        raise KeyError(
            f"{family}/{large_size} missing under modality 'hsc' in R² JSON"
        ) from e
    vals = []
    for prop in R2_PROPS:
        if prop not in entry:
            raise KeyError(
                f"{family}/{large_size} missing property {prop!r} under 'hsc'"
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
            s=30, label=style["label"], edgecolors="black", linewidths=0.4,
        )
        #for xi, yi, sz in zip(x[mask], y[mask], np.array(sizes)[mask]):
        #    ax.annotate(
        #        sz, (xi, yi), xytext=(4, 2), textcoords="offset points",
        #        fontsize=6, color=style["color"],
        #    )

    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() >= 3:
        rho, p_rho = spearmanr(x[finite], y[finite])
        r, p_r = pearsonr(x[finite], y[finite])
        ax.text(
            0.95, -0.01,
            f"ρ = {rho:.3f}  (p = {p_rho:.1g})\n",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
            bbox=None,
        )

        m, b = np.polyfit(x[finite], y[finite], 1)
        xlim = ax.get_xlim()
        xfit = np.linspace(xlim[0], xlim[1], 200)
        ax.plot(xfit, m * xfit + b, color="gray", lw=2, ls="--", zorder=0)
        ax.set_xlim(xlim)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)


def _make_figure(
    pairs: list[tuple],
    r2: dict,
    exclude_families: set[str],
    modalities: tuple[str, ...],
    col_map: dict[str, int],
    metric_label: str,
    out_name: str,
) -> None:
    unknown = exclude_families - set(FAMILY_STYLE)
    if unknown:
        raise ValueError(
            f"Unknown family names in --exclude-families: {sorted(unknown)}. "
            f"Valid: {sorted(FAMILY_STYLE)}"
        )

    kept = [p for p in pairs if p[0] not in exclude_families]
    if not kept:
        raise RuntimeError("No pairs left after applying --exclude-families")

    n_panels = len(modalities)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(8, 2.0), sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, modality in zip(axes, modalities):
        col = col_map[modality]
        xs, ys, fams, szs = [], [], [], []
        for p in kept:
            family, small, large = p[0], p[1], p[2]
            xs.append(float(p[col]) / 100.0)
            ys.append(r2_for_larger(r2, family, large))
            fams.append(family)
            szs.append(f"{small}→{large}")

        is_first = ax is axes[0]

        plot_scatter(
            ax,
            np.array(xs)*100, np.array(ys),
            fams, szs,
            xlabel=f"{MODALITY_LABEL[modality]} [{metric_label} %]",
            ylabel=(
                "Mean $R^2$"
                if is_first else ""
            ),
        )

    seen: dict[str, object] = {}
    for ax in axes:
        ax.tick_params(axis="x",direction="in")
        ax.tick_params(axis="y",direction="in")
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in seen:
                seen[lab] = h
    fig.legend(
        seen.values(), list(seen.keys()),
        loc="upper center", fontsize=9, ncol=len(seen),
        columnspacing=0.55,
        bbox_to_anchor=(0.52, 1.08),
        handletextpad=0.1,
        frameon=False
    )

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0)
    out = FIGS_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def make_figure_mknn(
    pairs: list[tuple],
    r2: dict,
    exclude_families: set[str],
    modalities: tuple[str, ...],
) -> None:
    _make_figure(
        pairs, r2, exclude_families, modalities,
        col_map=MKNN_COL,
        metric_label="MKNN",
        out_name="intramodal.pdf",
    )


def make_figure_cka(
    pairs: list[tuple],
    r2: dict,
    exclude_families: set[str],
    modalities: tuple[str, ...],
) -> None:
    _make_figure(
        pairs, r2, exclude_families, modalities,
        col_map=CKA_COL,
        metric_label="CKA",
        out_name="intramodal_cka.pdf",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-json", type=Path, default=DEFAULT_R2_JSON,
                        help="Path to r2_vs_params JSON")
    parser.add_argument("--modalities", nargs="+", default=list(MODALITIES),
                        choices=MODALITIES,
                        help="Subset of modality panels to plot")
    parser.add_argument("--exclude-families", nargs="+", default=["dinov3"],
                        metavar="FAMILY",
                        help=f"Family names to drop (valid: "
                             f"{sorted(FAMILY_STYLE)})")
    args = parser.parse_args()

    FIGS_DIR.mkdir(exist_ok=True)

    r2 = load_r2_json(args.r2_json)
    print(f"Loaded R² JSON from {args.r2_json}")
    print(f"Panels: {args.modalities}")
    print(f"Pairs: {len(PAIRS)} total, "
          f"{len([p for p in PAIRS if p[0] not in set(args.exclude_families)])} "
          f"after --exclude-families={args.exclude_families or '[]'}")

    exclude = set(args.exclude_families)
    modalities = tuple(args.modalities)
    make_figure_mknn(PAIRS, r2, exclude, modalities)
    make_figure_cka(PAIRS, r2, exclude, modalities)


if __name__ == "__main__":
    main()
