#!/usr/bin/env python3
"""
Plot per-model crossmodal MKNN / CKA against the HSC mean physics R² of
each model.

MKNN / CKA values are transcribed from the manuscript's crossmodal table
(one number per model per modality, for JWST / Legacy Survey / DESI —
each compared against the model's own HSC embeddings). R² is the HSC
mean across (redshift, mass, sSFR) from Ashod's
``r2_vs_params_45000galaxies_upsampled.json``, looked up for that model.

One panel per modality. One point per model. Tests whether models that
align well across modalities also retain physics information.
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

MODALITIES = ("jwst", "legacysurvey", "desi")
MODALITY_LABEL = {
    "jwst": "JWST",
    "legacysurvey": "Legacy Survey",
    "desi": "DESI",
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

# (family, size,
#  mknn_jwst%, mknn_legacysurvey%, mknn_desi%,
#  cka_jwst%,  cka_legacysurvey%,  cka_desi%)
# None denotes a missing measurement (shown as "--" in the manuscript).
# Family/size keys match r2_vs_params_45000galaxies_upsampled.json. The
# AstroPTv2 Small/Base/Large rows in the paper correspond to astropt
# 15m/95m/850m in the R² JSON.
MODELS = [
    ("astropt",   "15m",        11.62, None, 1.25, 43.42, None, 45.66),
    ("astropt",   "95m",        12.60, None, 1.32, 42.03, None, 45.39),
    ("astropt",   "850m",       14.30, None, 1.41, 41.93, None, 44.47),
    ("clip",      "base",       12.88, None, 1.29, 30.59, None, 33.39),
    ("clip",      "large",      14.07, None, 1.24, 31.89, None, 33.85),
    ("convnext",  "nano",       11.33, None, 1.01, 32.52, None, 32.29),
    ("convnext",  "tiny",       11.91, None, 1.05, 28.57, None, 29.64),
    ("convnext",  "base",       10.57, None, 0.87, 33.28, None, 33.32),
    ("convnext",  "large",      12.23, None, 1.01, 34.01, None, 36.50),
    ("dinov3",    "vits16",     14.55, None, 0.98, 52.18, None, 34.80),
    ("dinov3",    "vits16plus", 13.09, None, 0.97, 48.53, None, 35.39),
    ("dinov3",    "vitb16",     14.03, None, 0.92, 49.44, None, 33.25),
    ("dinov3",    "vitl16",     11.80, None, 0.79, 45.31, None, 30.14),
    ("dinov3",    "vith16plus", 10.35, None, 0.67, 31.58, None, 22.21),
    ("dinov3",    "vit7b16",    12.62, None, 0.82, 40.14, None, 30.13),
    ("ijepa",     "huge",        9.85, None, 0.56, 15.22, None, 15.86),
    ("ijepa",     "giant",      11.63, None, 0.73, 21.25, None, 25.40),
    ("llava_15",  "7b",         11.16, None, 1.04, 31.10, None, 36.88),
    ("llava_15",  "13b",        11.70, None, 1.06, 45.03, None, 38.11),
    ("paligemma", "3b",         11.50, None, 1.18, 43.20, None, 32.89),
    ("paligemma", "10b",        12.23, None, 1.17, 41.98, None, 33.97),
    ("paligemma", "28b",         7.03, None, None, 18.85, None, None),
    ("vit-mae",   "base",       10.09, None, 1.01, 59.34, None, 29.25),
    ("vit-mae",   "large",      10.75, None, 0.94, 59.23, None, 28.39),
    ("vit-mae",   "huge",       11.22, None, 0.93, 61.66, None, 28.59),
    ("vit",       "base",       10.48, None, 1.02, 30.13, None, 35.17),
    ("vit",       "large",      13.01, None, 1.03, 47.00, None, 34.98),
    ("vit",       "huge",       15.88, None, 1.14, 46.32, None, 37.18),
    ("vjepa",     "large",      14.27, None, 0.82, 59.63, None, 23.98),
    ("vjepa",     "huge",       10.98, None, 0.74, 24.86, None, 23.71),
    ("vjepa",     "giant",      13.19, None, 0.89, 24.57, None, 30.73),
]

MKNN_COL = {"jwst": 2, "legacysurvey": 3, "desi": 4}
CKA_COL  = {"jwst": 5, "legacysurvey": 6, "desi": 7}


def load_r2_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def r2_for_model(r2: dict, family: str, size: str) -> float:
    try:
        entry = r2["hsc"][family][size]
    except KeyError as e:
        raise KeyError(
            f"{family}/{size} missing under modality 'hsc' in R² JSON"
        ) from e
    vals = []
    for prop in R2_PROPS:
        if prop not in entry:
            raise KeyError(
                f"{family}/{size} missing property {prop!r} under 'hsc'"
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
        rho, p_rho = spearmanr(x[finite], y[finite])
        r, p_r = pearsonr(x[finite], y[finite])
        ax.text(
            0.95, -0.01,
            f"ρ = {rho:.3f}  (p = {p_rho:.1g})\n",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
            bbox=None,
        )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)


def _make_figure(
    models: list[tuple],
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

    kept = [m for m in models if m[0] not in exclude_families]
    if not kept:
        raise RuntimeError("No models left after applying --exclude-families")

    n_panels = len(modalities)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(8, 3.0), sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, modality in zip(axes, modalities):
        col = col_map[modality]
        xs, ys, fams, szs = [], [], [], []
        for m in kept:
            family, size = m[0], m[1]
            val = m[col]
            if val is None:
                continue
            xs.append(float(val) / 100.0)
            ys.append(r2_for_model(r2, family, size))
            fams.append(family)
            szs.append(size)

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
        frameon=False,
    )

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0)
    out = FIGS_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def make_figure_mknn(
    models: list[tuple],
    r2: dict,
    exclude_families: set[str],
    modalities: tuple[str, ...],
) -> None:
    _make_figure(
        models, r2, exclude_families, modalities,
        col_map=MKNN_COL,
        metric_label="MKNN",
        out_name="crossmodal_ashod.pdf",
    )


def make_figure_cka(
    models: list[tuple],
    r2: dict,
    exclude_families: set[str],
    modalities: tuple[str, ...],
) -> None:
    _make_figure(
        models, r2, exclude_families, modalities,
        col_map=CKA_COL,
        metric_label="CKA",
        out_name="crossmodal_ashod_cka.pdf",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-json", type=Path, default=DEFAULT_R2_JSON,
                        help="Path to Ashod's r2_vs_params JSON")
    parser.add_argument("--modalities", nargs="+",
                        default=["jwst", "desi"],
                        choices=MODALITIES,
                        help="Subset of modality panels to plot "
                             "(legacysurvey has no values in the table)")
    parser.add_argument("--exclude-families", nargs="+", default=["dinov3"],
                        metavar="FAMILY",
                        help=f"Family names to drop (valid: "
                             f"{sorted(FAMILY_STYLE)})")
    args = parser.parse_args()

    FIGS_DIR.mkdir(exist_ok=True)

    r2 = load_r2_json(args.r2_json)
    print(f"Loaded R² JSON from {args.r2_json}")
    print(f"Panels: {args.modalities}")
    print(f"Models: {len(MODELS)} total, "
          f"{len([m for m in MODELS if m[0] not in set(args.exclude_families)])} "
          f"after --exclude-families={args.exclude_families or '[]'}")

    exclude = set(args.exclude_families)
    modalities = tuple(args.modalities)
    make_figure_mknn(MODELS, r2, exclude, modalities)
    make_figure_cka(MODELS, r2, exclude, modalities)


if __name__ == "__main__":
    main()
