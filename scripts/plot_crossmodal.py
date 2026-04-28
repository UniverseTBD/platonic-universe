#!/usr/bin/env python3
"""
Plot per-model crossmodal MKNN / CKA against the HSC mean physics R² of
each model.

MKNN / CKA values are transcribed from the manuscript's crossmodal table
(one number per model per modality, for JWST / Legacy Survey / DESI —
each compared against the model's own HSC embeddings). R² is the HSC
mean across (redshift, mass, sSFR) from
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
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
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
    ("astropt",   "15m",        11.62, 3.36, 1.25, 43.42, 86.58, 45.66),
    ("astropt",   "95m",        12.60, 3.28, 1.32, 42.03, 84.89, 45.39),
    ("astropt",   "850m",       14.30, 3.84, 1.41, 41.93, 84.90, 44.47),
    ("clip",      "base",       12.88, 1.79, 1.29, 30.59, 59.32, 33.39),
    ("clip",      "large",      14.07, 2.28, 1.24, 31.89, 72.73, 33.85),
    ("convnext",  "nano",       11.33, 1.13, 1.01, 32.52, 54.06, 32.29),
    ("convnext",  "tiny",       11.91, 1.18, 1.05, 28.57, 52.93, 29.64),
    ("convnext",  "base",       10.57, 0.99, 0.87, 33.28, 55.73, 33.32),
    ("convnext",  "large",      12.23, 1.21, 1.01, 34.01, 50.49, 36.50),
    ("dinov3",    "vits16",     14.55, 2.42, 0.98, 52.18, 70.23, 34.80),
    ("dinov3",    "vits16plus", 13.09, 2.09, 0.97, 48.53, 68.06, 35.39),
    ("dinov3",    "vitb16",     14.03, 2.48, 0.92, 49.44, 70.29, 33.25),
    ("dinov3",    "vitl16",     11.80, 1.70, 0.79, 45.31, 66.07, 30.14),
    ("dinov3",    "vith16plus", 10.35, 1.16, 0.67, 31.58, 43.12, 22.21),
    ("dinov3",    "vit7b16",    12.62, 1.89, 0.82, 40.14, 41.65, 30.13),
    ("ijepa",     "huge",        9.85, 1.27, 0.56, 15.22, 31.64, 15.86),
    ("ijepa",     "giant",      11.63, 2.09, 0.73, 21.25, 51.31, 25.40),
    ("llava_15",  "7b",         11.16, 1.67, 1.04, 31.10, 56.45, 36.88),
    ("llava_15",  "13b",        11.70, 1.86, 1.06, 45.03, 73.30, 38.11),
    ("paligemma", "3b",         11.50, 1.84, 1.18, 43.20, 58.58, 32.89),
    ("paligemma", "10b",        12.23, 1.81, 1.17, 41.98, 47.51, 33.97),
    ("paligemma", "28b",         7.03, 0.56, 0.50, 18.85, 53.39, 22.21),
    ("vit-mae",   "base",       10.09, 1.89, 1.01, 59.34, 94.89, 29.25),
    ("vit-mae",   "large",      10.75, 1.98, 0.94, 59.23, 94.49, 28.39),
    ("vit-mae",   "huge",       11.22, 2.15, 0.93, 61.66, 95.35, 28.59),
    ("vit",       "base",       10.48, 1.20, 1.02, 30.13, 47.35, 35.17),
    ("vit",       "large",      13.01, 1.94, 1.03, 47.00, 67.19, 34.98),
    ("vit",       "huge",       15.88, 3.51, 1.14, 46.32, 74.42, 37.18),
    ("vjepa",     "large",      14.27, 3.04, 0.82, 59.63, 77.48, 23.98),
    ("vjepa",     "huge",       10.98, 1.77, 0.74, 24.86, 52.76, 23.71),
    ("vjepa",     "giant",      13.19, 2.11, 0.89, 24.57, 77.59, 30.73),
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


def ancova_partial_slope(
    x: np.ndarray, y: np.ndarray, families: list[str]
) -> dict:
    """Test partial effect of x on y after absorbing family means.

    Fits ``y ~ x + C(family)`` via :mod:`statsmodels` OLS (Patsy
    treatment-codes the family factor) and reports the Type-II ANOVA
    F-test on ``x``. With one continuous covariate this is identical
    to comparing the full model against ``y ~ C(family)``, and the
    F equals the square of the t-statistic on the ``x`` coefficient.
    """
    finite = np.isfinite(x) & np.isfinite(y)
    df = pd.DataFrame({
        "y": y[finite],
        "x": x[finite],
        "family": [f for f, ok in zip(families, finite) if ok],
    })
    if len(df) < 3 or df["family"].nunique() < 2:
        return {"slope": np.nan, "F": np.nan, "p": np.nan,
                "df_num": 1, "df_denom": max(len(df) - 2, 0),
                "n": len(df), "k": df["family"].nunique()}

    model = smf.ols("y ~ x + C(family)", data=df).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    return {
        "slope": float(model.params["x"]),
        "F": float(aov.loc["x", "F"]),
        "p": float(aov.loc["x", "PR(>F)"]),
        "df_num": int(aov.loc["x", "df"]),
        "df_denom": int(aov.loc["Residual", "df"]),
        "n": int(model.nobs),
        "k": int(df["family"].nunique()),
    }


def family_demean(values: np.ndarray, families: list[str]) -> np.ndarray:
    """Return values minus each entry's family-mean."""
    out = np.array(values, dtype=float).copy()
    fams = np.asarray(families)
    for fam in np.unique(fams):
        mask = fams == fam
        out[mask] = out[mask] - out[mask].mean()
    return out


def plot_partial(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    families: list[str],
    xlabel: str,
    ylabel: str,
) -> dict | None:
    """Added-variable plot for the ANCOVA partial slope.

    Family-demeans both axes and scatters the residuals; the OLS slope
    through them equals β₁ in ``y ~ x + family``, and its F-test
    matches :func:`ancova_partial_slope`. Singleton families contribute
    a (0, 0) point (no within-family variation) and so don't shift the
    fit.
    """
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]; y = y[finite]
    fams = [f for f, ok in zip(families, finite) if ok]
    if len(y) < 3:
        return None
    x_res = family_demean(x, fams)
    y_res = family_demean(y, fams)

    for family in FAMILY_STYLE:
        mask = np.array([f == family for f in fams])
        if not mask.any():
            continue
        st = FAMILY_STYLE[family]
        ax.scatter(
            x_res[mask], y_res[mask],
            color=st["color"], marker=st["marker"], s=30,
            label=st["label"], edgecolors="black", linewidths=0.4,
        )

    ancova = ancova_partial_slope(x, y, fams)
    m, b = np.polyfit(x_res, y_res, 1)
    xlim = ax.get_xlim()
    xfit = np.linspace(xlim[0], xlim[1], 200)
    ax.plot(xfit, m * xfit + b, color="gray", lw=2, ls="--", zorder=0)
    ax.set_xlim(xlim)
    ax.text(
        0.05, 0.95,
        f"β = {ancova['slope']:.3f} (p = {ancova['p']:.1g})\n",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
    )
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    return ancova


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
        1, n_panels, figsize=(8, 2.0), sharey=True,
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


def _make_partial_figure(
    models: list[tuple],
    r2: dict,
    exclude_families: set[str],
    modalities: tuple[str, ...],
    col_map: dict[str, int],
    metric_label: str,
    out_name: str,
) -> None:
    kept = [m for m in models if m[0] not in exclude_families]
    n_panels = len(modalities)
    fig, axes = plt.subplots(1, n_panels, figsize=(8, 2.0), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, modality in zip(axes, modalities):
        col = col_map[modality]
        xs, ys, fams = [], [], []
        for m in kept:
            family, size = m[0], m[1]
            val = m[col]
            if val is None:
                continue
            xs.append(float(val))
            ys.append(r2_for_model(r2, family, size))
            fams.append(family)

        is_first = ax is axes[0]
        plot_partial(
            ax,
            np.array(xs), np.array(ys),
            fams,
            xlabel=f"{MODALITY_LABEL[modality]} [$\Delta${metric_label} %]",
            ylabel=(f"$\Delta R^2$" if is_first else ""),
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
        columnspacing=0.55, bbox_to_anchor=(0.52, 1.08),
        handletextpad=0.1, frameon=False,
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
        out_name="crossmodal.pdf",
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
        out_name="crossmodal_cka.pdf",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-json", type=Path, default=DEFAULT_R2_JSON,
                        help="Path to r2_vs_params JSON")
    parser.add_argument("--modalities", nargs="+",
                        default=list(MODALITIES),
                        choices=MODALITIES,
                        help="Subset of modality panels to plot")
    parser.add_argument("--exclude-families", nargs="*", default=["dinov3"],
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
    _make_partial_figure(
        MODELS, r2, exclude, modalities,
        col_map=MKNN_COL, metric_label="MKNN",
        out_name="crossmodal_partial.pdf",
    )
    _make_partial_figure(
        MODELS, r2, exclude, modalities,
        col_map=CKA_COL, metric_label="CKA",
        out_name="crossmodal_cka_partial.pdf",
    )


if __name__ == "__main__":
    main()
