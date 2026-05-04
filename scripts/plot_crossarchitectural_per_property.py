#!/usr/bin/env python3
"""
Cross-architectural MKNN / CKA vs per-property HSC R² — appendix figure.

Rows: physics property (redshift, mass, sSFR).
Cols: modality (HSC, JWST).
One point per (family, size); x is the cross-architectural metric mean
under the column's modality, y is R² of the row's property.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
FIGS_DIR = ROOT / "figs"

sys.path.insert(0, str(SCRIPTS_DIR))
from plot_crossarchitectural import (  # noqa: E402
    DEFAULT_R2_JSON,
    FAMILY_STYLE,
    METHODS,
    MODALITIES,
    MODELS,
    N_SUB,
    R2_PROPS,
    SEED,
    assert_aligned,
    compute_or_load_matrices,
    load_r2_json,
    method_suffix,
    metric_axis_label,
    resolve_models,
    r2_for_model_prop,
)

PROP_LABEL = {
    "redshift": "redshift",
    "mass": r"$\log M\star$",
    "sSFR": r"$\log\,\mathrm{sSFR}$",
}


def plot_panel(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    families: list[str],
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

    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="x", direction="in", which="minor")
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="y", direction="in", which="minor")


def _make_figure(
    x_per_modality: dict[str, np.ndarray],
    families: list[str],
    models: list[tuple[str, str, Path]],
    r2: dict,
    metric: str,
    method: str,
    out_name: str,
) -> None:
    modalities = list(x_per_modality)
    n_rows = len(R2_PROPS)
    n_cols = len(modalities)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6, 6),
        sharex="col", 
        sharey="row", 
        squeeze=False,
    )

    scale = 100.0 if method == "compare" else 1.0
    for i, prop in enumerate(R2_PROPS):
        y = np.array([r2_for_model_prop(r2, f, s, prop) for f, s, _ in models])
        for j, modality in enumerate(modalities):
            ax = axes[i, j]
            x_scaled = x_per_modality[modality] * scale
            plot_panel(ax, x_scaled, y, families)

            if i == n_rows - 1:
                ax.set_xlabel(
                    metric_axis_label(modality, metric, method),
                    fontsize=10,
                )
            if j == 0:
                ax.set_ylabel(
                    rf"{PROP_LABEL[prop]} $[R^2]$ ", fontsize=10,
                )

    seen: dict[str, object] = {}
    for row in axes:
        for ax in row:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                if lab not in seen:
                    seen[lab] = h
    fig.legend(
        seen.values(), list(seen.keys()),
        loc="upper center", fontsize=9, ncol=(len(seen)//2 + 1),
        columnspacing=0.55,
        bbox_to_anchor=(0.52, 1.06),
        handletextpad=0.1,
        frameon=False,
    )

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.08)
    out = FIGS_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-json", type=Path, default=DEFAULT_R2_JSON)
    parser.add_argument("--n-sub", type=int, default=N_SUB,
                        help="Number of galaxies to subsample")
    parser.add_argument("--method", choices=METHODS, default="compare",
                        help="Metric backend: raw (compare) or "
                             "permutation-calibrated (calibrate)")
    parser.add_argument("--recompute", action="store_true",
                        help="Ignore cached matrices and recompute them")
    parser.add_argument("--exclude-families", nargs="*", default=["dinov3"],
                        metavar="FAMILY",
                        help=f"Family names to drop (valid: "
                             f"{sorted(FAMILY_STYLE)})")
    args = parser.parse_args()

    FIGS_DIR.mkdir(exist_ok=True)

    excluded = set(args.exclude_families)
    unknown = excluded - set(FAMILY_STYLE)
    if unknown:
        raise ValueError(
            f"Unknown family names in --exclude-families: {sorted(unknown)}. "
            f"Valid: {sorted(FAMILY_STYLE)}"
        )

    r2 = load_r2_json(args.r2_json)

    models = resolve_models(MODELS, r2, excluded=excluded)

    n_full = assert_aligned(models)
    n_sub = min(args.n_sub, n_full)
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(n_full, size=n_sub, replace=False))
    print(f"Subsampling {n_sub} rows (seed={SEED}).")
    print(f"Metric method: {args.method}")

    labels = [f"{f}_{s}" for f, s, _ in models]
    families = [f for f, s, _ in models]

    suffix = method_suffix(args.method)

    x_mknn_per_modality: dict[str, np.ndarray] = {}
    x_cka_per_modality: dict[str, np.ndarray] = {}
    for modality in MODALITIES:
        mats = compute_or_load_matrices(
            models, labels, idx, args.recompute, modality, args.method,
        )
        x_mknn_per_modality[modality] = np.nanmean(mats["mknn"], axis=1)
        x_cka_per_modality[modality] = np.nanmean(mats["cka"], axis=1)

    _make_figure(
        x_mknn_per_modality, families, models, r2,
        metric="mknn", method=args.method,
        out_name=f"crossarchitectural_per_property{suffix}.pdf",
    )
    _make_figure(
        x_cka_per_modality, families, models, r2,
        metric="cka", method=args.method,
        out_name=f"crossarchitectural_cka_per_property{suffix}.pdf",
    )


if __name__ == "__main__":
    main()
