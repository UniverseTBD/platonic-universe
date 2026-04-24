#!/usr/bin/env python3
"""
Cross-architectural MKNN / CKA vs per-property HSC R² — 1×3 appendix figure.

One row, three columns: redshift, log M*, log sSFR.
Same x-axis and model pool as plot_crossarchitectural_ashod.py; this
view just breaks the y-axis back into its three property components
instead of averaging them.
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
from plot_crossarchitectural_ashod import (  # noqa: E402
    DEFAULT_R2_JSON,
    FAMILY_STYLE,
    K_MAIN,
    MODALITIES,
    MODALITY_LABEL,
    MODELS,
    N_SUB,
    R2_PROPS,
    SEED,
    assert_aligned,
    compute_or_load_cka,
    compute_or_load_mknn,
    load_r2_json,
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
            s=45, label=style["label"], edgecolors="black", linewidths=0.3,
        )

    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() >= 3:
        rho, p_rho = spearmanr(x[finite], y[finite])
        ax.text(
            0.97, 0.03,
            f"ρ = {rho:.2f}  (p = {p_rho:.1g})",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=8,
        )

    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")


def _make_figure(
    mean_x: np.ndarray,
    families: list[str],
    models: list[tuple[str, str, Path]],
    r2: dict,
    metric_label: str,
    modality: str,
    out_name: str,
) -> None:
    n_cols = len(R2_PROPS)
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(2.9 * n_cols + 0.6, 2.8),
        sharey=False, squeeze=False,
    )

    x_pct = mean_x * 100.0
    for j, prop in enumerate(R2_PROPS):
        ax = axes[0, j]
        y = np.array([r2_for_model_prop(r2, f, s, prop) for f, s, _ in models])
        plot_panel(ax, x_pct, y, families)

        ax.set_xlabel(
            f"{MODALITY_LABEL[modality]} [cross-arch {metric_label} %]", fontsize=10,
        )
        ax.set_ylabel(rf"{PROP_LABEL[prop]} $[R^2]$ ", fontsize=10)

    seen: dict[str, object] = {}
    for row in axes:
        for ax in row:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                if lab not in seen:
                    seen[lab] = h
    fig.legend(
        seen.values(), list(seen.keys()),
        loc="upper center", fontsize=9, ncol=len(seen),
        columnspacing=0.55,
        bbox_to_anchor=(0.5, 1.02),
        handletextpad=0.1,
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.subplots_adjust(wspace=0.28)
    out = FIGS_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-json", type=Path, default=DEFAULT_R2_JSON)
    parser.add_argument("--n-sub", type=int, default=N_SUB,
                        help="Number of galaxies to subsample")
    parser.add_argument("--recompute", action="store_true",
                        help="Ignore MKNN/CKA caches and recompute them")
    parser.add_argument("--exclude-families", nargs="+", default=["dinov3"],
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

    labels = [f"{f}_{s}" for f, s, _ in models]
    families = [f for f, s, _ in models]

    for modality in MODALITIES:
        mknn_mats = compute_or_load_mknn(models, labels, idx, args.recompute, modality)
        cka_mat = compute_or_load_cka(models, labels, idx, args.recompute, modality)
        x_mknn = np.nanmean(mknn_mats[K_MAIN], axis=1)
        x_cka = np.nanmean(cka_mat, axis=1)
        _make_figure(
            x_mknn, families, models, r2,
            metric_label="MKNN",
            modality=modality,
            out_name=f"crossarchitectural_ashod_per_property_{modality}.pdf",
        )
        _make_figure(
            x_cka, families, models, r2,
            metric_label="CKA",
            modality=modality,
            out_name=f"crossarchitectural_ashod_cka_per_property_{modality}.pdf",
        )


if __name__ == "__main__":
    main()
