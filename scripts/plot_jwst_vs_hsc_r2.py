#!/usr/bin/env python3
"""
Plot JWST R² against HSC R² for each model.

Two figures:

* ``jwst_vs_hsc_r2.pdf`` — single panel, mean R² over (M*, sSFR, redshift).
* ``jwst_vs_hsc_r2_grid.pdf`` — 3×3 grid of all cross-property pairs
  (rows = JWST property, columns = HSC property). Diagonals are the
  same-property comparisons; off-diagonals show whether (e.g.) HSC M*
  R² predicts JWST sSFR R².

In every panel one point is one (family, size) model and Spearman's ρ
between the two axes is annotated.
"""

import argparse
import json
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
from plot_crossmodal import FAMILY_STYLE  # noqa: E402

DEFAULT_R2_JSON = ROOT / "r2_vs_params_45000galaxies_upsampled.json"

PROPS = ("mass", "sSFR", "redshift")
PROP_LABEL = {
    "mass":     r"$M_\ast$ [$R^2$]",
    "sSFR":     r"sSFR [$R^2$]",
    "redshift": r"$z$ [$R^2$]",
}


def load_r2(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def collect_property(
    r2: dict,
    hsc_prop: str | None,
    jwst_prop: str | None,
    excluded: set[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (hsc, jwst, families) — one entry per model.

    If ``hsc_prop`` (or ``jwst_prop``) is ``None`` the value is the mean
    of ``r2_mean`` across all three properties for that modality.
    """
    hsc_vals, jwst_vals, fams = [], [], []
    for family in FAMILY_STYLE:
        if family in excluded:
            continue
        if family not in r2.get("hsc", {}) or family not in r2.get("jwst", {}):
            continue
        sizes = sorted(set(r2["hsc"][family]) & set(r2["jwst"][family]))
        for size in sizes:
            try:
                if hsc_prop is None:
                    hsc = float(np.mean([
                        r2["hsc"][family][size][p]["r2_mean"] for p in PROPS
                    ]))
                else:
                    hsc = float(r2["hsc"][family][size][hsc_prop]["r2_mean"])
                if jwst_prop is None:
                    jwst = float(np.mean([
                        r2["jwst"][family][size][p]["r2_mean"] for p in PROPS
                    ]))
                else:
                    jwst = float(r2["jwst"][family][size][jwst_prop]["r2_mean"])
            except KeyError:
                continue
            hsc_vals.append(hsc)
            jwst_vals.append(jwst)
            fams.append(family)
    return np.array(hsc_vals), np.array(jwst_vals), fams


def plot_scatter(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    families: list[str],
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
        ax.text(
            0.05, 0.95,
            f"ρ = {rho:.3f}  (p = {p_rho:.1g})",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
        )

        xlim = ax.get_xlim()
        m, b = np.polyfit(x[finite], y[finite], 1)
        xfit = np.linspace(xlim[0], xlim[1], 200)
        ax.plot(xfit, m * xfit + b, color="gray", lw=2, ls="--", zorder=0)
        ax.set_xlim(xlim)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)


def add_top_legend(fig, axes) -> None:
    seen: dict[str, object] = {}
    for ax in np.atleast_1d(axes).flat:
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in seen:
                seen[lab] = h
    fig.legend(
        seen.values(), list(seen.keys()),
        loc="upper center", fontsize=9, ncol=len(seen)//2,
        columnspacing=0.55,
        bbox_to_anchor=(0.52, 1.08),
        handletextpad=0.1,
        frameon=False,
    )


def make_mean_figure(r2: dict, exclude: set[str], out_name: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(3.0, 2.6))
    hsc, jwst, fams = collect_property(r2, None, None, exclude)
    plot_scatter(
        ax, hsc, jwst, fams,
        xlabel=r"HSC [Mean $R^2$]",
        ylabel=r"JWST [Mean $R^2$]",
    )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", fontsize=9, ncol=max(1, len(labels) // 2),
        columnspacing=0.55,
        bbox_to_anchor=(0.52, 1.18),
        handletextpad=0.1,
        frameon=False,
    )
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")

    fig.tight_layout()
    out = FIGS_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def make_grid_figure(r2: dict, exclude: set[str], out_name: str) -> None:
    n = len(PROPS)
    fig, axes = plt.subplots(n, n, figsize=(6, 5.5), sharex="col", sharey="row")

    for row_idx, jwst_prop in enumerate(PROPS):
        for col_idx, hsc_prop in enumerate(PROPS):
            ax = axes[row_idx, col_idx]
            hsc, jwst, fams = collect_property(r2, hsc_prop, jwst_prop, exclude)
            is_first_col = col_idx == 0
            is_last_row = row_idx == n - 1
            plot_scatter(
                ax, hsc, jwst, fams,
                xlabel=f"HSC {PROP_LABEL[hsc_prop]}" if is_last_row else "",
                ylabel=f"JWST {PROP_LABEL[jwst_prop]}" if is_first_col else "",
            )

    add_top_legend(fig, axes)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.04, hspace=0.04)
    out = FIGS_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-json", type=Path, default=DEFAULT_R2_JSON,
                        help="Path to r2_vs_params JSON")
    parser.add_argument("--exclude-families", nargs="*", default=[],
                        metavar="FAMILY",
                        help=f"Family names to drop (valid: "
                             f"{sorted(FAMILY_STYLE)})")
    parser.add_argument("--out-mean", default="jwst_vs_hsc_r2.pdf",
                        help="Output filename for mean-R² panel")
    parser.add_argument("--out-grid", default="jwst_vs_hsc_r2_grid.pdf",
                        help="Output filename for 3×3 cross-property grid")
    args = parser.parse_args()

    FIGS_DIR.mkdir(exist_ok=True)

    r2 = load_r2(args.r2_json)
    print(f"Loaded R² JSON from {args.r2_json}")
    print(f"Excluded families: {args.exclude_families}")

    exclude = set(args.exclude_families)
    make_mean_figure(r2, exclude, args.out_mean)
    make_grid_figure(r2, exclude, args.out_grid)


if __name__ == "__main__":
    main()
