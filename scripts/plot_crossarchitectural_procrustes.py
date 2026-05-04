#!/usr/bin/env python3
"""
Plot per-model cross-architectural Procrustes distance against both the
parameter count and the HSC mean physics R².

For each model we compute its mean pairwise Procrustes distance to every
other model in the cross-family pool — its degree of geometric divergence
from the cross-architecture consensus — and plot that against its
parameter count (log) and its HSC mean R² across (redshift, mass, sSFR).

Procrustes distances are read from
``data/procrustes_distances_45000galaxies_upsampled.pkl``. The flat list
under ``cross_model_pairwise[modality][property]`` is the row-major
upper-triangular flattening of the 39-model order documented in
``cross_modal_hsc_vs_jwst[property]``; we reconstruct the full M×M
matrix, drop excluded / un-styled families, average over
(redshift, mass, sSFR), and take a per-row mean for each surviving model.

Single 2×2 figure: rows are modalities (HSC / JWST), columns are the two
x-axes (parameter count, mean R²). One point per model.

Complements ``plot_crossarchitectural.py`` (cross-arch MKNN / CKA) and
``plot_intramodal_procrustes.py`` (within-family adjacent Procrustes).
"""

import argparse
import json
import pickle
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
from plot_intramodal import FAMILY_STYLE  # noqa: E402
from plot_r2_vs_params import PARAM_COUNTS  # noqa: E402

DEFAULT_PROCRUSTES_PKL = (
    ROOT / "data" / "procrustes_distances_45000galaxies_upsampled.pkl"
)
DEFAULT_R2_JSON = ROOT / "r2_vs_params_45000galaxies_upsampled.json"
R2_PROPS = ("redshift", "mass", "sSFR")

MODALITIES = ("hsc", "jwst")
MODALITY_LABEL = {"hsc": "HSC", "jwst": "JWST"}


def normalize_size(family: str, size: str) -> str:
    """Map procrustes-pickle size labels to PARAM_COUNTS / R² JSON keys."""
    if family == "astropt":
        # "015M" -> "15m", "095M" -> "95m", "850M" -> "850m"
        s = size.lower().rstrip("m").lstrip("0") or "0"
        return s + "m"
    return size


def load_procrustes(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_r2(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def model_order(procrustes: dict, prop: str) -> list[tuple[str, str]]:
    """(family, raw_size) order used when flattening the pairwise matrix."""
    return [(fam, sz) for fam, sz, *_ in procrustes["cross_modal_hsc_vs_jwst"][prop]]


def reconstruct_pairwise(flat: list[float], n: int) -> np.ndarray:
    """Inflate a row-major upper-triangular flattening (i<j) to a full M×M."""
    expected = n * (n - 1) // 2
    if len(flat) != expected:
        raise ValueError(
            f"Pairwise flat list has {len(flat)} entries; expected {expected} "
            f"for n={n}"
        )
    mat = np.zeros((n, n), dtype=np.float64)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            v = float(flat[k])
            mat[i, j] = v
            mat[j, i] = v
            k += 1
    return mat


def collect_points(
    procrustes: dict,
    r2: dict,
    modality: str,
    excluded: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (params, mean_r2, mean_procrustes, families) for one modality."""
    order = model_order(procrustes, R2_PROPS[0])
    n = len(order)

    # Mean over (redshift, mass, sSFR) — one M×M matrix per modality.
    stack = np.stack([
        reconstruct_pairwise(procrustes["cross_model_pairwise"][modality][p], n)
        for p in R2_PROPS
    ])
    mat = np.nanmean(stack, axis=0)

    keep_idx, params, r2s, fams = [], [], [], []
    for i, (family, raw_size) in enumerate(order):
        if family in excluded or family not in FAMILY_STYLE:
            continue
        size = normalize_size(family, raw_size)
        if size not in PARAM_COUNTS.get(family, {}):
            continue
        try:
            r2_val = float(np.mean([
                r2[modality][family][size][p]["r2_mean"] for p in R2_PROPS
            ]))
        except KeyError:
            continue
        keep_idx.append(i)
        params.append(float(PARAM_COUNTS[family][size]))
        r2s.append(r2_val)
        fams.append(family)

    sub = mat[np.ix_(keep_idx, keep_idx)]
    np.fill_diagonal(sub, np.nan)
    dists = np.nanmean(sub, axis=1)
    return np.array(params), np.array(r2s), dists, fams


def plot_scatter(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    families: list[str],
    xlabel: str,
    ylabel: str,
    log_x: bool,
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
            0.95, 0.75,
            f"ρ = {rho:.3f}  (p = {p_rho:.1g})\n",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
        )

        if log_x:
            ax.set_xscale("log")
            xlim = ax.get_xlim()
            m, b = np.polyfit(np.log10(x[finite]), y[finite], 1)
            xfit = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 200)
            ax.plot(xfit, m * np.log10(xfit) + b,
                    color="gray", lw=2, ls="--", zorder=0)
            ax.set_xlim(xlim)
        else:
            xlim = ax.get_xlim()
            m, b = np.polyfit(x[finite], y[finite], 1)
            xfit = np.linspace(xlim[0], xlim[1], 200)
            ax.plot(xfit, m * xfit + b, color="gray", lw=2, ls="--", zorder=0)
            ax.set_xlim(xlim)
    elif log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)


def make_figure(
    procrustes: dict,
    r2: dict,
    exclude_families: set[str],
    modalities: tuple[str, ...],
    out_name: str,
) -> None:
    unknown = exclude_families - set(FAMILY_STYLE)
    if unknown:
        raise ValueError(
            f"Unknown family names in --exclude-families: {sorted(unknown)}. "
            f"Valid: {sorted(FAMILY_STYLE)}"
        )

    n_rows = len(modalities)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(4.5, 2.0 * n_rows), sharey="row"
    )
    if n_rows == 1:
        axes = np.array([axes])

    for row, modality in enumerate(modalities):
        params, r2_vals, dists, fams = collect_points(
            procrustes, r2, modality, exclude_families,
        )

        ax_params, ax_r2 = axes[row]
        plot_scatter(
            ax_params, params, dists, fams,
            xlabel=f"{MODALITY_LABEL[modality]} [Parameters]",
            ylabel="$d_{proc}$",
            log_x=True,
        )
        plot_scatter(
            ax_r2, r2_vals, dists, fams,
            xlabel=f"{MODALITY_LABEL[modality]} [Mean $R^2$]",
            ylabel="",
            log_x=False,
        )

    seen: dict[str, object] = {}
    for ax in axes.ravel():
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in seen:
                seen[lab] = h
    fig.legend(
        seen.values(), list(seen.keys()),
        loc="upper center", fontsize=9, ncol=len(seen)//2,
        columnspacing=0.55,
        bbox_to_anchor=(0.52, 1.1),
        handletextpad=0.1,
        frameon=False,
    )

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.35)
    out = FIGS_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--procrustes-pkl", type=Path,
                        default=DEFAULT_PROCRUSTES_PKL,
                        help="Path to procrustes distances pickle")
    parser.add_argument("--r2-json", type=Path, default=DEFAULT_R2_JSON,
                        help="Path to r2_vs_params JSON")
    parser.add_argument("--modalities", nargs="+", default=list(MODALITIES),
                        choices=MODALITIES,
                        help="Subset of modality rows to plot")
    parser.add_argument("--exclude-families", nargs="*", default=["dinov3"],
                        metavar="FAMILY",
                        help=f"Family names to drop (valid: "
                             f"{sorted(FAMILY_STYLE)})")
    parser.add_argument("--out-name", default="crossarchitectural_procrustes.pdf",
                        help="Output filename inside figs/")
    args = parser.parse_args()

    FIGS_DIR.mkdir(exist_ok=True)

    procrustes = load_procrustes(args.procrustes_pkl)
    r2 = load_r2(args.r2_json)
    print(f"Loaded procrustes pickle from {args.procrustes_pkl}")
    print(f"Loaded R² JSON from {args.r2_json}")
    print(f"Rows (modalities): {args.modalities}")
    print(f"Excluded families: {args.exclude_families}")

    make_figure(
        procrustes, r2,
        exclude_families=set(args.exclude_families),
        modalities=tuple(args.modalities),
        out_name=args.out_name,
    )


if __name__ == "__main__":
    main()
