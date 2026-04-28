#!/usr/bin/env python3
"""
Plot per-pair intramodal Procrustes distance against both the parameter
count and the mean physics R² of the *larger* model in each pair.

Procrustes distances are read from
``data/procrustes_distances_45000galaxies_upsampled.pkl`` under the
``intra_modal_adjacent`` key, which stores one distance per
(modality, family, property, adjacent-size pair). For each pair we
average the distance across (redshift, mass, sSFR). Parameter counts
come from ``PARAM_COUNTS`` in ``plot_r2_vs_params.py``; R² values come
from ``r2_vs_params_45000galaxies_upsampled.json`` and are averaged over
the same three properties for the larger member of each pair.

Single 2×2 figure: rows are modalities (HSC / JWST), columns are the two
x-axes (parameter count, mean R²). One point per model-pair.
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


def mean_r2_for_larger(r2: dict, modality: str, family: str, large: str) -> float:
    entry = r2[modality][family][large]
    return float(np.mean([entry[p]["r2_mean"] for p in R2_PROPS]))


def params_for_larger(family: str, large: str) -> float:
    return float(PARAM_COUNTS[family][large])


def collect_points(
    procrustes: dict,
    r2: dict,
    modality: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (params, mean_r2, mean_procrustes, families) for one modality."""
    intra = procrustes["intra_modal_adjacent"][modality]
    params, r2s, dists, fams = [], [], [], []
    for family, props in intra.items():
        if family not in FAMILY_STYLE:
            continue
        pair_labels = [lab for lab, _ in props[R2_PROPS[0]]]
        for i, pair_label in enumerate(pair_labels):
            small_raw, large_raw = pair_label.split("→")
            large = normalize_size(family, large_raw)
            if large not in PARAM_COUNTS.get(family, {}):
                continue
            try:
                d = float(np.mean([props[p][i][1] for p in R2_PROPS]))
            except (KeyError, IndexError):
                continue
            params.append(params_for_larger(family, large))
            r2s.append(mean_r2_for_larger(r2, modality, family, large))
            dists.append(d)
            fams.append(family)
    return np.array(params), np.array(r2s), np.array(dists), fams


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
            0.95, -0.01,
            f"ρ = {rho:.3f}  (p = {p_rho:.1g})\n",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
        )

        if log_x:
            log_x_vals = np.log10(x[finite])
            m, b = np.polyfit(log_x_vals, y[finite], 1)
            xfit = np.logspace(log_x_vals.min(), log_x_vals.max(), 200)
            ax.plot(xfit, m * np.log10(xfit) + b,
                    color="gray", lw=1, ls="--", zorder=0)
        else:
            xlim = ax.get_xlim()
            m, b = np.polyfit(x[finite], y[finite], 1)
            xfit = np.linspace(xlim[0], xlim[1], 200)
            ax.plot(xfit, m * xfit + b, color="gray", lw=1, ls="--", zorder=0)
            ax.set_xlim(xlim)

    if log_x:
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
        n_rows, 2, figsize=(7, 2.0 * n_rows), sharey="row",
    )
    if n_rows == 1:
        axes = np.array([axes])

    for row, modality in enumerate(modalities):
        params, r2_vals, dists, fams = collect_points(procrustes, r2, modality)
        keep = np.array([f not in exclude_families for f in fams])
        params, r2_vals, dists = params[keep], r2_vals[keep], dists[keep]
        fams = [f for f, k in zip(fams, keep) if k]

        ax_params, ax_r2 = axes[row]
        plot_scatter(
            ax_params, params, dists, fams,
            xlabel=f"{MODALITY_LABEL[modality]} [Parameters]",
            ylabel="Procrustes distance",
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
        loc="upper center", fontsize=9, ncol=len(seen),
        columnspacing=0.55,
        bbox_to_anchor=(0.52, 1.04),
        handletextpad=0.1,
        frameon=False,
    )

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.45)
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
    parser.add_argument("--exclude-families", nargs="+", default=["dinov3"],
                        metavar="FAMILY",
                        help=f"Family names to drop (valid: "
                             f"{sorted(FAMILY_STYLE)})")
    parser.add_argument("--out-name", default="procrustes.pdf",
                        help="Output filename inside figs/")
    args = parser.parse_args()

    FIGS_DIR.mkdir(exist_ok=True)

    procrustes = load_procrustes(args.procrustes_pkl)
    print(procrustes)
    exit(0)
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
