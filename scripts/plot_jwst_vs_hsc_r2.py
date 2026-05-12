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
import pandas as pd
import pingouin as pg
from scipy.stats import linregress, spearmanr
from sklearn.decomposition import PCA

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

        xlim = ax.get_xlim()
        lr = linregress(x[finite], y[finite])
        m, b, se_m, se_b = lr.slope, lr.intercept, lr.stderr, lr.intercept_stderr
        resid = y[finite] - (m * x[finite] + b)
        rmse = float(np.sqrt(np.mean(resid**2)))
        t = ax.text(
            0.05, 0.95,
            (f"ρ = {rho:.3f} (p = {p_rho:.1g})\n"
             f"β₀ = {b:.2f} ± {se_b:.1g}\n"
             f"β₁ = {m:.1f} ± {se_m:.1g}"),
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
        )
        t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))

        xfit = np.array([xlim[0], xlim[1]])
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


def _orient_pc1(pca: PCA, scores: np.ndarray | None = None) -> np.ndarray | None:
    """Force PC1 sign so that the mean loading is positive ('more is better').

    sklearn's PCA uses ``svd_flip`` (largest absolute loading positive), which is
    a different convention and doesn't guarantee that high R² → high PC1. We want
    the latter so the lollipop and biplot have a consistent direction of meaning.
    """
    if pca.components_[0].mean() < 0:
        pca.components_[0] *= -1
        if scores is not None:
            scores[:, 0] = -scores[:, 0]
    return scores


def _build_model_table(r2: dict, exclude: set[str]) -> pd.DataFrame:
    """One row per (family, size) model with HSC and JWST R² for each property."""
    rows = []
    for family in FAMILY_STYLE:
        if family in exclude:
            continue
        if family not in r2.get("hsc", {}) or family not in r2.get("jwst", {}):
            continue
        sizes = sorted(set(r2["hsc"][family]) & set(r2["jwst"][family]))
        for size in sizes:
            row: dict = {"model": f"{family}/{size}", "family": family}
            try:
                for prop in PROPS:
                    row[f"R2_HSC_{prop}"] = r2["hsc"][family][size][prop]["r2_mean"]
                    row[f"R2_JWST_{prop}"] = r2["jwst"][family][size][prop]["r2_mean"]
            except KeyError:
                continue
            rows.append(row)
    return pd.DataFrame(rows)


def compute_partial_correlations(r2: dict, exclude: set[str]) -> None:
    """Compute and print partial Spearman correlations for the Fig. 2 off-diagonals.

    For each of the five targeted pairs the table reports:
      - raw Spearman rho (matches the annotation in the Fig. 2 scatter panels)
      - partial Spearman rho(x, y | z) via pingouin residualisation
      - p-value for the partial correlation (df = n - 3)
      - 95 % CI via Fisher z transformation (from pingouin)

    Results are reported for the full model set and — as a sensitivity check —
    with DINOv3 excluded, because DINOv3 scales in the opposite direction and
    contributes ~6 clustered points at lower R² that can lever correlations.
    """
    # (x, y, z) → rho(x, y | z)
    tests = [
        ("R2_HSC_sSFR",     "R2_JWST_mass",      "R2_HSC_mass",     "HSC sSFR ↔ JWST M* | HSC M*"),
        ("R2_HSC_mass",     "R2_JWST_sSFR",      "R2_HSC_sSFR",     "HSC M* ↔ JWST sSFR | HSC sSFR"),
        ("R2_HSC_mass",     "R2_JWST_redshift",  "R2_HSC_redshift", "HSC M* ↔ JWST z | HSC z"),
        ("R2_HSC_sSFR",     "R2_JWST_redshift",  "R2_HSC_redshift", "HSC sSFR ↔ JWST z | HSC z"),
        ("R2_HSC_redshift", "R2_JWST_mass",      "R2_HSC_mass",     "HSC z ↔ JWST M* | HSC M*"),
    ]

    for label, subset_exclude in [("all models", exclude),
                                   ("excl. DINOv3", exclude | {"dinov3"})]:
        df = _build_model_table(r2, subset_exclude)
        n = len(df)

        rows = []
        for x, y, z, desc in tests:
            raw = pg.corr(df[x], df[y], method="spearman")
            partial = pg.partial_corr(data=df, x=x, y=y, covar=z, method="spearman")
            ci = partial["CI95"].iloc[0]
            rows.append({
                "Pair (conditioned on)": desc,
                "raw ρ":       f"{raw['r'].iloc[0]:+.3f}",
                "partial ρ":   f"{partial['r'].iloc[0]:+.3f}",
                "p (partial)": f"{partial['p_val'].iloc[0]:.2g}",
                "95 % CI":     f"[{ci[0]:+.3f}, {ci[1]:+.3f}]",
            })

        result_df = pd.DataFrame(rows)
        print(f"\n--- Partial Spearman correlations ({label}, n={n}) ---")
        print(result_df.to_string(index=False))

    # Complementary PCA: fraction of variance in the six R² columns explained
    # by the first principal component (general model-quality axis).
    df_all = _build_model_table(r2, exclude)
    r2_cols = [f"R2_{mod}_{prop}" for mod in ("HSC", "JWST") for prop in PROPS]
    pca = PCA(n_components=len(r2_cols))
    pca.fit(df_all[r2_cols].values)
    _orient_pc1(pca)

    print(f"\n--- PCA on all six R² columns (n={len(df_all)}) ---")
    for i, ve in enumerate(pca.explained_variance_ratio_, 1):
        print(f"  PC{i}: {ve:.1%}")
    print(f"  PC1 alone explains {pca.explained_variance_ratio_[0]:.1%} of total variance across all R² columns.")

    # Loadings: rows = PCs, columns = original variables. The sign pattern within
    # each row tells you what that PC encodes (e.g. all-positive = quality axis,
    # HSC-vs-JWST sign split = modality axis).
    ld = pd.DataFrame(pca.components_, columns=r2_cols,
                      index=[f"PC{i+1}" for i in range(pca.n_components_)])
    ld.insert(0, "var. (%)",
              [f"{v*100:5.1f}" for v in pca.explained_variance_ratio_])
    with pd.option_context("display.float_format", "{:+.3f}".format):
        print("\n--- PCA loadings (rows = PC, cols = variable) ---")
        print(ld.to_string())


def plot_pc1_scores(r2: dict, exclude: set[str], out_name: str) -> None:
    """Horizontal lollipop chart of PC1 scores (general model-quality axis)."""
    df = _build_model_table(r2, exclude)
    r2_cols = [f"R2_{mod}_{prop}" for mod in ("HSC", "JWST") for prop in PROPS]

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(df[r2_cols].values)
    _orient_pc1(pca, pc1)
    pc1 = pc1.ravel()

    df = df.copy()
    df["pc1"] = pc1
    df = df.sort_values("pc1", ascending=True).reset_index(drop=True)

    # readable y-tick labels: "Family size"
    labels = [
        f"{FAMILY_STYLE[row.family]['label']} {row.model.split('/')[1]}"
        for row in df.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(4.5, 0.35 * len(df) + 0.8))

    for i, row in df.iterrows():
        style = FAMILY_STYLE[row["family"]]
        ax.plot([0, row["pc1"]], [i, i], color=style["color"], lw=1.2, zorder=1)
        ax.scatter(row["pc1"], i,
                   color=style["color"], marker=style["marker"],
                   s=45, edgecolors="black", linewidths=0.4, zorder=2,
                   label=style["label"])

    ax.axvline(0, color="black", lw=0.8, ls="--", zorder=0)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("PC1 score (general model-quality axis)", fontsize=10)
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", length=0)

    # deduplicated legend
    seen: dict[str, object] = {}
    for h, lab in zip(*ax.get_legend_handles_labels()):
        if lab not in seen:
            seen[lab] = h
    ax.legend(seen.values(), list(seen.keys()),
              loc="lower right", fontsize=7, frameon=False,
              handletextpad=0.2, labelspacing=0.3)

    fig.tight_layout()
    out = FIGS_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


LOADING_LABEL = {
    "R2_HSC_mass":     r"HSC $M_\ast$",
    "R2_HSC_sSFR":     r"HSC sSFR",
    "R2_HSC_redshift": r"HSC $z$",
    "R2_JWST_mass":    r"JWST $M_\ast$",
    "R2_JWST_sSFR":    r"JWST sSFR",
    "R2_JWST_redshift": r"JWST $z$",
}
# HSC arrows in blue, JWST arrows in red so the modality split is immediately visible
MODALITY_COLOR = {"HSC": "#1f77b4", "JWST": "#d62728"}


def plot_pca_biplot(r2: dict, exclude: set[str], out_name: str) -> None:
    """PC1 vs PC2 biplot: model scores colored by family, loading arrows by modality.

    If PC2 separates HSC from JWST probes (arrows of each modality point opposite
    directions) it is a modality axis. If model families cluster in the score space
    it is a model-type axis. Empirically PC2 here is *not* a clean modality axis —
    the loading printout in ``compute_partial_correlations`` shows the actual
    structure across all six PCs.
    """
    df = _build_model_table(r2, exclude)
    r2_cols = [f"R2_{mod}_{prop}" for mod in ("HSC", "JWST") for prop in PROPS]

    pca = PCA(n_components=2)
    scores = pca.fit_transform(df[r2_cols].values)   # (n_models, 2)
    _orient_pc1(pca, scores)
    loadings = pca.components_.T                      # (6, 2) — one row per variable

    score_radius = np.max(np.abs(scores))
    loading_radius = np.max(np.linalg.norm(loadings, axis=1))
    arrow_scale = 0.85 * score_radius / loading_radius
    tips = loadings * arrow_scale                     # (6, 2)

    # Group near-identical loadings so we draw one arrow + one combined label
    # per cluster (HSC sSFR and HSC z load almost identically here).
    overlap_tol = 0.02 * score_radius
    used = [False] * len(r2_cols)
    arrow_groups: list[list[int]] = []
    for j in range(len(r2_cols)):
        if used[j]:
            continue
        group = [j]
        used[j] = True
        for k in range(j + 1, len(r2_cols)):
            if used[k]:
                continue
            if np.linalg.norm(tips[j] - tips[k]) < overlap_tol:
                group.append(k)
                used[k] = True
        arrow_groups.append(group)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))

    # ── model scores ──────────────────────────────────────────────────────────
    for family in FAMILY_STYLE:
        mask = df["family"].values == family
        if not mask.any():
            continue
        style = FAMILY_STYLE[family]
        ax.scatter(scores[mask, 0], scores[mask, 1],
                   color=style["color"], marker=style["marker"],
                   s=55, edgecolors="black", linewidths=0.4,
                   label=style["label"], zorder=3)

    # ── loading arrows + labels ───────────────────────────────────────────────
    label_positions: list[tuple[float, float]] = []
    label_pad = 0.04 * score_radius
    for group in arrow_groups:
        tip = np.mean(tips[group], axis=0)
        modalities = {"JWST" if "JWST" in r2_cols[i] else "HSC" for i in group}
        color = MODALITY_COLOR[next(iter(modalities))] if len(modalities) == 1 else "purple"
        ax.annotate(
            "", xy=tuple(tip), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                            mutation_scale=10),
            zorder=4,
        )
        angle = np.arctan2(tip[1], tip[0])
        lx = tip[0] + label_pad * np.cos(angle)
        ly = tip[1] + label_pad * np.sin(angle)
        text = " / ".join(LOADING_LABEL[r2_cols[i]] for i in group)
        ax.text(lx, ly, text, color=color, fontsize=8,
                ha="center", va="center", zorder=5)
        label_positions.append((lx, ly))

    # ── axis limits: include scores AND label positions, with padding ─────────
    label_xy = np.array(label_positions)
    all_x = np.concatenate([scores[:, 0], label_xy[:, 0]])
    all_y = np.concatenate([scores[:, 1], label_xy[:, 1]])
    pad_x = 0.08 * (all_x.max() - all_x.min())
    pad_y = 0.08 * (all_y.max() - all_y.min())
    ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)

    ax.axhline(0, color="gray", lw=0.6, ls="--", zorder=0)
    ax.axvline(0, color="gray", lw=0.6, ls="--", zorder=0)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)", fontsize=10)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)", fontsize=10)
    ax.tick_params(direction="in")

    # ── modality legend (loading arrow colors) ────────────────────────────────
    from matplotlib.lines import Line2D
    modality_handles = [
        Line2D([0], [0], color=MODALITY_COLOR["HSC"], lw=2, label="HSC probe"),
        Line2D([0], [0], color=MODALITY_COLOR["JWST"], lw=2, label="JWST probe"),
    ]
    modality_legend = ax.legend(
        handles=modality_handles, fontsize=7, frameon=False,
        loc="upper left", title="Loadings", title_fontsize=7,
        handletextpad=0.4, labelspacing=0.2,
    )
    ax.add_artist(modality_legend)

    # ── family legend (score markers) ─────────────────────────────────────────
    seen: dict[str, object] = {}
    for h, lab in zip(*ax.get_legend_handles_labels()):
        if lab not in seen:
            seen[lab] = h
    ax.legend(seen.values(), list(seen.keys()),
              fontsize=7, frameon=False, loc="lower left",
              handletextpad=0.2, labelspacing=0.3,
              title="Family", title_fontsize=7)

    fig.tight_layout()
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
    parser.add_argument("--out-pc1", default="pc1_scores.pdf",
                        help="Output filename for PC1 lollipop chart")
    parser.add_argument("--out-biplot", default="pca_biplot.pdf",
                        help="Output filename for PC1 vs PC2 biplot")
    args = parser.parse_args()

    FIGS_DIR.mkdir(exist_ok=True)

    r2 = load_r2(args.r2_json)
    print(f"Loaded R² JSON from {args.r2_json}")
    print(f"Excluded families: {args.exclude_families}")

    exclude = set(args.exclude_families)
    make_mean_figure(r2, exclude, args.out_mean)
    make_grid_figure(r2, exclude, args.out_grid)
    plot_pc1_scores(r2, exclude, args.out_pc1)
    plot_pca_biplot(r2, exclude, args.out_biplot)
    compute_partial_correlations(r2, exclude)


if __name__ == "__main__":
    main()
