#!/usr/bin/env python3
"""
Plot MKNN embedding similarity (mean MKNN vs every other model) against
physics R² (mean over redshift, mass, sSFR) taken from the precomputed
table at ``r2_vs_params_45000galaxies_upsampled.json``.

One point per model. Tests the PRH prediction that models whose
neighborhood structure is closer to the consensus also encode more physics.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
DATA_DIR = ROOT / "data"
FIGS_DIR = ROOT / "figs"
PHYSICS_DIR = DATA_DIR / "physics"

sys.path.insert(0, str(SCRIPTS_DIR))
from plot_r2_vs_params import (  # noqa: E402
    FAMILY_STYLE,
    PARAM_COUNTS,
)

N_SUB = 10_000
K_VALUES = (5, 10, 20)
K_MAIN = 10
SEED = 0

MKNN_CACHE = DATA_DIR / "mknn_matrix.parquet"
DEFAULT_R2_JSON = ROOT / "r2_vs_params_45000galaxies_upsampled.json"
R2_PROPS = ("redshift", "mass", "sSFR")


def parse_filename(path: Path) -> tuple[str, str] | None:
    stem = path.stem
    if not stem.endswith("_test"):
        return None
    stem = stem[: -len("_test")]
    family, _, size = stem.rpartition("_")
    if not family or family not in PARAM_COUNTS:
        return None
    if size not in PARAM_COUNTS[family]:
        return None
    return family, size


def discover_models(
    json_families: dict[str, set[str]] | None = None,
) -> list[tuple[str, str, Path]]:
    found: dict[tuple[str, str], Path] = {}
    for p in sorted(PHYSICS_DIR.glob("*_test.parquet")):
        parsed = parse_filename(p)
        if parsed:
            found[parsed] = p
    ordered: list[tuple[str, str, Path]] = []
    dropped: list[tuple[str, str]] = []
    for family, sizes in PARAM_COUNTS.items():
        for size in sizes:
            if (family, size) not in found:
                continue
            if json_families is not None:
                if family not in json_families or size not in json_families[family]:
                    dropped.append((family, size))
                    continue
            ordered.append((family, size, found[(family, size)]))
    if dropped:
        print("Dropping models missing from R² JSON:")
        for fam, sz in dropped:
            print(f"  {fam}_{sz}")
    return ordered


def load_embeddings(family: str, size: str, path: Path) -> np.ndarray:
    df = pl.read_parquet(path)
    col = f"{family}_{size}_galaxies"
    return np.stack(df[col].to_list()).astype(np.float32, copy=False)


def unit_normalize(Z: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return Z / norms


def compute_knn_indices(Z: np.ndarray, k: int) -> np.ndarray:
    # Cosine NN on Z == Euclidean NN on L2-normalised Z (same ordering),
    # and the latter is markedly faster via BLAS.
    Zn = unit_normalize(Z)
    return (
        NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="brute")
        .fit(Zn)
        .kneighbors(return_distance=False)
    )


def mknn_from_indices(nn1: np.ndarray, nn2: np.ndarray, k: int) -> float:
    a = nn1[:, :k]
    b = nn2[:, :k]
    overlaps = np.fromiter(
        (len(set(ai).intersection(bi)) for ai, bi in zip(a, b)),
        dtype=np.int32,
        count=a.shape[0],
    )
    return float(overlaps.mean() / k)


def assert_aligned(models: list[tuple[str, str, Path]]) -> int:
    lengths = {}
    for family, size, path in models:
        df = pl.read_parquet(path, columns=[f"{family}_{size}_galaxies"])
        lengths[(family, size)] = df.height
    n_unique = set(lengths.values())
    if len(n_unique) != 1:
        msg = "Parquets have different row counts:\n" + "\n".join(
            f"  {fam}_{sz}: {n}" for (fam, sz), n in lengths.items()
        )
        raise RuntimeError(msg)
    return n_unique.pop()


def load_r2_json(json_path: Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


def r2_families_for_modality(r2: dict, modality: str) -> dict[str, set[str]]:
    if modality not in r2:
        raise KeyError(
            f"Modality {modality!r} not in R² JSON (have: {sorted(r2)})"
        )
    return {fam: set(sizes) for fam, sizes in r2[modality].items()}


def build_r2_df(
    r2: dict,
    modality: str,
    models: list[tuple[str, str, Path]],
) -> pl.DataFrame:
    """Build a DataFrame of mean R² per model from the r2_vs_params JSON.

    Averages ``r2_mean`` across ``R2_PROPS`` for each (family, size).
    """
    rows = []
    block = r2[modality]
    for family, size, _ in models:
        try:
            entry = block[family][size]
        except KeyError as e:
            raise KeyError(
                f"{family}/{size} missing under modality {modality!r}"
            ) from e
        per_prop = {}
        for prop in R2_PROPS:
            if prop not in entry:
                raise KeyError(
                    f"{family}/{size} missing property {prop!r} "
                    f"under modality {modality!r}"
                )
            per_prop[prop] = float(entry[prop]["r2_mean"])
        mean_r2 = float(np.mean(list(per_prop.values())))
        row = {
            "family": family,
            "size": size,
            "mean_r2": mean_r2,
            "n_props": len(per_prop),
        }
        row.update({f"r2_{p}": v for p, v in per_prop.items()})
        rows.append(row)
        print(f"  {family:<10} {size:<12} mean R² = {mean_r2:.4f}")
    return pl.DataFrame(rows)


def mknn_matrix_to_df(
    mats: dict[int, np.ndarray], labels: list[str]
) -> pl.DataFrame:
    rows = []
    M = len(labels)
    for k, mat in mats.items():
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                rows.append({"k": k, "model_i": labels[i], "model_j": labels[j],
                             "mknn": mat[i, j]})
    return pl.DataFrame(rows)


def df_to_mknn_matrix(df: pl.DataFrame, labels: list[str]) -> dict[int, np.ndarray]:
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    M = len(labels)
    mats: dict[int, np.ndarray] = {}
    for k in sorted(df["k"].unique().to_list()):
        sub = df.filter(pl.col("k") == k)
        mat = np.full((M, M), np.nan, dtype=np.float64)
        for row in sub.iter_rows(named=True):
            i = label_to_idx.get(row["model_i"])
            j = label_to_idx.get(row["model_j"])
            if i is None or j is None:
                continue
            mat[i, j] = row["mknn"]
        mats[int(k)] = mat
    return mats


def compute_mknn_matrix(
    models: list[tuple[str, str, Path]],
    idx: np.ndarray,
) -> dict[int, np.ndarray]:
    """Compute symmetric MKNN matrices for each k in K_VALUES.

    We fit kNN once per model at k_max = max(K_VALUES) and slice for smaller k.
    """
    k_max = max(K_VALUES)
    print(f"Computing kNN (k={k_max}) for {len(models)} models on {idx.size} samples")
    nn_indices: list[np.ndarray] = []
    for family, size, path in tqdm(models, desc="kNN per model"):
        Z = load_embeddings(family, size, path)[idx]
        nn_indices.append(compute_knn_indices(Z, k_max))

    M = len(models)
    mats = {k: np.full((M, M), np.nan, dtype=np.float64) for k in K_VALUES}
    pairs = [(i, j) for i in range(M) for j in range(i + 1, M)]
    print(f"Computing MKNN over {len(pairs)} model pairs")
    for i, j in tqdm(pairs, desc="Pairwise MKNN"):
        for k in K_VALUES:
            v = mknn_from_indices(nn_indices[i], nn_indices[j], k)
            mats[k][i, j] = v
            mats[k][j, i] = v
    return mats


def plot_scatter(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    families: list[str],
    sizes: list[str],
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    for family in PARAM_COUNTS:
        mask = np.array([f == family for f in families])
        if not mask.any():
            continue
        style = FAMILY_STYLE[family]
        ax.scatter(
            x[mask], y[mask],
            color=style["color"], marker=style["marker"],
            s=70, label=style["label"], edgecolors="black", linewidths=0.4,
        )
        for xi, yi, sz in zip(x[mask], y[mask], np.array(sizes)[mask]):
            ax.annotate(
                sz, (xi, yi), xytext=(4, 2), textcoords="offset points",
                fontsize=6, color=style["color"],
            )

    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() >= 3:
        rho, p_rho = spearmanr(x[finite], y[finite])
        r, p_r = pearsonr(x[finite], y[finite])
        ax.text(
            0.02, 0.98,
            f"Spearman ρ = {rho:.3f}  (p = {p_rho:.2g})\n"
            f"Pearson  r = {r:.3f}  (p = {p_r:.2g})\n"
            f"n = {int(finite.sum())} models",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
        )

    if xlabel is None:
        xlabel = f"Mean MKNN to other models (cosine, k={K_MAIN})"
    if ylabel is None:
        ylabel = "Mean $R^2$ (redshift, mass, sSFR — 45k upsampled)"
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)


def make_main_figure(
    mknn_mat: np.ndarray,
    r2_df: pl.DataFrame,
    families: list[str],
    sizes: list[str],
    modality: str,
) -> None:
    x = np.nanmean(mknn_mat, axis=1)
    y = r2_df["mean_r2"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_scatter(
        ax, x, y, families, sizes,
        title=f"Representational convergence vs physics R² "
              f"(MKNN k={K_MAIN}, {modality.upper()}, R² from 45k upsampled JSON)",
    )
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    fig.tight_layout()
    out = FIGS_DIR / f"mknn_vs_r2_{modality}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def make_per_property_figure(
    mknn_mat: np.ndarray,
    r2_df: pl.DataFrame,
    families: list[str],
    sizes: list[str],
    modality: str,
) -> None:
    x = np.nanmean(mknn_mat, axis=1)
    fig, axes = plt.subplots(1, len(R2_PROPS), figsize=(6 * len(R2_PROPS), 5), sharex=True)
    if len(R2_PROPS) == 1:
        axes = [axes]
    for ax, prop in zip(axes, R2_PROPS):
        y = r2_df[f"r2_{prop}"].to_numpy()
        plot_scatter(
            ax, x, y, families, sizes,
            title=prop,
            ylabel=f"$R^2$ ({prop}) — 45k upsampled",
        )
    axes[0].legend(fontsize=8, loc="lower right", ncol=2)
    fig.suptitle(
        f"MKNN (k={K_MAIN}) vs per-property R² "
        f"({modality.upper()}, R² from 45k upsampled JSON)",
        fontsize=13,
    )
    fig.tight_layout()
    out = FIGS_DIR / f"mknn_vs_r2_{modality}_per_property.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-alignment", action="store_true",
                        help="Only verify all parquets have the same row count and exit")
    parser.add_argument("--n-sub", type=int, default=N_SUB,
                        help="Number of galaxies to subsample for MKNN")
    parser.add_argument("--recompute", action="store_true",
                        help="Ignore MKNN cache and recompute it")
    parser.add_argument("--modality", choices=("hsc", "jwst"), default="hsc",
                        help="Which modality block of the R² JSON to use")
    parser.add_argument("--r2-json", type=Path, default=DEFAULT_R2_JSON,
                        help="Path to r2_vs_params JSON")
    parser.add_argument("--exclude-families", nargs="+", default=[],
                        metavar="FAMILY",
                        help="Family names to drop from the plot (e.g. dinov3)")
    args = parser.parse_args()

    FIGS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    r2 = load_r2_json(args.r2_json)
    json_families = r2_families_for_modality(r2, args.modality)
    print(f"R² JSON: {args.r2_json}")
    print(f"Modality: {args.modality}  "
          f"({sum(len(s) for s in json_families.values())} models across "
          f"{len(json_families)} families)")

    models = discover_models(json_families=json_families)
    if args.exclude_families:
        excluded = set(args.exclude_families)
        unknown = excluded - set(PARAM_COUNTS)
        if unknown:
            raise ValueError(
                f"Unknown family names in --exclude-families: {sorted(unknown)}. "
                f"Valid: {sorted(PARAM_COUNTS)}"
            )
        before = len(models)
        models = [m for m in models if m[0] not in excluded]
        print(f"Excluded families {sorted(excluded)}: "
              f"{before} -> {len(models)} models")
    if not models:
        raise RuntimeError(
            f"No physics parquets (intersected with R² JSON) found under {PHYSICS_DIR}"
        )
    print(f"Discovered {len(models)} models under {PHYSICS_DIR}:")
    for family, size, path in models:
        print(f"  {family}_{size}  ({path.name})")

    n_full = assert_aligned(models)
    print(f"\nAll parquets aligned at {n_full} rows.")
    if args.check_alignment:
        return

    n_sub = min(args.n_sub, n_full)
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(n_full, size=n_sub, replace=False))
    print(f"Subsampling {n_sub} rows (seed={SEED}).")

    labels = [f"{f}_{s}" for f, s, _ in models]
    families = [f for f, _, _ in models]
    sizes = [s for _, s, _ in models]

    if MKNN_CACHE.exists() and not args.recompute:
        print(f"Loading cached MKNN matrix from {MKNN_CACHE}")
        mats = df_to_mknn_matrix(pl.read_parquet(MKNN_CACHE), labels)
        missing_ks = [k for k in K_VALUES if k not in mats]
        missing_models = any(np.all(np.isnan(mats[k]), axis=0).any() for k in mats)
        if missing_ks or missing_models:
            print(f"Cache incomplete (missing_ks={missing_ks}, "
                  f"missing_models={missing_models}); recomputing MKNN")
            mats = compute_mknn_matrix(models, idx)
            mknn_matrix_to_df(mats, labels).write_parquet(MKNN_CACHE)
    else:
        mats = compute_mknn_matrix(models, idx)
        mknn_matrix_to_df(mats, labels).write_parquet(MKNN_CACHE)
        print(f"Saved MKNN matrix to {MKNN_CACHE}")

    print(f"\nBuilding R² table from {args.r2_json} (modality={args.modality}):")
    r2_df = build_r2_df(r2, args.modality, models)

    # Align r2_df row order with `models` (build_r2_df iterates `models` so this
    # is a no-op today, but keep the guard in case inputs diverge later).
    key_to_row = {(r["family"], r["size"]): r for r in r2_df.iter_rows(named=True)}
    r2_df = pl.DataFrame([key_to_row[(f, s)] for f, s, _ in models])

    make_main_figure(mats[K_MAIN], r2_df, families, sizes, args.modality)
    make_per_property_figure(mats[K_MAIN], r2_df, families, sizes, args.modality)


if __name__ == "__main__":
    main()
