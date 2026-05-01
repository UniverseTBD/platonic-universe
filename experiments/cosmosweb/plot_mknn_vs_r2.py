#!/usr/bin/env python3
"""
Plot MKNN embedding similarity (mean MKNN vs every other model) against
physics linear-probe R² (mean over stable Smith42/galaxies properties).

One point per model. Tests the PRH prediction that models whose
neighborhood structure is closer to the consensus also encode more physics.
"""

import argparse
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
    EXCLUDE_PROPERTIES,
    FAMILY_STYLE,
    PARAM_COUNTS,
)

sys.path.insert(0, str(ROOT / "src"))
from pu.metrics.physics import ALL_PROPERTIES, DEFAULT_PROPERTIES, linear_probe  # noqa: E402

N_SUB = 10_000
K_VALUES = (5, 10, 20)
K_MAIN = 10
SEED = 0

MKNN_CACHE = DATA_DIR / "mknn_matrix.parquet"
R2_CACHE = DATA_DIR / "physics_mean_r2.parquet"

STABLE_PROPS = [p for p in DEFAULT_PROPERTIES if p not in EXCLUDE_PROPERTIES]


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


def discover_models() -> list[tuple[str, str, Path]]:
    found: dict[tuple[str, str], Path] = {}
    for p in sorted(PHYSICS_DIR.glob("*_test.parquet")):
        parsed = parse_filename(p)
        if parsed:
            found[parsed] = p
    ordered: list[tuple[str, str, Path]] = []
    for family, sizes in PARAM_COUNTS.items():
        for size in sizes:
            if (family, size) in found:
                ordered.append((family, size, found[(family, size)]))
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


def load_labels(n_samples: int) -> dict[str, np.ndarray]:
    from datasets import load_dataset

    ds = load_dataset(
        "Smith42/galaxies", revision="v2.0", split="test", streaming=True,
    )
    keys = STABLE_PROPS
    buckets: dict[str, list[float]] = {k: [] for k in keys}
    for i, ex in enumerate(tqdm(ds, total=n_samples, desc="Streaming labels")):
        if i >= n_samples:
            break
        for k in keys:
            col = ALL_PROPERTIES.get(k, k)
            v = ex.get(col)
            buckets[k].append(float("nan") if v is None else float(v))

    out: dict[str, np.ndarray] = {}
    for k, vals in buckets.items():
        if len(vals) != n_samples:
            print(f"  warn: dropping {k} (got {len(vals)}, expected {n_samples})")
            continue
        arr = np.asarray(vals, dtype=np.float64)
        if k in ("sfr", "ssfr"):
            arr[arr <= -99] = np.nan
        if np.any(np.isfinite(arr)):
            out[k] = arr
        else:
            print(f"  warn: dropping {k} (all NaN)")
    return out


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


def compute_mean_r2(
    models: list[tuple[str, str, Path]],
    idx: np.ndarray,
    labels_full: dict[str, np.ndarray],
) -> pl.DataFrame:
    labels = {k: v[idx] for k, v in labels_full.items()}
    rows = []
    for family, size, path in tqdm(models, desc="Linear probes"):
        Z = load_embeddings(family, size, path)[idx]
        r2s: list[float] = []
        per_prop: dict[str, float] = {}
        for prop in STABLE_PROPS:
            if prop not in labels:
                continue
            res = linear_probe(Z, labels[prop], cv=5)
            if isinstance(res, dict):
                r2 = res["mean"]
            else:
                r2 = float(res)
            if np.isfinite(r2):
                r2s.append(r2)
                per_prop[prop] = r2
        mean_r2 = float(np.mean(r2s)) if r2s else float("nan")
        row = {"family": family, "size": size, "mean_r2": mean_r2, "n_props": len(r2s)}
        row.update({f"r2_{p}": v for p, v in per_prop.items()})
        rows.append(row)
        print(f"  {family:<10} {size:<12} mean R² = {mean_r2:.4f} over {len(r2s)} props")
    return pl.DataFrame(rows)


def plot_scatter(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    families: list[str],
    sizes: list[str],
    title: str,
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

    ax.set_xlabel("Mean MKNN to other models (cosine, k=%d)" %
                  (K_MAIN if title is None else K_MAIN), fontsize=11)
    ax.set_ylabel("Mean $R^2$ (linear probe)", fontsize=11)
    ax.set_title(title, fontsize=11)


def make_main_figure(
    mknn_mat: np.ndarray,
    r2_df: pl.DataFrame,
    families: list[str],
    sizes: list[str],
) -> None:
    x = np.nanmean(mknn_mat, axis=1)
    y = r2_df["mean_r2"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_scatter(
        ax, x, y, families, sizes,
        title=f"Representational convergence vs physics R² "
              f"(MKNN k={K_MAIN}, N={N_SUB} galaxies)",
    )
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    fig.tight_layout()
    out = FIGS_DIR / "mknn_vs_r2.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def make_sensitivity_figure(
    mats: dict[int, np.ndarray],
    r2_df: pl.DataFrame,
    families: list[str],
    sizes: list[str],
) -> None:
    ks = sorted(mats.keys())
    fig, axes = plt.subplots(1, len(ks), figsize=(6 * len(ks), 5), sharey=True)
    if len(ks) == 1:
        axes = [axes]
    y = r2_df["mean_r2"].to_numpy()
    for ax, k in zip(axes, ks):
        x = np.nanmean(mats[k], axis=1)
        plot_scatter(ax, x, y, families, sizes, title=f"k = {k}")
        ax.set_xlabel(f"Mean MKNN to other models (cosine, k={k})", fontsize=10)
    axes[0].legend(fontsize=8, loc="lower right", ncol=2)
    fig.suptitle("MKNN-vs-R² sensitivity to k", fontsize=13)
    fig.tight_layout()
    out = FIGS_DIR / "mknn_vs_r2_k_sensitivity.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-alignment", action="store_true",
                        help="Only verify all parquets have the same row count and exit")
    parser.add_argument("--n-sub", type=int, default=N_SUB,
                        help="Number of galaxies to subsample for MKNN and probes")
    parser.add_argument("--recompute", action="store_true",
                        help="Ignore caches and recompute everything")
    args = parser.parse_args()

    FIGS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    models = discover_models()
    if not models:
        raise RuntimeError(f"No physics parquets found under {PHYSICS_DIR}")
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
        if missing_ks:
            print(f"Cache missing k={missing_ks}, recomputing")
            mats = compute_mknn_matrix(models, idx)
            mknn_matrix_to_df(mats, labels).write_parquet(MKNN_CACHE)
    else:
        mats = compute_mknn_matrix(models, idx)
        mknn_matrix_to_df(mats, labels).write_parquet(MKNN_CACHE)
        print(f"Saved MKNN matrix to {MKNN_CACHE}")

    if R2_CACHE.exists() and not args.recompute:
        print(f"Loading cached R² table from {R2_CACHE}")
        r2_df = pl.read_parquet(R2_CACHE)
        cached_keys = {(r["family"], r["size"]) for r in r2_df.iter_rows(named=True)}
        needed_keys = {(f, s) for f, s, _ in models}
        if cached_keys != needed_keys:
            print("Cache model set differs from discovered; recomputing R²")
            labels_full = load_labels(n_full)
            r2_df = compute_mean_r2(models, idx, labels_full)
            r2_df.write_parquet(R2_CACHE)
    else:
        labels_full = load_labels(n_full)
        r2_df = compute_mean_r2(models, idx, labels_full)
        r2_df.write_parquet(R2_CACHE)
        print(f"Saved R² table to {R2_CACHE}")

    # Align r2_df row order with `models`
    key_to_row = {(r["family"], r["size"]): r for r in r2_df.iter_rows(named=True)}
    r2_df = pl.DataFrame([key_to_row[(f, s)] for f, s, _ in models])

    make_main_figure(mats[K_MAIN], r2_df, families, sizes)
    make_sensitivity_figure(mats, r2_df, families, sizes)


if __name__ == "__main__":
    main()
