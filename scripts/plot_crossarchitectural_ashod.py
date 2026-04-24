#!/usr/bin/env python3
"""
Plot cross-architectural MKNN / CKA against HSC mean physics R².

For each model we compute its mean pairwise MKNN (resp. CKA) to every
other model — its degree of convergence to the cross-family consensus —
and plot that against its HSC mean R² across (redshift, mass, sSFR)
from Ashod's ``r2_vs_params_45000galaxies_upsampled.json``.

Embeddings are loaded from ``data/jwst/jwst_{family}_{size}.parquet``,
which each contain both an ``_hsc`` and a ``_jwst`` column, giving one
cross-architectural MKNN/CKA panel per modality (HSC and JWST side by
side), mirroring the layout of ``plot_crossmodal_ashod.py``.

One panel per modality, one point per model. Tests the PRH prediction
that models closer to the cross-architecture consensus also retain more
physics information.

Complements ``plot_intramodal_ashod.py`` (same-family scaling) and
``plot_crossmodal_ashod.py`` (same-model across modalities).
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
DATA_DIR = ROOT / "data"
FIGS_DIR = ROOT / "figs"
JWST_DIR = DATA_DIR / "jwst"

sys.path.insert(0, str(ROOT / "src"))
from pu.metrics.kernel import cka  # noqa: E402

DEFAULT_R2_JSON = ROOT / "r2_vs_params_45000galaxies_upsampled.json"
R2_PROPS = ("redshift", "mass", "sSFR")

MODALITIES = ("hsc", "jwst")
MODALITY_LABEL = {"hsc": "HSC", "jwst": "JWST"}

N_SUB = 10_000
K_VALUES = (5, 10, 20)
K_MAIN = 10
SEED = 0


def _mknn_cache(modality: str) -> Path:
    return DATA_DIR / f"mknn_matrix_{modality}.parquet"


def _cka_cache(modality: str) -> Path:
    return DATA_DIR / f"cka_matrix_{modality}.parquet"

# Explicit model list matching the pattern in plot_crossmodal_ashod.py.
# Each entry is (family, json_size) — both are also the column-name tokens
# in the JWST parquets (e.g. "astropt_15m_hsc", "astropt_15m_jwst").
MODELS: list[tuple[str, str]] = [
    ("astropt",   "15m"),
    ("astropt",   "95m"),
    ("astropt",   "850m"),
    ("clip",      "base"),
    ("clip",      "large"),
    ("convnext",  "nano"),
    ("convnext",  "tiny"),
    ("convnext",  "base"),
    ("convnext",  "large"),
    ("dinov3",    "vits16"),
    ("dinov3",    "vits16plus"),
    ("dinov3",    "vitb16"),
    ("dinov3",    "vitl16"),
    ("dinov3",    "vith16plus"),
    ("dinov3",    "vit7b16"),
    ("ijepa",     "huge"),
    ("ijepa",     "giant"),
    ("paligemma", "3b"),
    ("paligemma", "10b"),
    ("paligemma", "28b"),
    ("vit",       "base"),
    ("vit",       "large"),
    ("vit",       "huge"),
    ("vit-mae",   "base"),
    ("vit-mae",   "large"),
    ("vit-mae",   "huge"),
    ("vjepa",     "large"),
    ("vjepa",     "huge"),
    ("vjepa",     "giant"),
]

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


def resolve_models(
    models: list[tuple[str, str]],
    r2: dict,
    excluded: set[str] = frozenset(),
) -> list[tuple[str, str, Path]]:
    """Return (family, json_size, path) for each entry in MODELS.

    Skips entries whose family is in *excluded*, whose R² JSON entry is
    absent, or whose JWST parquet does not exist on disk.
    """
    result: list[tuple[str, str, Path]] = []
    for family, json_size in models:
        if family in excluded:
            continue
        try:
            r2["hsc"][family][json_size]
        except KeyError:
            print(f"  Skipping {family}_{json_size}: not in R² JSON under 'hsc'")
            continue
        path = JWST_DIR / f"jwst_{family}_{json_size}.parquet"
        if not path.exists():
            print(f"  Skipping {family}_{json_size}: {path.name} not found")
            continue
        result.append((family, json_size, path))
    return result


def assert_aligned(models: list[tuple[str, str, Path]]) -> int:
    lengths = {}
    for family, json_size, path in models:
        col = f"{family}_{json_size}_hsc"
        lengths[(family, json_size)] = pl.read_parquet(path, columns=[col]).height
    n_unique = set(lengths.values())
    if len(n_unique) != 1:
        msg = "Parquets have different row counts:\n" + "\n".join(
            f"  {fam}_{sz}: {n}" for (fam, sz), n in lengths.items()
        )
        raise RuntimeError(msg)
    return n_unique.pop()


def load_embeddings(
    family: str, json_size: str, path: Path, modality: str
) -> np.ndarray:
    col = f"{family}_{json_size}_{modality}"
    df = pl.read_parquet(path, columns=[col])
    return np.stack(df[col].to_list()).astype(np.float32, copy=False)


def unit_normalize(Z: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return Z / norms


def compute_knn_indices(Z: np.ndarray, k: int) -> np.ndarray:
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


def compute_mknn_matrix(
    models: list[tuple[str, str, Path]],
    idx: np.ndarray,
    modality: str,
) -> dict[int, np.ndarray]:
    k_max = max(K_VALUES)
    print(f"Computing kNN (k={k_max}, {modality}) "
          f"for {len(models)} models on {idx.size} samples")
    nn_indices: list[np.ndarray] = []
    for family, json_size, path in tqdm(models, desc="kNN per model"):
        Z = load_embeddings(family, json_size, path, modality)[idx]
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


def compute_cka_matrix(
    models: list[tuple[str, str, Path]],
    idx: np.ndarray,
    modality: str,
) -> np.ndarray:
    print(f"Loading embeddings for CKA ({modality}, "
          f"{len(models)} models, {idx.size} samples)")
    Zs: list[np.ndarray] = []
    for family, json_size, path in tqdm(models, desc="Embeddings"):
        Zs.append(load_embeddings(family, json_size, path, modality)[idx])

    M = len(models)
    mat = np.full((M, M), np.nan, dtype=np.float64)
    pairs = [(i, j) for i in range(M) for j in range(i + 1, M)]
    print(f"Computing CKA over {len(pairs)} model pairs")
    for i, j in tqdm(pairs, desc="Pairwise CKA"):
        v = float(cka(Zs[i], Zs[j], kernel="linear"))
        mat[i, j] = v
        mat[j, i] = v
    return mat


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


def df_to_mknn_matrix(
    df: pl.DataFrame, labels: list[str]
) -> dict[int, np.ndarray]:
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


def cka_matrix_to_df(mat: np.ndarray, labels: list[str]) -> pl.DataFrame:
    rows = []
    M = len(labels)
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            rows.append({"model_i": labels[i], "model_j": labels[j],
                         "cka": mat[i, j]})
    return pl.DataFrame(rows)


def df_to_cka_matrix(df: pl.DataFrame, labels: list[str]) -> np.ndarray:
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    M = len(labels)
    mat = np.full((M, M), np.nan, dtype=np.float64)
    for row in df.iter_rows(named=True):
        i = label_to_idx.get(row["model_i"])
        j = label_to_idx.get(row["model_j"])
        if i is None or j is None:
            continue
        mat[i, j] = row["cka"]
    return mat


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


def r2_for_model_prop(r2: dict, family: str, size: str, prop: str) -> float:
    try:
        return float(r2["hsc"][family][size][prop]["r2_mean"])
    except KeyError as e:
        raise KeyError(
            f"{family}/{size}/{prop} missing under 'hsc' in R² JSON"
        ) from e


def compute_or_load_mknn(
    models: list[tuple[str, str, Path]],
    labels: list[str],
    idx: np.ndarray,
    recompute: bool,
    modality: str,
) -> dict[int, np.ndarray]:
    cache = _mknn_cache(modality)
    if cache.exists() and not recompute:
        print(f"Loading cached MKNN matrix ({modality}) from {cache}")
        mats = df_to_mknn_matrix(pl.read_parquet(cache), labels)
        missing_ks = [k for k in K_VALUES if k not in mats]
        missing_models = any(
            np.all(np.isnan(mats[k]), axis=0).any() for k in mats
        )
        if missing_ks or missing_models:
            print(f"Cache incomplete (missing_ks={missing_ks}, "
                  f"missing_models={missing_models}); recomputing MKNN")
            mats = compute_mknn_matrix(models, idx, modality)
            mknn_matrix_to_df(mats, labels).write_parquet(cache)
        return mats
    mats = compute_mknn_matrix(models, idx, modality)
    mknn_matrix_to_df(mats, labels).write_parquet(cache)
    print(f"Saved MKNN matrix ({modality}) to {cache}")
    return mats


def compute_or_load_cka(
    models: list[tuple[str, str, Path]],
    labels: list[str],
    idx: np.ndarray,
    recompute: bool,
    modality: str,
) -> np.ndarray:
    cache = _cka_cache(modality)
    if cache.exists() and not recompute:
        print(f"Loading cached CKA matrix ({modality}) from {cache}")
        mat = df_to_cka_matrix(pl.read_parquet(cache), labels)
        if np.all(np.isnan(mat), axis=0).any():
            print(f"CKA cache ({modality}) incomplete; recomputing")
            mat = compute_cka_matrix(models, idx, modality)
            cka_matrix_to_df(mat, labels).write_parquet(cache)
        return mat
    mat = compute_cka_matrix(models, idx, modality)
    cka_matrix_to_df(mat, labels).write_parquet(cache)
    print(f"Saved CKA matrix ({modality}) to {cache}")
    return mat


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

    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)


def _make_figure(
    x_per_modality: dict[str, np.ndarray],
    mean_y: np.ndarray,
    families: list[str],
    metric_label: str,
    out_name: str,
) -> None:
    modalities = list(x_per_modality)
    n_panels = len(modalities)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(4.2 * n_panels, 3.6),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, modality in zip(axes, modalities):
        is_first = ax is axes[0]
        plot_scatter(
            ax,
            x_per_modality[modality] * 100.0, mean_y,
            families,
            xlabel=f"{MODALITY_LABEL[modality]} [cross-arch {metric_label} %]",
            ylabel="Mean $R^2$" if is_first else "",
        )

    seen: dict[str, object] = {}
    for ax in axes:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-json", type=Path, default=DEFAULT_R2_JSON,
                        help="Path to Ashod's r2_vs_params JSON")
    parser.add_argument("--n-sub", type=int, default=N_SUB,
                        help="Number of galaxies to subsample")
    parser.add_argument("--recompute", action="store_true",
                        help="Ignore MKNN/CKA caches and recompute them")
    parser.add_argument("--check-alignment", action="store_true",
                        help="Only verify all parquets have the same row count and exit")
    parser.add_argument("--exclude-families", nargs="+", default=["dinov3"],
                        metavar="FAMILY",
                        help=f"Family names to drop (valid: "
                             f"{sorted(FAMILY_STYLE)})")
    args = parser.parse_args()

    FIGS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    excluded = set(args.exclude_families)
    unknown = excluded - set(FAMILY_STYLE)
    if unknown:
        raise ValueError(
            f"Unknown family names in --exclude-families: {sorted(unknown)}. "
            f"Valid: {sorted(FAMILY_STYLE)}"
        )

    r2 = load_r2_json(args.r2_json)
    print(f"R² JSON: {args.r2_json}")

    models = resolve_models(MODELS, r2, excluded=excluded)
    if not models:
        raise RuntimeError(
            f"No models left (check JWST_DIR={JWST_DIR} and R² JSON)."
        )
    print(f"Using {len(models)} models (embeddings from {JWST_DIR}):")
    for family, json_size, path in models:
        print(f"  {family}_{json_size}  ({path.name})")

    n_full = assert_aligned(models)
    print(f"\nAll parquets aligned at {n_full} rows.")
    if args.check_alignment:
        return

    n_sub = min(args.n_sub, n_full)
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(n_full, size=n_sub, replace=False))
    print(f"Subsampling {n_sub} rows (seed={SEED}).")

    labels = [f"{f}_{s}" for f, s, _ in models]
    families = [f for f, s, _ in models]
    y = np.array([r2_for_model(r2, f, s) for f, s, _ in models])

    x_mknn: dict[str, np.ndarray] = {}
    x_cka: dict[str, np.ndarray] = {}
    for modality in MODALITIES:
        mknn_mats = compute_or_load_mknn(models, labels, idx, args.recompute, modality)
        x_mknn[modality] = np.nanmean(mknn_mats[K_MAIN], axis=1)
        cka_mat = compute_or_load_cka(models, labels, idx, args.recompute, modality)
        x_cka[modality] = np.nanmean(cka_mat, axis=1)

    _make_figure(
        x_mknn, y, families,
        metric_label="MKNN",
        out_name="crossarchitectural_ashod.pdf",
    )
    _make_figure(
        x_cka, y, families,
        metric_label="CKA",
        out_name="crossarchitectural_ashod_cka.pdf",
    )


if __name__ == "__main__":
    main()
