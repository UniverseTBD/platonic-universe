#!/usr/bin/env python3
"""
Plot cross-architectural MKNN / CKA against HSC mean physics R².

For each model we compute its mean pairwise MKNN (resp. CKA) to every
other model — its degree of convergence to the cross-family consensus —
and plot that against its HSC mean R² across (redshift, mass, sSFR)
from ``r2_vs_params_45000galaxies_upsampled.json``.

Embeddings are loaded from ``data/jwst/jwst_{family}_{size}.parquet``,
which each contain both an ``_hsc`` and a ``_jwst`` column, giving one
cross-architectural MKNN/CKA panel per modality (HSC and JWST side by
side), mirroring the layout of ``plot_crossmodal.py``.

One panel per modality, one point per model. Tests the PRH prediction
that models closer to the cross-architecture consensus also retain more
physics information.

Complements ``plot_intramodal.py`` (same-family scaling) and
``plot_crossmodal.py`` (same-model across modalities).

Metric computation is delegated to ``pu.metrics``: use ``--method
compare`` (default) for raw MKNN/CKA or ``--method calibrate`` for
permutation-calibrated scores via ``pu.metrics.calibrate``.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIGS_DIR = ROOT / "figs"
JWST_DIR = DATA_DIR / "jwst"

sys.path.insert(0, str(ROOT / "src"))
from pu.metrics import METRICS_REGISTRY, calibrate, compare  # noqa: E402

DEFAULT_R2_JSON = ROOT / "r2_vs_params_45000galaxies_upsampled.json"
R2_PROPS = ("redshift", "mass", "sSFR")

MODALITIES = ("hsc", "jwst")
MODALITY_LABEL = {"hsc": "HSC", "jwst": "JWST"}

METHODS = ("compare", "calibrate")
METRICS = ("mknn", "cka")

N_SUB = 5_000
K_MAIN = 10
N_PERMUTATIONS = 200
SEED = 0


def _cache_path(method: str, modality: str) -> Path:
    return DATA_DIR / f"crossarch_{method}_{modality}.parquet"


# Explicit model list matching the pattern in plot_crossmodal.py.
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


def _pair_scores(Zi: np.ndarray, Zj: np.ndarray, method: str) -> dict[str, float]:
    """Return {"mknn": ..., "cka": ...} for a model pair using *method*."""
    if method == "compare":
        return compare(Zi, Zj, metrics=list(METRICS), mknn__k=K_MAIN)
    if method == "calibrate":
        from functools import partial

        out: dict[str, float] = {}
        for name in METRICS:
            fn = METRICS_REGISTRY[name]
            if name == "mknn":
                fn = partial(fn, k=K_MAIN)
            # Note: pu.metrics.calibrate forwards `seed` to the underlying
            # calibrated_similarity package, which doesn't accept it.
            result = calibrate(Zi, Zj, fn, n_permutations=N_PERMUTATIONS)
            out[name] = float(result["calibrated_score"])
        return out
    raise ValueError(f"Unknown method: {method!r}")


_WORKER_ZS: list[np.ndarray] | None = None
_WORKER_METHOD: str | None = None
_WORKER_BLAS_THREADS: int = 1


def _worker_init(blas_threads: int) -> None:
    """Pin BLAS / OpenMP / torch threads inside each worker.

    Workers inherit Zs and the method string from the parent via fork, so
    no initargs for them — avoids re-pickling the embedding stack.
    ``blas_threads`` is passed explicitly so we can budget
    workers × blas_threads ≈ physical cores.
    """
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(blas_threads)
    except ImportError:
        pass
    try:
        import torch
        torch.set_num_threads(blas_threads)
    except ImportError:
        pass


def _worker_pair(ij: tuple[int, int]) -> tuple[int, int, dict[str, float]]:
    i, j = ij
    return i, j, _pair_scores(_WORKER_ZS[i], _WORKER_ZS[j], _WORKER_METHOD)


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def compute_metric_matrices(
    models: list[tuple[str, str, Path]],
    idx: np.ndarray,
    modality: str,
    method: str,
    workers: int = 1,
    blas_threads: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute pairwise MKNN + CKA matrices over *models* using *method*."""
    print(f"Loading embeddings ({modality}, {len(models)} models, "
          f"{idx.size} samples)")
    Zs: list[np.ndarray] = []
    for family, json_size, path in tqdm(models, desc="Embeddings"):
        Zs.append(load_embeddings(family, json_size, path, modality)[idx])

    M = len(models)
    mats = {name: np.full((M, M), np.nan, dtype=np.float64) for name in METRICS}
    pairs = [(i, j) for i in range(M) for j in range(i + 1, M)]
    n_workers = max(1, min(workers, len(pairs)))
    ncpu = _available_cpus()
    if blas_threads is None:
        blas_threads = max(1, ncpu // n_workers)
    print(f"Computing {method} MKNN+CKA over {len(pairs)} model pairs "
          f"({n_workers} worker{'s' if n_workers != 1 else ''}, "
          f"{blas_threads} BLAS thread{'s' if blas_threads != 1 else ''} each; "
          f"ncpu={ncpu})")

    if n_workers == 1:
        for i, j in tqdm(pairs, desc=f"Pairwise ({method})"):
            scores = _pair_scores(Zs[i], Zs[j], method)
            for name in METRICS:
                v = float(scores[name]) if scores[name] is not None else np.nan
                mats[name][i, j] = v
                mats[name][j, i] = v
        return mats

    global _WORKER_ZS, _WORKER_METHOD
    _WORKER_ZS = Zs
    _WORKER_METHOD = method
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(blas_threads,),
    ) as ex:
        for i, j, scores in tqdm(
            ex.map(_worker_pair, pairs, chunksize=1),
            total=len(pairs),
            desc=f"Pairwise ({method}, x{n_workers})",
        ):
            for name in METRICS:
                v = float(scores[name]) if scores[name] is not None else np.nan
                mats[name][i, j] = v
                mats[name][j, i] = v
    return mats


def matrices_to_df(
    mats: dict[str, np.ndarray], labels: list[str]
) -> pl.DataFrame:
    rows = []
    M = len(labels)
    for name, mat in mats.items():
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                rows.append({
                    "metric": name,
                    "model_i": labels[i],
                    "model_j": labels[j],
                    "value": mat[i, j],
                })
    return pl.DataFrame(rows)


def df_to_matrices(
    df: pl.DataFrame, labels: list[str]
) -> dict[str, np.ndarray]:
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    M = len(labels)
    mats: dict[str, np.ndarray] = {}
    for name in df["metric"].unique().to_list():
        sub = df.filter(pl.col("metric") == name)
        mat = np.full((M, M), np.nan, dtype=np.float64)
        for row in sub.iter_rows(named=True):
            i = label_to_idx.get(row["model_i"])
            j = label_to_idx.get(row["model_j"])
            if i is None or j is None:
                continue
            mat[i, j] = row["value"]
        mats[name] = mat
    return mats


def compute_or_load_matrices(
    models: list[tuple[str, str, Path]],
    labels: list[str],
    idx: np.ndarray,
    recompute: bool,
    modality: str,
    method: str,
    workers: int = 1,
    blas_threads: int | None = None,
) -> dict[str, np.ndarray]:
    cache = _cache_path(method, modality)
    if cache.exists() and not recompute:
        print(f"Loading cached {method} matrices ({modality}) from {cache}")
        mats = df_to_matrices(pl.read_parquet(cache), labels)
        missing_metrics = [m for m in METRICS if m not in mats]
        missing_models = any(
            np.all(np.isnan(mats[m]), axis=0).any() for m in mats
        )
        if missing_metrics or missing_models:
            print(f"Cache incomplete (missing_metrics={missing_metrics}, "
                  f"missing_models={missing_models}); recomputing")
            mats = compute_metric_matrices(
                models, idx, modality, method,
                workers=workers, blas_threads=blas_threads,
            )
            matrices_to_df(mats, labels).write_parquet(cache)
        return mats
    mats = compute_metric_matrices(
        models, idx, modality, method,
        workers=workers, blas_threads=blas_threads,
    )
    matrices_to_df(mats, labels).write_parquet(cache)
    print(f"Saved {method} matrices ({modality}) to {cache}")
    return mats


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

    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)


def metric_axis_label(modality: str, metric: str, method: str) -> str:
    metric_label = metric.upper()
    if method == "compare":
        return f"{MODALITY_LABEL[modality]} [cross-arch {metric_label} %]"
    return f"{MODALITY_LABEL[modality]} [cross-arch calibrated {metric_label}]"


def method_suffix(method: str) -> str:
    return "" if method == "compare" else f"_{method}"


def _make_figure(
    x_per_modality: dict[str, np.ndarray],
    mean_y: np.ndarray,
    families: list[str],
    metric: str,
    method: str,
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

    scale = 100.0 if method == "compare" else 1.0

    for ax, modality in zip(axes, modalities):
        is_first = ax is axes[0]
        plot_scatter(
            ax,
            x_per_modality[modality] * scale, mean_y,
            families,
            xlabel=metric_axis_label(modality, metric, method),
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
                        help="Path to r2_vs_params JSON")
    parser.add_argument("--n-sub", type=int, default=N_SUB,
                        help="Number of galaxies to subsample")
    parser.add_argument("--method", choices=METHODS, default="compare",
                        help="Metric backend: raw (compare) or "
                             "permutation-calibrated (calibrate)")
    parser.add_argument("--recompute", action="store_true",
                        help="Ignore cached matrices and recompute them")
    parser.add_argument("--check-alignment", action="store_true",
                        help="Only verify all parquets have the same row count and exit")
    parser.add_argument("--exclude-families", nargs="*", default=["dinov3"],
                        metavar="FAMILY",
                        help=f"Family names to drop (valid: "
                             f"{sorted(FAMILY_STYLE)})")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers for the pairwise metric loop. "
                             "Default: min(8, ncpu // 2). Set 1 to disable "
                             "multiprocessing.")
    parser.add_argument("--blas-threads", type=int, default=None,
                        help="BLAS / OpenMP / torch threads per worker. "
                             "Default: ncpu // workers so the product ≈ ncpu.")
    args = parser.parse_args()

    ncpu = _available_cpus()
    if args.workers is None:
        args.workers = max(1, min(8, ncpu // 2))

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
    print(f"Metric method: {args.method}")

    labels = [f"{f}_{s}" for f, s, _ in models]
    families = [f for f, s, _ in models]
    y = np.array([r2_for_model(r2, f, s) for f, s, _ in models])

    x_per_metric: dict[str, dict[str, np.ndarray]] = {m: {} for m in METRICS}
    for modality in MODALITIES:
        mats = compute_or_load_matrices(
            models, labels, idx, args.recompute, modality, args.method,
            workers=args.workers, blas_threads=args.blas_threads,
        )
        for name in METRICS:
            x_per_metric[name][modality] = np.nanmean(mats[name], axis=1)

    suffix = method_suffix(args.method)
    _make_figure(
        x_per_metric["mknn"], y, families,
        metric="mknn", method=args.method,
        out_name=f"crossarchitectural{suffix}.pdf",
    )
    _make_figure(
        x_per_metric["cka"], y, families,
        metric="cka", method=args.method,
        out_name=f"crossarchitectural_cka{suffix}.pdf",
    )


if __name__ == "__main__":
    main()
