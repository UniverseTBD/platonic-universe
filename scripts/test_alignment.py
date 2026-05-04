"""Crossmodal and intramodal alignment with shared permutation null.

Three analyses from the same loaded data:

1. CROSSMODAL: model(HSC) vs model(comparison_mode) per model.
   Shared row permutations -> calibrated z-scores -> scaling regression.
   For DESI, the Specformer embedding is shared across all image models,
   giving a natural 'shared y' experiment.

2. INTRAMODAL (same family): model_A(HSC) vs model_B(HSC) within families.

3. ROWWISE: Per-galaxy alignment scores across all models, enabling a
   repeated-measures scaling test with n_galaxies independent observations.

Run:
  uv run python scripts/test_alignment.py
  uv run python scripts/test_alignment.py --mode legacysurvey --subsample 5000
  uv run python scripts/test_alignment.py --mode desi
  uv run python scripts/test_alignment.py --intramodal  # also run intramodal tests
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, ttest_1samp

from pu.metrics import mknn


HF_REPO = "UniverseTBD/pu-embeddings"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"

FAMILIES: dict[str, dict] = {
    "vit": {
        "sizes": ["base", "large", "huge"],
        "params": {"base": 86e6, "large": 304e6, "huge": 632e6},
    },
    "dino": {
        "sizes": ["small", "base", "large", "giant"],
        "params": {"small": 22e6, "base": 86e6, "large": 304e6, "giant": 1.1e9},
    },
    "dinov3": {
        "sizes": ["vits16", "vits16plus", "vitb16", "vitl16", "vith16plus", "vit7b16"],
        "params": {
            "vits16": 22e6, "vits16plus": 29e6, "vitb16": 86e6,
            "vitl16": 304e6, "vith16plus": 632e6, "vit7b16": 7e9,
        },
    },
    "convnext": {
        "sizes": ["nano", "tiny", "base", "large"],
        "params": {"nano": 15e6, "tiny": 28e6, "base": 89e6, "large": 198e6},
    },
    "clip": {
        "sizes": ["base", "large"],
        "params": {"base": 86e6, "large": 304e6},
    },
    "ijepa": {
        "sizes": ["huge", "giant"],
        "params": {"huge": 630e6, "giant": 1.0e9},
    },
    "vjepa": {
        "sizes": ["large", "huge", "giant"],
        "params": {"large": 304e6, "huge": 632e6, "giant": 1.1e9},
    },
    "vit-mae": {
        "sizes": ["base", "large", "huge"],
        "params": {"base": 86e6, "large": 304e6, "huge": 632e6},
    },
    "sam2": {
        "sizes": ["tiny", "small", "base-plus", "large"],
        "params": {"tiny": 27e6, "small": 41e6, "base-plus": 81e6, "large": 224e6},
    },
    "astropt": {
        "sizes": ["015M", "095M", "850M"],
        "params": {"015M": 15e6, "095M": 95e6, "850M": 850e6},
    },
    "llava_15": {
        "sizes": ["7b", "13b"],
        "params": {"7b": 7e9, "13b": 13e9},
    },
    "paligemma_3b": {
        "sizes": ["3b"],
        "params": {"3b": 3e9},
    },
    "paligemma_10b": {
        "sizes": ["10b"],
        "params": {"10b": 10e9},
    },
    "paligemma_28b": {
        "sizes": ["28b"],
        "params": {"28b": 28e9},
    },
}

TOPK = 10
SEED = 42


# ---------------------------------------------------------------------------
# PRH preprocessing (outlier clip + L2 normalize)
# ---------------------------------------------------------------------------

def remove_outliers(feats: torch.Tensor, q: float = 0.95) -> torch.Tensor:
    if q == 1:
        return feats
    q_val = torch.quantile(feats.abs().flatten(start_dim=1), q, dim=1).mean()
    return feats.clamp(-q_val, q_val)


def prepare(feats: np.ndarray, q: float = 0.95) -> np.ndarray:
    t = torch.as_tensor(feats, dtype=torch.float32)
    t = remove_outliers(t, q=q)
    t = F.normalize(t, p=2, dim=-1)
    return t.numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Rowwise (per-galaxy) mutual k-NN overlap
# ---------------------------------------------------------------------------

def rowwise_mknn(X: np.ndarray, Y: np.ndarray, k: int = 10) -> np.ndarray:
    """Per-galaxy mutual k-NN overlap, shape (n,). Mean = global mknn."""
    Xt = torch.from_numpy(X).float()
    Yt = torch.from_numpy(Y).float()

    sim_x = Xt @ Xt.T
    sim_x.fill_diagonal_(-1e8)
    sim_y = Yt @ Yt.T
    sim_y.fill_diagonal_(-1e8)

    knn_x = sim_x.topk(k, dim=1).indices
    knn_y = sim_y.topk(k, dim=1).indices

    n = X.shape[0]
    rng = torch.arange(n).unsqueeze(1)
    mx = torch.zeros(n, n)
    my = torch.zeros(n, n)
    mx[rng, knn_x] = 1.0
    my[rng, knn_y] = 1.0

    return ((mx * my).sum(dim=1) / k).numpy()


# ---------------------------------------------------------------------------
# Data loading — handles inconsistent HF repo naming
# ---------------------------------------------------------------------------

def _candidate_urls(family: str, size: str, comp_mode: str) -> list[str]:
    """Return candidate parquet URLs in order of likelihood."""
    urls = []
    if comp_mode == "jwst":
        urls.append(f"{HF_BASE}/jwst/{family}_{size}.parquet")
        urls.append(f"{HF_BASE}/jwst_gio/jwst_{family}_{size}.parquet")
    elif comp_mode == "desi":
        urls.append(f"{HF_BASE}/desi/desi_{family}_{size}.parquet")
        urls.append(f"{HF_BASE}/desi/{family}_{size}.parquet")
    elif comp_mode == "legacysurvey":
        urls.append(f"{HF_BASE}/legacysurvey/legacysurvey_{family}_{size}.parquet")
        urls.append(f"{HF_BASE}/legacysurvey/{family}_{size}.parquet")
    return urls


def _discover_columns(
    df: pl.DataFrame, comp_mode: str,
) -> tuple[str | None, str | None]:
    """Find the (hsc_col, comp_col) pair in the dataframe."""
    hsc_cols = [c for c in df.columns if c.endswith("_hsc")]
    comp_cols = [c for c in df.columns if c.endswith(f"_{comp_mode}")]
    if len(hsc_cols) == 1 and len(comp_cols) == 1:
        return hsc_cols[0], comp_cols[0]
    return None, None


def try_load(
    family: str, size: str, comp_mode: str,
) -> tuple[pl.DataFrame, str, str] | None:
    """Try candidate URLs, discover columns, return (df, hsc_col, comp_col)."""
    for url in _candidate_urls(family, size, comp_mode):
        try:
            df = pl.read_parquet(url)
        except Exception:
            continue
        hsc_col, comp_col = _discover_columns(df, comp_mode)
        if hsc_col and comp_col:
            return df, hsc_col, comp_col
    return None


# ---------------------------------------------------------------------------
# Null distribution
# ---------------------------------------------------------------------------

def compute_null_mknn(
    X: np.ndarray, Y: np.ndarray, perms: list[np.ndarray], k: int = TOPK,
) -> list[float]:
    return [mknn(X, Y[p], k=k) for p in perms]


def zscore(observed: float, null: list[float]) -> float:
    std = np.std(null)
    if std < 1e-12:
        return 0.0
    return (observed - np.mean(null)) / std


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Crossmodal + intramodal alignment")
    parser.add_argument("--mode", default="jwst",
                        choices=["jwst", "legacysurvey", "desi"])
    parser.add_argument("--n-perm", type=int, default=200)
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results (TSV + rowwise npz)")
    parser.add_argument("--input", type=str, default=None,
                        help="Load cached results from this directory (skip computation)")
    parser.add_argument("--intramodal", action="store_true",
                        help="Also run intramodal (same-family HSC vs HSC) analysis")
    args = parser.parse_args()

    comp_mode = args.mode

    # --- Load from cache or compute ---
    if args.input:
        in_dir = Path(args.input)
        tsv_path = in_dir / f"crossmodal_{comp_mode}.tsv"
        rowwise_path = in_dir / f"rowwise_{comp_mode}.npz"

        print(f"Loading cached results from {in_dir} ...")
        results = []
        with open(tsv_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                results.append({
                    "key": row["model"],
                    "family": row["family"],
                    "params": float(row["params"]),
                    "mknn": float(row["mknn"]),
                    "z_mknn": float(row["z_mknn"]),
                })

        rw = np.load(rowwise_path)
        rowwise_mat = rw["rowwise"]
        for i, r in enumerate(results):
            r["rowwise"] = rowwise_mat[:, i]
        n = rowwise_mat.shape[0]

        print(f"  Loaded {len(results)} models, {n} galaxies")
        for r in results:
            print(f"  {r['key']:25s}  params={r['params']:>10.0f}  "
                  f"mknn={r['mknn']:.4f}(z={r['z_mknn']:+.1f})")

        # Load intramodal if present and requested
        intra_tsv = in_dir / f"intramodal_{comp_mode}.tsv"
        intra_rw_path = in_dir / f"intramodal_rowwise_{comp_mode}.npz"
        intra_results = []
        if args.intramodal and intra_tsv.exists():
            with open(intra_tsv) as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    intra_results.append({
                        "model_a": row["model_a"],
                        "model_b": row["model_b"],
                        "family": row["family"],
                        "geomean_params": float(row["geomean_params"]),
                        "mknn": float(row["mknn"]),
                        "z_mknn": float(row["z_mknn"]),
                    })
            if intra_rw_path.exists():
                irw = np.load(intra_rw_path)
                for i, r in enumerate(intra_results):
                    r["rowwise"] = irw["rowwise"][:, i]

    else:
        # --- Load all models ---
        print(f"Loading embeddings for mode={comp_mode} ...")
        embeddings: dict[str, tuple[np.ndarray, np.ndarray, float, str]] = {}

        for family, cfg in FAMILIES.items():
            for size in cfg["sizes"]:
                key = f"{family}_{size}"
                result = try_load(family, size, comp_mode)
                if result is None:
                    print(f"  {key}: not found, skipping")
                    continue

                df, hsc_col, comp_col = result
                X = prepare(np.stack(df[hsc_col].to_list()))
                Y = prepare(np.stack(df[comp_col].to_list()))
                params = cfg["params"][size]
                embeddings[key] = (X, Y, params, family)
                print(f"  {key}: {X.shape[0]} x ({X.shape[1]}, {Y.shape[1]})  "
                      f"[{hsc_col}, {comp_col}]")

        if len(embeddings) < 2:
            print("Need at least 2 models loaded.")
            return

        # Verify row counts
        n_set = {k: v[0].shape[0] for k, v in embeddings.items()}
        unique_n = set(n_set.values())
        if len(unique_n) > 1:
            print(f"Row count mismatch: {n_set}")
            return
        n = unique_n.pop()

        if args.subsample and args.subsample < n:
            rng_sub = np.random.default_rng(SEED)
            idx = rng_sub.choice(n, size=args.subsample, replace=False)
            idx.sort()
            embeddings = {
                k: (X[idx], Y[idx], p, f)
                for k, (X, Y, p, f) in embeddings.items()
            }
            n = args.subsample
            print(f"Subsampled to {n} rows")

        # Generate shared permutations
        rng = np.random.default_rng(SEED)
        perms = [rng.permutation(n) for _ in range(args.n_perm)]

        # ===============================================================
        # EXPERIMENT 1: CROSSMODAL ALIGNMENT
        # ===============================================================
        print(f"\n{'='*80}")
        print(f"CROSSMODAL: HSC vs {comp_mode.upper()} (n={n}, {args.n_perm} shared perms)")
        print(f"{'='*80}\n")

        results = []
        keys_ordered = sorted(embeddings.keys(),
                              key=lambda k: embeddings[k][2])

        for key in keys_ordered:
            X, Y, params, family = embeddings[key]

            obs_mknn = mknn(X, Y, k=TOPK)
            null_mknn = compute_null_mknn(X, Y, perms)
            z = zscore(obs_mknn, null_mknn)
            per_galaxy = rowwise_mknn(X, Y, k=TOPK)

            results.append({
                "key": key, "family": family, "params": params,
                "mknn": obs_mknn, "z_mknn": z, "rowwise": per_galaxy,
            })

            print(f"  {key:25s}  params={params:>10.0f}  "
                  f"mknn={obs_mknn:.4f}(z={z:+.1f})")

        # --- Save crossmodal results ---
        if args.output:
            out_dir = Path(args.output)
            out_dir.mkdir(parents=True, exist_ok=True)

            tsv_path = out_dir / f"crossmodal_{comp_mode}.tsv"
            with open(tsv_path, "w", newline="") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["model", "family", "params", "mknn", "z_mknn"])
                for r in results:
                    w.writerow([r["key"], r["family"], r["params"],
                                f"{r['mknn']:.6f}", f"{r['z_mknn']:.4f}"])
            print(f"\n  Saved {tsv_path}")

            rowwise_path = out_dir / f"rowwise_{comp_mode}.npz"
            np.savez(
                rowwise_path,
                rowwise=np.column_stack([r["rowwise"] for r in results]),
                models=np.array([r["key"] for r in results]),
                params=np.array([r["params"] for r in results]),
            )
            print(f"  Saved {rowwise_path}")

        # ===============================================================
        # EXPERIMENT 2: INTRAMODAL (same family, all HSC pairs)
        # ===============================================================
        intra_results = []
        if args.intramodal:
            print(f"\n{'='*80}")
            print(f"INTRAMODAL (same family): HSC vs HSC, all pairs (n={n}, {args.n_perm} shared perms)")
            print(f"{'='*80}")

            for family, cfg in FAMILIES.items():
                sizes = cfg["sizes"]
                keys_fam = [f"{family}_{s}" for s in sizes if f"{family}_{s}" in embeddings]
                if len(keys_fam) < 2:
                    continue

                print(f"\n  {family}:")
                for i in range(len(keys_fam)):
                    for j in range(i + 1, len(keys_fam)):
                        ki, kj = keys_fam[i], keys_fam[j]
                        Xi, _, pi, _ = embeddings[ki]
                        Xj, _, pj, _ = embeddings[kj]

                        geomean_p = np.sqrt(pi * pj)
                        score = mknn(Xi, Xj, k=TOPK)
                        null = compute_null_mknn(Xi, Xj, perms)
                        z = zscore(score, null)
                        rw = rowwise_mknn(Xi, Xj, k=TOPK)

                        intra_results.append({
                            "model_a": ki, "model_b": kj, "family": family,
                            "geomean_params": geomean_p,
                            "mknn": score, "z_mknn": z, "rowwise": rw,
                        })
                        print(f"    {ki:25s} vs {kj:25s}  "
                              f"mknn={score:.4f}(z={z:+.1f})")

            if args.output and intra_results:
                out_dir = Path(args.output)
                tsv_path = out_dir / f"intramodal_{comp_mode}.tsv"
                with open(tsv_path, "w", newline="") as f:
                    w = csv.writer(f, delimiter="\t")
                    w.writerow(["model_a", "model_b", "family", "geomean_params",
                                "mknn", "z_mknn"])
                    for r in intra_results:
                        w.writerow([r["model_a"], r["model_b"], r["family"],
                                    r["geomean_params"],
                                    f"{r['mknn']:.6f}", f"{r['z_mknn']:.4f}"])
                print(f"\n  Saved {tsv_path}")

                rw_path = out_dir / f"intramodal_rowwise_{comp_mode}.npz"
                np.savez(
                    rw_path,
                    rowwise=np.column_stack([r["rowwise"] for r in intra_results]),
                    pairs=np.array([(r["model_a"], r["model_b"]) for r in intra_results]),
                    geomean_params=np.array([r["geomean_params"] for r in intra_results]),
                )
                print(f"  Saved {rw_path}")

    # ===================================================================
    # ANALYSIS (runs on both computed and cached results)
    # ===================================================================

    # --- Per-family fixed-effects scaling regression ---
    print(f"\n{'='*80}")
    print(f"CROSSMODAL SCALING ANALYSIS")
    print(f"{'='*80}")
    print(f"\nPer-family scaling (family-fixed-effects z-scores):")
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_family[r["family"]].append(r)

    residual_z = []
    residual_logp = []
    for fam, fam_results in sorted(by_family.items()):
        if len(fam_results) < 2:
            continue
        fam_zvals = [r["z_mknn"] for r in fam_results]
        fam_mean = np.mean(fam_zvals)
        fam_logp = np.log10([r["params"] for r in fam_results])
        rho, p = spearmanr(fam_logp, fam_zvals)
        print(f"  {fam:15s} ({len(fam_results)} models): rho={rho:+.3f}  p={p:.2e}")
        for r in fam_results:
            residual_z.append(r["z_mknn"] - fam_mean)
            residual_logp.append(np.log10(r["params"]))

    if len(residual_z) >= 3:
        rho, p = spearmanr(residual_logp, residual_z)
        print(f"  {'POOLED':15s} ({len(residual_z)} models): rho={rho:+.3f}  p={p:.2e}")

    # --- Per-family rowwise repeated-measures test ---
    print(f"\nPer-family rowwise repeated-measures test:")
    for fam, fam_results in sorted(by_family.items()):
        if len(fam_results) < 2:
            continue
        fam_results_sorted = sorted(fam_results, key=lambda r: r["params"])
        fam_logp = np.log10([r["params"] for r in fam_results_sorted])
        fam_rowwise = np.column_stack([r["rowwise"] for r in fam_results_sorted])
        fam_rhos = np.array([
            spearmanr(fam_logp, fam_rowwise[i]).statistic
            for i in range(n)
        ])
        valid = ~np.isnan(fam_rhos)
        if valid.sum() > 1:
            mean_rho = fam_rhos[valid].mean()
            t_stat, t_p = ttest_1samp(fam_rhos[valid], 0.0)
            print(f"  {fam:15s}: mean_rho={mean_rho:+.4f}  "
                  f"t={t_stat:.2f}  p={t_p:.2e}  (n={valid.sum()})")

    # --- Intramodal scaling ---
    if intra_results:
        print(f"\n{'='*80}")
        print(f"INTRAMODAL SCALING ANALYSIS")
        print(f"{'='*80}")

        # Group by family
        intra_by_family: dict[str, list[dict]] = defaultdict(list)
        for r in intra_results:
            intra_by_family[r["family"]].append(r)

        # Per-family z-score scaling
        print(f"\nPer-family scaling (z-scores vs log geomean params):")
        intra_residual_z = []
        intra_residual_logp = []
        for fam, fam_pairs in sorted(intra_by_family.items()):
            fam_zvals = [r["z_mknn"] for r in fam_pairs]
            fam_logp = np.log10([r["geomean_params"] for r in fam_pairs])
            for r in fam_pairs:
                print(f"    {r['model_a']:25s} vs {r['model_b']:25s}  "
                      f"mknn={r['mknn']:.4f}(z={r['z_mknn']:+.1f})")
            if len(fam_pairs) >= 3:
                rho, p = spearmanr(fam_logp, fam_zvals)
                print(f"  {fam:15s} ({len(fam_pairs)} pairs): rho={rho:+.3f}  p={p:.2e}")
            else:
                print(f"  {fam:15s} ({len(fam_pairs)} pairs): too few for Spearman")
            fam_mean = np.mean(fam_zvals)
            for r in fam_pairs:
                intra_residual_z.append(r["z_mknn"] - fam_mean)
                intra_residual_logp.append(np.log10(r["geomean_params"]))

        if len(intra_residual_z) >= 3:
            rho, p = spearmanr(intra_residual_logp, intra_residual_z)
            print(f"\n  {'POOLED':15s} ({len(intra_residual_z)} pairs): rho={rho:+.3f}  p={p:.2e}")

        # Per-family rowwise repeated-measures test
        has_rowwise = all("rowwise" in r for r in intra_results)
        if has_rowwise:
            print(f"\nPer-family rowwise repeated-measures test (intramodal):")
            for fam, fam_pairs in sorted(intra_by_family.items()):
                if len(fam_pairs) < 3:
                    continue
                fam_pairs_sorted = sorted(fam_pairs, key=lambda r: r["geomean_params"])
                fam_logp = np.log10([r["geomean_params"] for r in fam_pairs_sorted])
                fam_rowwise = np.column_stack([r["rowwise"] for r in fam_pairs_sorted])
                fam_rhos = np.array([
                    spearmanr(fam_logp, fam_rowwise[i]).statistic
                    for i in range(n)
                ])
                valid = ~np.isnan(fam_rhos)
                if valid.sum() > 1:
                    mean_rho = fam_rhos[valid].mean()
                    t_stat, t_p = ttest_1samp(fam_rhos[valid], 0.0)
                    print(f"  {fam:15s}: mean_rho={mean_rho:+.4f}  "
                          f"t={t_stat:.2f}  p={t_p:.2e}  (n={valid.sum()})")



if __name__ == "__main__":
    main()
