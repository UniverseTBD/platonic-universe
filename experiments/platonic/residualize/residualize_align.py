#!/usr/bin/env python3
"""Calibrated CKA / MKNN before and after residualizing astrophysics confounds.

For a pair of models, loads their embeddings and the COSMOS-Web physics catalog
(row-aligned by stream order), residualizes the confounds out of each embedding,
and reports null-calibrated CKA and MKNN before and after, plus a linear-probe
R² per property as a removal check. All O(n²) work is done on a random subsample
of ``--n-sim`` galaxies, averaged over ``--n-draws`` draws.

    python residualize_align.py --emb-repo UniverseTBD/pu-embeddings \\
        --tag-a <path.parquet> --tag-b <path.parquet> \\
        --label-a vit/base --label-b convnext/base \\
        --confounds redshift,mass,sSFR --out-dir out/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from pu.metrics.deconfound import residualize, residualize_crossfit
from pu.pu_datasets.cosmosweb import CATALOG_COLUMNS

try:
    from calibrated_similarity import calibrate as cs_calibrate
except ImportError as e:  # pragma: no cover
    raise SystemExit("calibrated_similarity is not installed (pip install calibrated_similarity).") from e


def cka_torch(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Linear Centered Kernel Alignment (feature-space form)."""
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    hsic = (Xc.t() @ Yc).pow(2).sum()
    denom = torch.linalg.norm(Xc.t() @ Xc) * torch.linalg.norm(Yc.t() @ Yc)
    if denom < 1e-12:
        return torch.zeros((), device=X.device)
    return hsic / denom


def make_mknn_torch(k: int):
    """Return a Mutual-kNN overlap similarity fn (cosine, self-excluded)."""

    def mknn_torch(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        n = X.shape[0]
        kk = min(k, n - 1)
        Xn = X / (X.norm(dim=1, keepdim=True) + 1e-12)
        Yn = Y / (Y.norm(dim=1, keepdim=True) + 1e-12)
        Sx, Sy = Xn @ Xn.t(), Yn @ Yn.t()
        diag = torch.arange(n, device=X.device)
        Sx[diag, diag] = -float("inf")
        Sy[diag, diag] = -float("inf")
        Mx = torch.zeros(n, n, dtype=torch.bool, device=X.device)
        My = torch.zeros(n, n, dtype=torch.bool, device=X.device)
        Mx.scatter_(1, Sx.topk(kk, dim=1).indices, True)
        My.scatter_(1, Sy.topk(kk, dim=1).indices, True)
        return ((Mx & My).sum(dim=1).float() / kk).mean()

    return mknn_torch


def calibrated(X, Y, sim, perms: int, device: torch.device, seed: int) -> dict[str, float]:
    """Null-calibrated similarity (permutation test, Algorithm 1)."""
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)
    gen = torch.Generator(device=device.type)
    gen.manual_seed(seed)
    score, p, tau = cs_calibrate(Xt, Yt, sim, K=perms, generator=gen, smax=1.0)
    return {"raw": float(sim(Xt, Yt)), "calibrated": float(score),
            "p_value": float(p), "tau": float(tau)}


def load_catalog(n_rows: int, dataset: str) -> dict[str, np.ndarray]:
    """Stream the physics labels in row order (scalar columns only)."""
    from datasets import load_dataset

    cols = list(CATALOG_COLUMNS.values())
    ds = load_dataset(dataset, split="train", streaming=True).select_columns(cols)
    raw = {c: [] for c in cols}
    for i, row in enumerate(ds):
        for c in cols:
            raw[c].append(row[c])
        if i + 1 >= n_rows:
            break
    out = {param: np.asarray(raw[col], dtype=np.float64) for param, col in CATALOG_COLUMNS.items()}
    out["g-r"] = out["mag_g"] - out["mag_r"]
    return out


def _load_emb_file(path: str) -> np.ndarray:
    """Load an (n, d) embedding matrix from .npy or .parquet."""
    if str(path).endswith(".parquet"):
        import polars as pl

        df = pl.read_parquet(path)
        col = "embeddings" if "embeddings" in df.columns else df.columns[0]
        return np.vstack([np.asarray(v, dtype=np.float32) for v in df[col].to_list()])
    return np.load(path)


def resolve_emb(path, repo, tag, token) -> np.ndarray:
    if path:
        return _load_emb_file(path)
    if repo and tag:
        from huggingface_hub import hf_hub_download

        fn = tag if tag.endswith((".parquet", ".npy")) else f"{tag}.npy"
        return _load_emb_file(hf_hub_download(repo, fn, repo_type="dataset", token=token))
    raise SystemExit("provide --emb-<x> PATH or (--emb-repo and --tag-<x>)")


def probe_r2(Z: np.ndarray, y: np.ndarray, cv: int = 5) -> float:
    """5-fold ridge R² for property *y* from embeddings *Z*."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(Z) < cv * 2:
        return float("nan")
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    return float(np.mean(cross_val_score(model, Z, y, cv=cv, scoring="r2")))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb-a", default=None)
    p.add_argument("--emb-b", default=None)
    p.add_argument("--emb-repo", default=os.environ.get("PU_EMB_REPO") or None)
    p.add_argument("--tag-a", default=None)
    p.add_argument("--tag-b", default=None)
    p.add_argument("--label-a", default="model_a")
    p.add_argument("--label-b", default="model_b")
    p.add_argument("--modality", default="hsc")
    p.add_argument("--dataset", default=os.environ.get("PU_REGRESS_DATASET",
                   "Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2"))
    p.add_argument("--confounds", default="redshift,mass,sSFR")
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--cv", type=int, default=0, help="0 = in-sample; K>0 = K-fold cross-fitted")
    p.add_argument("--n-sim", type=int, default=3000)
    p.add_argument("--n-draws", type=int, default=3)
    p.add_argument("--perms", type=int, default=200)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    print(f"[cfg] device={device} confounds={args.confounds} degree={args.degree} "
          f"n_sim={args.n_sim} n_draws={args.n_draws} perms={args.perms}", flush=True)

    Za = resolve_emb(args.emb_a, args.emb_repo, args.tag_a, token).astype(np.float64)
    Zb = resolve_emb(args.emb_b, args.emb_repo, args.tag_b, token).astype(np.float64)
    n = min(len(Za), len(Zb))
    Za, Zb = Za[:n], Zb[:n]
    print(f"[emb] A={args.label_a} {Za.shape}  B={args.label_b} {Zb.shape}", flush=True)

    catalog = load_catalog(n, args.dataset)
    conf_keys = [c.strip() for c in args.confounds.split(",") if c.strip()]
    C = np.column_stack([catalog[c][:n] for c in conf_keys])
    valid = np.all(np.isfinite(C), axis=1)
    if "redshift" in conf_keys:
        valid &= C[:, conf_keys.index("redshift")] > 0
    Za, Zb, C = Za[valid], Zb[valid], C[valid]
    nv = len(Za)
    print(f"[valid] {nv}/{n} galaxies with finite confounds {conf_keys}", flush=True)
    args.n_sim = min(args.n_sim, nv - 100)

    res_fn = (lambda Z: residualize_crossfit(Z, C, degree=args.degree, n_splits=args.cv, seed=args.seed)) \
        if args.cv > 0 else (lambda Z: residualize(Z, C, degree=args.degree))
    Za_res = res_fn(Za).astype(np.float64)
    Zb_res = res_fn(Zb).astype(np.float64)

    diag = []
    for ci, key in enumerate(conf_keys):
        y = C[:, ci]
        for lbl, Zr, Zrr in ((args.label_a, Za, Za_res), (args.label_b, Zb, Zb_res)):
            diag.append({"model": lbl, "property": key,
                         "r2_before": probe_r2(Zr, y), "r2_after": probe_r2(Zrr, y)})

    metric_fns = {"cka": cka_torch, "mknn": make_mknn_torch(args.k)}
    rng = np.random.default_rng(args.seed)
    per_draw: list[dict] = []
    for d in range(args.n_draws):
        idx = rng.choice(nv, size=args.n_sim, replace=False)
        phases = {"before": (Za[idx], Zb[idx]), "after": (Za_res[idx], Zb_res[idx])}
        for mname, sim in metric_fns.items():
            for phase, (Xs, Ys) in phases.items():
                r = calibrated(Xs, Ys, sim, args.perms, device, args.seed * 1000 + d)
                per_draw.append({"draw": d, "metric": mname, "phase": phase, **r})
                print(f"[draw {d}] {mname:4s} {phase:6s} raw={r['raw']:.4f} "
                      f"calibrated={r['calibrated']:.4f} p={r['p_value']:.3f}", flush=True)

    def agg(metric, phase, field):
        vals = [r[field] for r in per_draw if r["metric"] == metric and r["phase"] == phase]
        return float(np.mean(vals)), float(np.std(vals))

    summary = []
    for metric in ("cka", "mknn"):
        cb, cb_s = agg(metric, "before", "calibrated")
        ca, ca_s = agg(metric, "after", "calibrated")
        summary.append({
            "metric": metric, "calibrated_before": cb, "calibrated_before_std": cb_s,
            "calibrated_after": ca, "calibrated_after_std": ca_s, "delta_calibrated": ca - cb,
            "frac_retained": (ca / cb) if cb > 1e-9 else float("nan"),
            "raw_before": agg(metric, "before", "raw")[0], "raw_after": agg(metric, "after", "raw")[0],
        })

    out = {"config": {"label_a": args.label_a, "label_b": args.label_b, "modality": args.modality,
                      "confounds": conf_keys, "degree": args.degree, "cv": args.cv, "n_valid": nv,
                      "n_sim": args.n_sim, "n_draws": args.n_draws, "perms": args.perms, "k": args.k,
                      "d_a": int(Za.shape[1]), "d_b": int(Zb.shape[1])},
           "summary": summary, "probe_diagnostic": diag, "per_draw": per_draw}
    cv_tag = f"_cv{args.cv}" if args.cv > 0 else ""
    stem = f"residualize_{args.modality}_{args.label_a}_vs_{args.label_b}{cv_tag}".replace("/", "-")
    out_path = args.out_dir / f"{stem}.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 78)
    print(f"{args.label_a} vs {args.label_b} ({args.modality}) | remove {conf_keys} | n_valid={nv}")
    print(f"{'metric':6s} {'calib_before':>16s} {'calib_after':>16s} {'retained':>10s}")
    for s in summary:
        print(f"{s['metric']:6s} {s['calibrated_before']:8.4f}±{s['calibrated_before_std']:.4f} "
              f"{s['calibrated_after']:8.4f}±{s['calibrated_after_std']:.4f} "
              f"{100 * s['frac_retained']:9.1f}%")
    print("probe R² (after should ≈ 0):")
    for r in diag:
        print(f"  {r['model']:>14s} R²({r['property']:8s}) before={r['r2_before']:+.3f} after={r['r2_after']:+.3f}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
