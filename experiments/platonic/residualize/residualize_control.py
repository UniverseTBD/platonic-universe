#!/usr/bin/env python3
"""Random-subspace control for the astrophysics residualization.

Compares calibrated CKA/MKNN drops for a model pair under three coherent
projections (same directions removed from both models, mean included):
  physics       the confound subspace (real residualization)
  random-rank   a random subspace of the same rank as the confound basis
  random-var    a random subspace removing the same (centered) variance fraction
Averaged over ``--n-rand`` draws, on a subsample of ``--n-sim`` galaxies.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import residualize_align as ra  # noqa: E402
from pu.metrics.deconfound import confound_basis, residualize  # noqa: E402


def cvarfrac(Z, Zres):
    """Fraction of centered variance removed (mean excluded)."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    Dc = (Z - Zres) - (Z - Zres).mean(axis=0, keepdims=True)
    return float(np.sum(Dc**2) / (np.sum(Zc**2) + 1e-12))


def coherent_random_removal(Za, Zb, m, rng):
    """Remove the same random m-dim subspace (incl. the mean) from both models."""
    G = rng.standard_normal((Za.shape[0], m))
    G[:, 0] = 1.0
    Q, _ = np.linalg.qr(G)
    return Za - Q @ (Q.T @ Za), Zb - Q @ (Q.T @ Zb)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb-repo", default=os.environ.get("PU_EMB_REPO") or "UniverseTBD/pu-embeddings")
    p.add_argument("--tag-a", required=True)
    p.add_argument("--tag-b", required=True)
    p.add_argument("--label-a", default="model_a")
    p.add_argument("--label-b", default="model_b")
    p.add_argument("--modality", default="hsc")
    p.add_argument("--dataset", default=os.environ.get("PU_REGRESS_DATASET",
                   "Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2"))
    p.add_argument("--confounds", default="redshift,mass,sSFR")
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--n-sim", type=int, default=3000)
    p.add_argument("--n-rand", type=int, default=6)
    p.add_argument("--perms", type=int, default=150)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    device = torch.device("cpu")
    mknn = ra.make_mknn_torch(args.k)

    Za = ra.resolve_emb(None, args.emb_repo, args.tag_a, token).astype(np.float64)
    Zb = ra.resolve_emb(None, args.emb_repo, args.tag_b, token).astype(np.float64)
    n = min(len(Za), len(Zb))
    Za, Zb = Za[:n], Zb[:n]
    print(f"[emb] A={args.label_a} {Za.shape}  B={args.label_b} {Zb.shape}", flush=True)

    catalog = ra.load_catalog(n, args.dataset)
    conf = [c.strip() for c in args.confounds.split(",")]
    C = np.column_stack([catalog[c][:n] for c in conf])
    valid = np.all(np.isfinite(C), axis=1)
    if "redshift" in conf:
        valid &= C[:, conf.index("redshift")] > 0
    Za, Zb, C = Za[valid], Zb[valid], C[valid]
    nv = len(Za)

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(nv, size=min(args.n_sim, nv), replace=False)
    Zsa, Zsb, Cs = Za[idx], Zb[idx], C[idx]
    ns = len(idx)

    Zsa_res = residualize(Zsa, Cs, degree=args.degree).astype(np.float64)
    Zsb_res = residualize(Zsb, Cs, degree=args.degree).astype(np.float64)
    r_phys = confound_basis(Cs, degree=args.degree).shape[1]
    v_phys = 0.5 * (cvarfrac(Zsa, Zsa_res) + cvarfrac(Zsb, Zsb_res))
    m_var = int(min(ns - 1, max(r_phys, round(v_phys * ns))))
    print(f"[cfg] n_valid={nv} n_sim={ns} physics_rank={r_phys} "
          f"var_physics={v_phys:.3f} random_var_rank={m_var}", flush=True)

    def cal(X, Y, sim):
        return ra.calibrated(X, Y, sim, args.perms, device, args.seed)["calibrated"]

    def random_mean(sim, m):
        vals, vfr = [], []
        for d in range(args.n_rand):
            Zar, Zbr = coherent_random_removal(Zsa, Zsb, m, np.random.default_rng(args.seed * 7919 + d))
            vfr.append(0.5 * (cvarfrac(Zsa, Zar) + cvarfrac(Zsb, Zbr)))
            vals.append(cal(Zar, Zbr, sim))
        return float(np.mean(vals)), float(np.std(vals)), float(np.mean(vfr))

    res = {}
    for mname, sim in (("cka", ra.cka_torch), ("mknn", mknn)):
        before = cal(Zsa, Zsb, sim)
        phys = cal(Zsa_res, Zsb_res, sim)
        rr_mean, rr_std, rr_var = random_mean(sim, r_phys)
        rv_mean, rv_std, rv_var = random_mean(sim, m_var)
        drop = lambda x: 100 * (1 - x / before) if before > 1e-9 else float("nan")
        res[mname] = {"before": before, "physics_after": phys,
                      "random_rank_after": rr_mean, "random_rank_std": rr_std, "random_rank_var": rr_var,
                      "random_var_after": rv_mean, "random_var_std": rv_std, "random_var_var": rv_var,
                      "drop_physics": drop(phys), "drop_random_rank": drop(rr_mean), "drop_random_var": drop(rv_mean)}
        print(f"[{mname}] before={before:.4f} physics={phys:.4f} "
              f"rand_rank={rr_mean:.4f} rand_var={rv_mean:.4f}", flush=True)

    out = {"config": {"label_a": args.label_a, "label_b": args.label_b, "modality": args.modality,
                      "confounds": conf, "degree": args.degree, "n_valid": nv, "n_sim": ns,
                      "physics_rank": r_phys, "random_var_rank": m_var,
                      "var_removed_physics_centered": v_phys}, "result": res}
    stem = f"control_{args.modality}_{args.label_a}_vs_{args.label_b}".replace("/", "-")
    (args.out_dir / f"{stem}.json").write_text(json.dumps(out, indent=2))

    print("=" * 84)
    print(f"{args.label_a} vs {args.label_b} ({args.modality}) | physics removes "
          f"{100*v_phys:.0f}% centered var (rank {r_phys})")
    print(f"{'metric':6s} {'dropP':>6s} {'dropR-rank':>11s} {'dropR-var':>10s}")
    for m in ("cka", "mknn"):
        r = res[m]
        print(f"{m:6s} {r['drop_physics']:5.1f}% {r['drop_random_rank']:10.1f}% {r['drop_random_var']:9.1f}%")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
