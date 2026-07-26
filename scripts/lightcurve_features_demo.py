#!/usr/bin/env python
"""Light-curve feature-extractor demo (CPU, no downloads).

Generates synthetic light curves from three archetypes — a sinusoidal pulsator,
an eclipsing dip, and a flat noise floor — extracts `light-curve` features with
the `lightcurve` adapter, and reports how well those hand-engineered features
separate the classes (silhouette score in feature space + a 1-NN leave-one-out
accuracy). This is the classical baseline the Platonic Representation Hypothesis
compares learned time-series embeddings against.

    pip install ".[lightcurve]"
    python scripts/lightcurve_features_demo.py --preset default --n-per-class 60
"""
import argparse

import numpy as np

from pu.models import get_adapter


def make_lightcurve(kind: str, rng: np.random.RandomState, n: int = 200):
    """Return a (3, n) = (time, magnitude, sigma) synthetic light curve."""
    t = np.sort(rng.uniform(0, 100, n))
    noise = rng.normal(0, 0.03, n)
    if kind == "pulsator":               # coherent sinusoid
        period = rng.uniform(3, 12)
        m = 15 + 0.4 * np.sin(2 * np.pi * t / period) + noise
    elif kind == "eclipse":              # periodic narrow dips
        period = rng.uniform(5, 15)
        phase = (t % period) / period
        m = 15 + noise + 0.8 * (np.abs(phase - 0.5) < 0.05)
    elif kind == "flat":                 # constant + noise
        m = 15 + noise
    else:
        raise ValueError(kind)
    sigma = np.full(n, 0.03)
    return np.stack([t, m, sigma], axis=0)


def separation_scores(X: np.ndarray, y: np.ndarray) -> dict:
    """Silhouette + leave-one-out 1-NN accuracy on standardized features."""
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    # leave-one-out 1-NN
    D = np.linalg.norm(Xs[:, None, :] - Xs[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    acc = float((y[D.argmin(1)] == y).mean())
    # silhouette (mean over samples)
    sils = []
    for i in range(len(Xs)):
        same = y == y[i]
        same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        b = min(D[i, y == c].mean() for c in np.unique(y) if c != y[i])
        sils.append((b - a) / (max(a, b) + 1e-12))
    return {"loo_1nn_accuracy": acc, "silhouette": float(np.mean(sils))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default="default", choices=["cheap", "default", "full"])
    p.add_argument("--n-per-class", type=int, default=60)
    p.add_argument("--seq-len", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    classes = ["pulsator", "eclipse", "flat"]
    series, labels = [], []
    for ci, kind in enumerate(classes):
        for _ in range(args.n_per_class):
            series.append(make_lightcurve(kind, rng, args.seq_len))
            labels.append(ci)
    batch = {"lc": np.stack(series, axis=0)}   # (B, 3, T)
    y = np.asarray(labels)

    ad = get_adapter("lightcurve")("light-curve", args.preset, alias="lightcurve")
    ad.load()
    emb = ad.embed_for_mode(batch, "lc").numpy()   # (B, D)

    print(f"preset={args.preset}  features={len(ad.feature_names)}  "
          f"light curves={emb.shape[0]}  classes={classes}")
    scores = separation_scores(emb, y)
    print(f"leave-one-out 1-NN accuracy : {scores['loo_1nn_accuracy']:.3f}  "
          f"(chance = {1/len(classes):.3f})")
    print(f"silhouette score            : {scores['silhouette']:.3f}")
    # Most discriminative single features (by F-ratio) for a quick sanity read.
    fr = _f_ratio(emb, y)
    top = np.argsort(fr)[::-1][:5]
    print("top-5 discriminative features:",
          [ad.feature_names[i] for i in top])


def _f_ratio(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Between- over within-class variance per feature (higher = more class info)."""
    grand = X.mean(0)
    between = np.zeros(X.shape[1])
    within = np.zeros(X.shape[1])
    for c in np.unique(y):
        Xc = X[y == c]
        between += len(Xc) * (Xc.mean(0) - grand) ** 2
        within += ((Xc - Xc.mean(0)) ** 2).sum(0)
    return between / (within + 1e-12)


if __name__ == "__main__":
    main()
