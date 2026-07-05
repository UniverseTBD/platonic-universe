#!/usr/bin/env python
"""Layer-wise MKNN between two time-series foundation models on dummy data.

This is the smoke test / reference harness for the time-series API contract:
it loads two foundation models, iterates over every extraction layer of each,
pushes the *same* dummy batch through both, and compares their per-layer
representations with Mutual k-Nearest-Neighbours (MKNN).

It exists because during PRH revision we saw layer-wise MKNN move around
run-to-run even with seeds fixed. So this script also runs explicit
reproducibility checks (forward twice, self-MKNN == 1) before reporting the
cross-model numbers, and uses ``set_deterministic`` to pin BLAS threading.

No real data or GPU required. Defaults compare two *different* Chronos
architectures (T5 vs Bolt), both CPU-friendly:

    python scripts/timeseries_layerwise_mknn.py
    python scripts/timeseries_layerwise_mknn.py \
        --model-a chronos --repo-a amazon/chronos-t5-small \
        --model-b chronos --repo-b amazon/chronos-bolt-small \
        --n 128 --seq-len 256 --k 10
"""
import argparse

import numpy as np
import torch

from pu.metrics.neighbors import mknn
from pu.models import get_adapter
from pu.models.base import set_deterministic


def make_dummy_batch(n: int, seq_len: int, n_proto: int = 8, seed: int = 0) -> torch.Tensor:
    """Synthetic univariate series with real neighbour structure.

    Each series is one of ``n_proto`` sinusoidal prototypes (random frequency /
    phase) plus small noise, so nearest-neighbour sets are meaningful and stable
    rather than the degenerate all-noise case where MKNN is pure chance.

    Returns a ``(n, 1, seq_len)`` float tensor (the channels-first contract).
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 8 * np.pi, seq_len)
    freqs = rng.uniform(0.5, 3.0, size=n_proto)
    phases = rng.uniform(0, 2 * np.pi, size=n_proto)
    proto_id = rng.integers(0, n_proto, size=n)
    series = np.stack([
        np.sin(freqs[p] * t + phases[p]) + 0.05 * rng.standard_normal(seq_len)
        for p in proto_id
    ]).astype(np.float32)
    return torch.from_numpy(series)[:, None, :]  # (n, 1, seq_len)


def extract_layers(adapter, batch: torch.Tensor, granularity: str = "blocks"):
    """Return an ordered {layer_name: (n, d) numpy array} for one model.

    Layers whose row count is not the batch size are dropped: some models (e.g.
    PatchTST) expose internal modules that flatten the channel dim into the
    batch dim (B*C rows), which MKNN cannot compare row-to-row. This mirrors the
    row-count filtering in experiments_layerwise.py.
    """
    n = batch.shape[0]
    with torch.no_grad():
        layers = adapter.embed_all_layers_for_mode({"lc": batch}, "lc", granularity=granularity)
    return {
        name: z.detach().cpu().numpy()
        for name, z in layers.items()
        if z.shape[0] == n
    }


def check_determinism(adapter, batch: torch.Tensor, granularity: str, k: int) -> dict:
    """Forward twice; report max drift and per-layer self-MKNN (should be 1.0)."""
    a = extract_layers(adapter, batch, granularity)
    b = extract_layers(adapter, batch, granularity)
    max_drift = max(float(np.max(np.abs(a[name] - b[name]))) for name in a)
    self_mknn = [mknn(a[name], b[name], k=k) for name in a]
    return {
        "max_abs_drift": max_drift,
        "min_self_mknn": float(min(self_mknn)),
        "layers": list(a.keys()),
        "embeddings": a,
    }


def layerwise_cross_mknn(za: dict, zb: dict, k: int) -> np.ndarray:
    """Full (len(za) x len(zb)) MKNN matrix between two models' layers."""
    names_a, names_b = list(za), list(zb)
    m = np.zeros((len(names_a), len(names_b)))
    for i, na in enumerate(names_a):
        for j, nb in enumerate(names_b):
            m[i, j] = mknn(za[na], zb[nb], k=k)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-a", default="chronos")
    ap.add_argument("--repo-a", default="amazon/chronos-t5-tiny")
    ap.add_argument("--model-b", default="chronos")
    ap.add_argument("--repo-b", default="amazon/chronos-bolt-tiny")
    ap.add_argument("--n", type=int, default=96, help="Number of dummy series.")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--k", type=int, default=10, help="k for MKNN.")
    ap.add_argument("--granularity", default="blocks", choices=["blocks", "residual", "leaves", "all"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_deterministic(args.seed)
    batch = make_dummy_batch(args.n, args.seq_len, seed=args.seed)
    print(f"dummy batch: {tuple(batch.shape)}  (n, channels, seq_len)\n")

    results = {}
    for tag, model_alias, repo in [("A", args.model_a, args.repo_a), ("B", args.model_b, args.repo_b)]:
        print(f"[{tag}] loading {model_alias}: {repo}")
        adapter = get_adapter(model_alias)(repo, repo.split("/")[-1], alias=model_alias)
        adapter.load()
        det = check_determinism(adapter, batch, args.granularity, args.k)
        verdict = "OK" if det["max_abs_drift"] == 0.0 and det["min_self_mknn"] == 1.0 else "NON-DETERMINISTIC"
        print(f"    layers={len(det['layers'])}  "
              f"max|Δ| over 2 runs={det['max_abs_drift']:.2e}  "
              f"min self-MKNN={det['min_self_mknn']:.4f}  -> {verdict}\n")
        results[tag] = det

    za, zb = results["A"]["embeddings"], results["B"]["embeddings"]
    m = layerwise_cross_mknn(za, zb, args.k)
    names_a, names_b = list(za), list(zb)

    print(f"cross-model layer-wise MKNN (k={args.k})  rows=A layers, cols=B layers")
    print(f"  A={args.repo_a}   B={args.repo_b}\n")
    header = "A\\B".ljust(22) + "".join(f"{j:>7}" for j in range(len(names_b)))
    print(header)
    for i, na in enumerate(names_a):
        row = "".join(f"{m[i, j]:7.3f}" for j in range(len(names_b)))
        print(f"{na[:21].ljust(22)}{row}")

    # Headline numbers.
    i_last, j_last = len(names_a) - 1, len(names_b) - 1
    bi, bj = np.unravel_index(int(np.argmax(m)), m.shape)
    print("\nsummary")
    print(f"  final-vs-final MKNN : {m[i_last, j_last]:.4f}  "
          f"({names_a[i_last]} vs {names_b[j_last]})")
    print(f"  best-aligned pair   : {m[bi, bj]:.4f}  ({names_a[bi]} vs {names_b[bj]})")


if __name__ == "__main__":
    main()
