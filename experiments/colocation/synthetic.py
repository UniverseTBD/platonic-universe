"""
Synthetic "universe" with a *planted* manifold depth, for validating the
triple-co-location pipeline end to end without GPUs or model weights.

Why synthetic data is the right validation here: co-location is a claim about
recovering a hidden depth. The only way to know the pipeline recovers the right
answer is to plant a known one. Each "model" is a stack of per-layer embedding
matrices whose signal-to-noise peaks at a chosen relative depth; at that layer
the embedding is a low-dimensional linear image of a *shared* physical latent,
so by construction all three special depths coincide there:

  * cross-model neighbourhood agreement peaks (every model images the same
    latent, so their peak-layer neighbour graphs match)   -> L_conv
  * the intrinsic dimension is minimised (signal is low-dim, noise is not)  -> L*
  * overlap with the physical catalogue peaks             -> L_phys

The generator also emits two controls used in Step 7:

  * an *instrument-confounded* model whose peak images the **nuisance** catalogue
    Q instead of the physical catalogue P (physics overlap should stay flat while
    the nuisance curve peaks): the pipeline must flag it;
  * an *untrained* model that is isotropic noise at every layer (all calibrated
    scores should sit near zero everywhere).

Nothing here is astrophysics; it is a controlled test-bed with a known answer.
Run the real pipeline by pointing the notebook at the layerwise parquet files
from ``pu.experiments_layerwise.extract_all_layers`` and a matching catalogue.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Column names mirror the COSMOS-Web / Smith42 catalogues the real run uses.
PHYS_COLUMNS = ["redshift", "log_mass", "g_minus_r", "log_ssfr", "mag_r"]
NUISANCE_COLUMNS = ["seeing", "sky_background", "snr"]


@dataclass
class ModelSpec:
    name: str
    n_layers: int
    peak_rel: float          # planted relative depth of the signal peak
    embed_dim: int
    kind: str = "physics"    # "physics" | "nuisance" | "untrained"
    sigma: float = 0.16      # width of the signal bump in relative-depth units


# The four physics models co-locate but at *heterogeneous* depths (0.55–0.72):
# a deliberate reminder that depths must not be averaged across models before
# the coincidence test.
DEFAULT_MODELS: list[ModelSpec] = [
    ModelSpec("alpha", n_layers=12, peak_rel=0.55, embed_dim=64),
    ModelSpec("beta", n_layers=16, peak_rel=0.60, embed_dim=96),
    ModelSpec("gamma", n_layers=10, peak_rel=0.58, embed_dim=48),
    ModelSpec("delta", n_layers=14, peak_rel=0.72, embed_dim=72),
    ModelSpec("instr", n_layers=12, peak_rel=0.60, embed_dim=64, kind="nuisance"),
    ModelSpec("untrained", n_layers=12, peak_rel=0.60, embed_dim=64, kind="untrained"),
]

PHYSICS_MODELS = ["alpha", "beta", "gamma", "delta"]
CONTROL_MODELS = ["instr", "untrained"]


def _standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


def _make_catalog(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Correlated physical catalogue P and an independent nuisance catalogue Q."""
    # A few correlated latent drivers -> realistic covariance between columns.
    z = rng.uniform(0.05, 3.0, size=n)                     # redshift
    log_mass = 10.5 + 0.4 * z + rng.normal(0, 0.4, n)      # mass grows with z (selection)
    g_r = 0.6 + 0.25 * (log_mass - 10.5) + rng.normal(0, 0.15, n)  # redder if massive
    log_ssfr = -9.0 - 0.5 * (log_mass - 10.5) + rng.normal(0, 0.3, n)  # SF main sequence
    mag_r = 20.0 + 2.5 * z - 1.2 * (log_mass - 10.5) + rng.normal(0, 0.3, n)
    P = np.column_stack([z, log_mass, g_r, log_ssfr, mag_r])

    seeing = rng.uniform(0.5, 1.2, n)
    sky = rng.normal(0.0, 1.0, n)
    snr = rng.uniform(5.0, 60.0, n)
    Q = np.column_stack([seeing, sky, snr])
    return P, Q


def _bump(rel_depths: np.ndarray, peak: float, sigma: float) -> np.ndarray:
    return np.exp(-((rel_depths - peak) ** 2) / (2 * sigma**2))


def _model_layers(
    spec: ModelSpec,
    P_std: np.ndarray,
    Q_std: np.ndarray,
    rng: np.random.Generator,
    nuisance_leak: float = 0.15,
) -> list[np.ndarray]:
    """Build one model's per-layer embedding stack.

    ``X_l = w * signal + sqrt(1 - w^2) * noise`` with a small nuisance leak in
    the shallow layers (a mild instrument signature). ``w`` is a Gaussian bump
    centred on the planted relative depth.
    """
    n = P_std.shape[0]
    d = spec.embed_dim
    L = spec.n_layers
    rels = np.linspace(0.0, 1.0, L)
    w = _bump(rels, spec.peak_rel, spec.sigma)

    if spec.kind == "untrained":
        return [rng.standard_normal((n, d)) for _ in range(L)]

    latent = Q_std if spec.kind == "nuisance" else P_std
    p = latent.shape[1]
    R = rng.standard_normal((p, d)) / np.sqrt(p)      # random up-projection
    signal = _standardize(latent @ R)                  # unit-variance columns

    # Shallow-layer instrument signature (shared-ish nuisance), independent of P.
    S = rng.standard_normal((Q_std.shape[1], d)) / np.sqrt(Q_std.shape[1])
    leak = _standardize(Q_std @ S)

    layers = []
    for li in range(L):
        wl = w[li]
        noise = rng.standard_normal((n, d))
        early = (1.0 - rels[li])  # strongest in shallow layers
        x = (
            wl * signal
            + nuisance_leak * early * (1.0 - wl) * leak
            + np.sqrt(max(1e-6, 1.0 - wl**2)) * noise
        )
        layers.append(x.astype(np.float64))
    return layers


@dataclass
class SyntheticUniverse:
    P: np.ndarray                       # (n, p) raw physical catalogue
    P_std: np.ndarray                   # column-standardised
    Q: np.ndarray                       # (n, q) raw nuisance catalogue
    Q_std: np.ndarray
    models: dict[str, list[np.ndarray]]  # name -> [layer_0, layer_1, ...]
    specs: dict[str, ModelSpec]
    truth: dict[str, dict] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return self.P.shape[0]

    def physics_models(self) -> list[str]:
        return [n for n, s in self.specs.items() if s.kind == "physics"]

    def control_models(self) -> list[str]:
        return [n for n, s in self.specs.items() if s.kind != "physics"]


def make_universe(
    n: int = 2000,
    seed: int = 0,
    models: list[ModelSpec] | None = None,
) -> SyntheticUniverse:
    """Generate a full synthetic universe with a planted manifold depth per model."""
    rng = np.random.default_rng(seed)
    specs = models or DEFAULT_MODELS
    P, Q = _make_catalog(rng, n)
    P_std, Q_std = _standardize(P), _standardize(Q)

    model_layers: dict[str, list[np.ndarray]] = {}
    truth: dict[str, dict] = {}
    for spec in specs:
        model_layers[spec.name] = _model_layers(spec, P_std, Q_std, rng)
        rels = np.linspace(0.0, 1.0, spec.n_layers)
        peak_idx = int(np.argmin(np.abs(rels - spec.peak_rel)))
        truth[spec.name] = {
            "kind": spec.kind,
            "planted_peak_rel": spec.peak_rel,
            "planted_peak_layer": peak_idx,
            "n_layers": spec.n_layers,
        }

    return SyntheticUniverse(
        P=P,
        P_std=P_std,
        Q=Q,
        Q_std=Q_std,
        models=model_layers,
        specs={s.name: s for s in specs},
        truth=truth,
    )
