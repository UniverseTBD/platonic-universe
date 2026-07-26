"""
Tests for the triple co-location primitives (``pu.metrics.colocation``).

All tests are pure NumPy / scikit-learn: no torch, no network, no model
weights. The heavy correctness invariant is that the permutation *remap* used
to build the calibration null is identical to physically permuting the rows and
rebuilding the kNN graph, and that permutation calibration collapses a random
pair to ~0 while leaving a genuinely aligned pair with a large z-score.
"""

import os
import sys

import numpy as np
import pytest

from pu.metrics.colocation import (
    _overlap_from_indices,
    _permute_graph,
    bootstrap_depth,
    calibrate_overlap_graphs,
    cca_curve,
    colocation_test,
    cross_layer_matrix,
    intervals_mutually_overlap,
    intrinsic_dimension_curve,
    iqr,
    knn_indices,
    l2_normalize_rows,
    l_conv_from_pairs,
    manifold_layer,
    overlap,
    physics_overlap_curve,
    probe_r2_curve,
    relative_depth,
    summarize_colocation,
    twonn_dimension,
)

# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def shared_manifold():
    """Two 'models' that are different linear images of one shared latent."""
    rng = np.random.default_rng(0)
    n, d, p = 800, 24, 4
    Z = rng.standard_normal((n, p))
    A = Z @ rng.standard_normal((p, d)) + 0.05 * rng.standard_normal((n, d))
    B = Z @ rng.standard_normal((p, d)) + 0.05 * rng.standard_normal((n, d))
    R = rng.standard_normal((n, d))  # unrelated
    return Z, A, B, R


def _planted_stack(rng, Z, n_layers, peak, d, sigma=0.16):
    """A per-layer stack whose signal (a linear image of Z) peaks at ``peak``."""
    n, p = Z.shape
    Rp = rng.standard_normal((p, d)) / np.sqrt(p)
    sig = Z @ Rp
    sig = (sig - sig.mean(0)) / sig.std(0)
    rels = np.linspace(0, 1, n_layers)
    w = np.exp(-((rels - peak) ** 2) / (2 * sigma**2))
    return [w[i] * sig + np.sqrt(max(1e-6, 1 - w[i] ** 2)) * rng.standard_normal((n, d))
            for i in range(n_layers)]


# --------------------------------------------------------------------------- #
# normalisation & neighbour graphs
# --------------------------------------------------------------------------- #


def test_l2_normalize_rows_unit_norm():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 8)) * 10
    Xn = l2_normalize_rows(X)
    assert np.allclose(np.linalg.norm(Xn, axis=1), 1.0)


def test_l2_normalize_zero_row_safe():
    X = np.zeros((3, 4))
    Xn = l2_normalize_rows(X)
    assert np.all(np.isfinite(Xn))


def test_knn_indices_shape_excludes_self():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((40, 5))
    idx = knn_indices(X, k=7)
    assert idx.shape == (40, 7)
    for i, row in enumerate(idx):
        assert i not in row  # a point is never its own neighbour


def test_overlap_self_is_one(shared_manifold):
    _, A, _, _ = shared_manifold
    assert overlap(A, A, k=10, backend="numpy") == pytest.approx(1.0)


def test_overlap_aligned_high_random_low(shared_manifold):
    _, A, B, R = shared_manifold
    assert overlap(A, B, k=10, backend="numpy") > 0.4
    assert overlap(A, R, k=10, backend="numpy") < 0.05


def test_dadapy_backend_agrees_when_available(shared_manifold):
    # DADApy does not build against numpy>=2, so this is skipped on the repo's
    # pinned stack and only runs in a numpy<2 environment where dadapy exists.
    pytest.importorskip("dadapy")
    _, A, B, _ = shared_manifold
    num = overlap(A, B, k=10, backend="numpy")
    dad = overlap(A, B, k=10, backend="dadapy")
    assert abs(num - dad) < 0.1  # same quantity; only tie-breaking may differ


# --------------------------------------------------------------------------- #
# the permutation-remap invariant (core correctness)
# --------------------------------------------------------------------------- #


def test_permute_remap_equals_rebuild(shared_manifold):
    _, A, B, _ = shared_manifold
    rng = np.random.default_rng(3)
    perm = rng.permutation(A.shape[0])
    idx_a = knn_indices(A, k=10)
    idx_b = knn_indices(B, k=10)
    remap = _permute_graph(idx_b, perm)
    rebuild = knn_indices(B[perm], k=10)
    # Overlap against A must be identical for remap vs a full rebuild.
    assert _overlap_from_indices(idx_a, remap) == pytest.approx(
        _overlap_from_indices(idx_a, rebuild)
    )


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #


def test_calibration_kills_random_keeps_aligned(shared_manifold):
    _, A, B, R = shared_manifold
    idx_a = knn_indices(A, k=10)
    z_aligned = calibrate_overlap_graphs(idx_a, knn_indices(B, k=10), n_perm=10)
    z_random = calibrate_overlap_graphs(idx_a, knn_indices(R, k=10), n_perm=10)
    assert z_aligned.calibrated > 30           # genuine alignment survives
    assert abs(z_random.calibrated) < 5        # chance alignment is calibrated away
    assert z_aligned.null_mean < z_aligned.raw  # the null sits below the real overlap


def test_calibration_null_is_small_overlap(shared_manifold):
    _, A, B, _ = shared_manifold
    cs = calibrate_overlap_graphs(knn_indices(A, k=10), knn_indices(B, k=10), n_perm=10)
    # Shuffling rows destroys correspondence, so the null overlap is tiny.
    assert cs.null_mean < 0.05


# --------------------------------------------------------------------------- #
# cross-layer matrix & L_conv
# --------------------------------------------------------------------------- #


def test_cross_layer_peak_at_planted_depth():
    rng = np.random.default_rng(4)
    Z = rng.standard_normal((600, 4))
    la = _planted_stack(rng, Z, n_layers=9, peak=0.5, d=32)
    lb = _planted_stack(rng, Z, n_layers=11, peak=0.5, d=40)
    res = cross_layer_matrix(la, lb, k=10, n_perm=10)
    # Both models plant their signal near the middle layer.
    assert res.peak_layer_a in (3, 4, 5)
    assert res.peak_layer_b in (4, 5, 6)
    assert res.calibrated.shape == (9, 11)


def test_l_conv_from_pairs_median():
    peaks = {"m": [2, 4, 6], "n": [5, 5]}
    out = l_conv_from_pairs(peaks, {"m": 11, "n": 11})
    assert out["m"]["layer"] == 4.0
    assert out["m"]["relative"] == pytest.approx(4 / 10)
    assert out["n"]["layer"] == 5.0


# --------------------------------------------------------------------------- #
# physics overlap curve & supporting probes
# --------------------------------------------------------------------------- #


def test_physics_curve_peaks_at_planted_layer():
    rng = np.random.default_rng(5)
    Z = rng.standard_normal((800, 4))
    layers = _planted_stack(rng, Z, n_layers=11, peak=0.6, d=32)
    P_std = (Z - Z.mean(0)) / Z.std(0)
    curve = physics_overlap_curve(layers, P_std, k=10, n_perm=10)
    planted = int(round(0.6 * 10))
    assert abs(int(np.argmax(curve["calibrated"])) - planted) <= 1
    assert curve["calibrated"].max() > 20


def test_probe_and_cca_curves_run():
    rng = np.random.default_rng(6)
    Z = rng.standard_normal((400, 3))
    layers = _planted_stack(rng, Z, n_layers=6, peak=0.6, d=16)
    P = (Z - Z.mean(0)) / Z.std(0)
    r2 = probe_r2_curve(layers, P)
    cc = cca_curve(layers, P)
    assert r2.shape == (6,) and cc.shape == (6,)
    # The probe should explain more variance at the signal peak than at layer 0.
    assert np.nanmax(r2) > r2[0]


# --------------------------------------------------------------------------- #
# intrinsic dimension / L*
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("p", [2, 4, 8])
def test_twonn_recovers_dimension(p):
    rng = np.random.default_rng(7)
    X = rng.standard_normal((3000, p))
    est = twonn_dimension(X, normalize_rows=False)
    assert est == pytest.approx(p, rel=0.2)


def test_manifold_layer_is_argmin_of_id():
    rng = np.random.default_rng(8)
    Z = rng.standard_normal((800, 4))
    layers = _planted_stack(rng, Z, n_layers=11, peak=0.5, d=48)
    idc = intrinsic_dimension_curve(layers)
    # Intrinsic dimension is lowest where the low-dim signal dominates.
    assert manifold_layer(idc) in (4, 5, 6)


# --------------------------------------------------------------------------- #
# depth helpers, bootstrap, coincidence test
# --------------------------------------------------------------------------- #


def test_relative_depth_and_iqr():
    assert relative_depth(0, 12) == 0.0
    assert relative_depth(11, 12) == 1.0
    assert iqr([1, 2, 3, 4, 5]) == pytest.approx(2.0)


def test_bootstrap_depth_structure():
    b = bootstrap_depth(lambda idx: 0.5 + 0.001 * len(idx) * 0, n_rows=200, seeds=range(10))
    assert len(b.estimates) == 10
    assert b.iqr >= 0
    assert b.q25 <= b.median <= b.q75


def test_intervals_mutually_overlap():
    assert intervals_mutually_overlap([(0.1, 0.3), (0.2, 0.4), (0.25, 0.5)])
    assert not intervals_mutually_overlap([(0.1, 0.2), (0.3, 0.4), (0.15, 0.35)])


def test_colocation_test_pass_and_fail():
    good = colocation_test(
        {"conv": 0.60, "star": 0.62, "phys": 0.58},
        {"conv": (0.58, 0.62), "star": (0.60, 0.64), "phys": (0.56, 0.60)},
        gap_threshold=0.10,
    )
    assert good.passed and good.max_gap <= 0.10 and good.iqrs_overlap

    far = colocation_test(
        {"conv": 0.30, "star": 0.60, "phys": 0.85},
        {"conv": (0.28, 0.32), "star": (0.58, 0.62), "phys": (0.83, 0.87)},
        gap_threshold=0.10,
    )
    assert not far.passed and far.max_gap > 0.10


def test_colocation_fails_when_intervals_disjoint():
    # Gaps are tiny but the bootstrap intervals do not overlap -> not co-located.
    v = colocation_test(
        {"conv": 0.50, "star": 0.52, "phys": 0.54},
        {"conv": (0.499, 0.501), "star": (0.519, 0.521), "phys": (0.539, 0.541)},
        gap_threshold=0.10,
    )
    assert v.max_gap <= 0.10
    assert not v.iqrs_overlap
    assert not v.passed


def test_summarize_colocation_fraction():
    v = {
        "a": colocation_test({"conv": 0.5, "star": 0.5, "phys": 0.5},
                             {"conv": (0.49, 0.51), "star": (0.49, 0.51), "phys": (0.49, 0.51)}),
        "b": colocation_test({"conv": 0.2, "star": 0.5, "phys": 0.8},
                             {"conv": (0.19, 0.21), "star": (0.49, 0.51), "phys": (0.79, 0.81)}),
    }
    s = summarize_colocation(v)
    assert s["n_models"] == 2
    assert s["fraction_passing"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# integration: the planted universe co-locates; controls do not
# --------------------------------------------------------------------------- #


def _load_synthetic():
    colo = os.path.join(os.path.dirname(__file__), "..", "experiments", "colocation")
    sys.path.insert(0, os.path.abspath(colo))
    import synthetic  # noqa: E402

    return synthetic


def test_integration_physics_model_colocates():
    synthetic = _load_synthetic()
    U = synthetic.make_universe(n=1200, seed=0)
    m = "alpha"
    layers = U.models[m]
    L = len(layers)
    planted = U.truth[m]["planted_peak_layer"]

    l_phys = int(np.argmax(physics_overlap_curve(layers, U.P_std, k=10, n_perm=10)["calibrated"]))
    l_star = manifold_layer(intrinsic_dimension_curve(layers))
    assert abs(l_phys - planted) <= 1
    assert abs(l_star - planted) <= 1
    # The three depths land within the co-location gap threshold of each other.
    d_phys, d_star = relative_depth(l_phys, L), relative_depth(l_star, L)
    assert abs(d_phys - d_star) <= 0.10


def test_integration_untrained_control_has_no_alignment():
    synthetic = _load_synthetic()
    U = synthetic.make_universe(n=1200, seed=0)
    layers = U.models["untrained"]
    phys = physics_overlap_curve(layers, U.P_std, k=10, n_perm=10)
    nuis = physics_overlap_curve(layers, U.Q_std, k=10, n_perm=10)
    assert phys["calibrated"].max() < 6
    assert nuis["calibrated"].max() < 6


def test_integration_instrument_control_flagged():
    synthetic = _load_synthetic()
    U = synthetic.make_universe(n=1200, seed=0)
    layers = U.models["instr"]
    phys = physics_overlap_curve(layers, U.P_std, k=10, n_perm=10)
    nuis = physics_overlap_curve(layers, U.Q_std, k=10, n_perm=10)
    # Instrument-confounded: nuisance alignment should dominate physics.
    assert nuis["calibrated"].max() > phys["calibrated"].max()
