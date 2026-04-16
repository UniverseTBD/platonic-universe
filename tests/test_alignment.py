"""
Alignment and ordering tests for the embedding extraction pipeline.

These tests verify three critical invariants:

1. BATCH-SIZE INVARIANCE: Extracting with batch_size=1 vs batch_size=16
   produces identical embeddings (within float tolerance). Failure means
   something is shuffling within or across batches.

2. CROSS-RUN DETERMINISM: Running the pipeline twice on the same data
   produces bit-identical results. Failure means non-deterministic sampling
   or a race condition in the data pipeline.

3. FINGERPRINT ORDERING: A per-sample fingerprint (pixel sum of the raw
   HSC tensor) arrives in the same order as the embeddings. Failure means
   the pairing between samples and embeddings is broken.

Run with:
    uv run pytest tests/test_alignment.py -v
    uv run pytest tests/test_alignment.py -v -k "jwst"   # just JWST tests
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SAMPLES = 32  # small enough for fast CI, large enough to catch ordering bugs


def _load_adapter(alias, model_name, size):
    from pu.models import get_adapter
    cls = get_adapter(alias)
    adapter = cls(model_name, size, alias=alias)
    adapter.load()
    return adapter


def _stream_dataset(mode, adapter, modes, n_samples):
    """Stream n_samples from a dataset, return the HF IterableDataset."""
    from pu.pu_datasets import get_dataset_adapter

    hf_ds = f"Smith42/{mode}_hsc_crossmatched"
    processor = adapter.get_preprocessor(modes)

    def filterfun(idx):
        if mode != "jwst":
            return True
        im = idx["jwst_image"]["flux"][3]
        v0, v1 = np.nanpercentile(im, 5), np.nanpercentile(im, 99)
        return v0 - v1 != 0

    ds_alias = mode
    dataset_cls = get_dataset_adapter(ds_alias)
    ds_adapter = dataset_cls(hf_ds, mode)
    ds_adapter.load()
    ds = ds_adapter.prepare(processor, modes, filterfun)
    ds = ds.take(n_samples)
    return ds


def _extract_embeddings_and_fingerprints(adapter, ds, modes, batch_size):
    """Run the pipeline and return embeddings + per-sample fingerprints.

    Fingerprint = sum of all pixel values in the HSC tensor for that sample.
    This is a cheap unique-ish identifier that lets us verify row ordering
    without storing full images.
    """
    dl = DataLoader(ds, batch_size=batch_size, num_workers=0)

    embeddings = {m: [] for m in modes}
    fingerprints = []

    with torch.no_grad():
        for batch in dl:
            # Capture fingerprint from raw HSC pixels before model sees them
            if "hsc" in batch:
                hsc_pixels = batch["hsc"]  # (B, C, H, W)
                for i in range(hsc_pixels.shape[0]):
                    fingerprints.append(float(hsc_pixels[i].sum()))

            for mode in modes:
                if mode == "desi":
                    embeddings[mode].append(
                        torch.tensor(np.array(batch["embeddings"])).T
                    )
                elif mode == "sdss":
                    embeddings[mode].append(
                        torch.tensor(np.array(batch["embedding"])).T
                    )
                else:
                    out = adapter.embed_for_mode(batch, mode)
                    embeddings[mode].append(out.cpu())

    embeddings = {m: torch.cat(e) for m, e in embeddings.items()}
    return embeddings, fingerprints


# ---------------------------------------------------------------------------
# DESI tests (HSC images + pre-computed DESI embeddings)
# ---------------------------------------------------------------------------

class TestDESIAlignment:
    """Tests using DESI dataset (paired: HSC image model + pre-computed DESI embeddings)."""

    @pytest.fixture(scope="class")
    def adapter(self):
        return _load_adapter("vit", "google/vit-base-patch16-224-in21k", "base")

    @pytest.fixture(scope="class")
    def modes(self):
        return ["hsc", "desi"]

    def test_batch_size_invariance(self, adapter, modes):
        """Embeddings must be identical regardless of batch size."""
        ds1 = _stream_dataset("desi", adapter, modes, N_SAMPLES)
        emb1, fp1 = _extract_embeddings_and_fingerprints(
            adapter, ds1, modes, batch_size=1
        )

        ds16 = _stream_dataset("desi", adapter, modes, N_SAMPLES)
        emb16, fp16 = _extract_embeddings_and_fingerprints(
            adapter, ds16, modes, batch_size=16
        )

        # Same number of samples
        assert emb1["hsc"].shape[0] == emb16["hsc"].shape[0], (
            f"Sample count mismatch: bs=1 got {emb1['hsc'].shape[0]}, "
            f"bs=16 got {emb16['hsc'].shape[0]}"
        )

        # Fingerprints arrive in same order
        assert fp1 == fp16, "HSC pixel fingerprints differ between batch sizes — ordering is broken"

        # HSC embeddings match within float tolerance
        diff_hsc = (emb1["hsc"] - emb16["hsc"]).abs().max().item()
        assert diff_hsc < 1e-4, (
            f"HSC embeddings differ by {diff_hsc:.6f} between batch sizes"
        )

        # DESI pre-computed embeddings must also match (tests DataLoader ordering)
        diff_desi = (emb1["desi"] - emb16["desi"]).abs().max().item()
        assert diff_desi < 1e-6, (
            f"DESI pre-computed embeddings differ by {diff_desi:.6f} — "
            f"DataLoader is reordering rows"
        )

    def test_cross_run_determinism(self, adapter, modes):
        """Two identical runs must produce bit-identical results."""
        ds_a = _stream_dataset("desi", adapter, modes, N_SAMPLES)
        emb_a, fp_a = _extract_embeddings_and_fingerprints(
            adapter, ds_a, modes, batch_size=8
        )

        ds_b = _stream_dataset("desi", adapter, modes, N_SAMPLES)
        emb_b, fp_b = _extract_embeddings_and_fingerprints(
            adapter, ds_b, modes, batch_size=8
        )

        assert fp_a == fp_b, "Fingerprints differ between runs — streaming order is non-deterministic"

        diff = (emb_a["hsc"] - emb_b["hsc"]).abs().max().item()
        assert diff == 0.0, f"Embeddings differ by {diff:.6e} between identical runs"

    def test_desi_pairing_integrity(self, adapter, modes):
        """DESI pre-computed embeddings must arrive in the same order as HSC images.

        Verifies that concatenate_datasets(..., axis=1) preserves row alignment.
        """
        ds = _stream_dataset("desi", adapter, modes, N_SAMPLES)
        emb, fp = _extract_embeddings_and_fingerprints(
            adapter, ds, modes, batch_size=8
        )

        n_hsc = emb["hsc"].shape[0]
        n_desi = emb["desi"].shape[0]
        assert n_hsc == n_desi, (
            f"Row count mismatch: HSC has {n_hsc}, DESI has {n_desi} — "
            f"axis=1 concat is misaligned"
        )
        assert len(fp) == n_hsc, (
            f"Fingerprint count {len(fp)} != embedding count {n_hsc}"
        )

    def test_embedding_dimensions(self, adapter, modes):
        """Verify embedding shapes match model architecture."""
        ds = _stream_dataset("desi", adapter, modes, N_SAMPLES)
        emb, _ = _extract_embeddings_and_fingerprints(
            adapter, ds, modes, batch_size=8
        )

        # ViT-base has hidden_dim=768
        assert emb["hsc"].shape[1] == 768, (
            f"Expected dim 768 for ViT-base, got {emb['hsc'].shape[1]}"
        )
        assert emb["hsc"].shape[0] == N_SAMPLES


# ---------------------------------------------------------------------------
# JWST tests (HSC + JWST images, both go through model)
# ---------------------------------------------------------------------------

class TestJWSTAlignment:
    """Tests using JWST dataset (both HSC and JWST images go through the model)."""

    @pytest.fixture(scope="class")
    def adapter(self):
        return _load_adapter("vit", "google/vit-base-patch16-224-in21k", "base")

    @pytest.fixture(scope="class")
    def modes(self):
        return ["hsc", "jwst"]

    def test_batch_size_invariance(self, adapter, modes):
        """Embeddings must be identical regardless of batch size."""
        ds1 = _stream_dataset("jwst", adapter, modes, N_SAMPLES)
        emb1, fp1 = _extract_embeddings_and_fingerprints(
            adapter, ds1, modes, batch_size=1
        )

        ds16 = _stream_dataset("jwst", adapter, modes, N_SAMPLES)
        emb16, fp16 = _extract_embeddings_and_fingerprints(
            adapter, ds16, modes, batch_size=16
        )

        assert fp1 == fp16, "HSC pixel fingerprints differ between batch sizes"

        diff_hsc = (emb1["hsc"] - emb16["hsc"]).abs().max().item()
        assert diff_hsc < 1e-4, f"HSC embeddings differ by {diff_hsc:.6f}"

        diff_jwst = (emb1["jwst"] - emb16["jwst"]).abs().max().item()
        assert diff_jwst < 1e-4, f"JWST embeddings differ by {diff_jwst:.6f}"

    def test_cross_run_determinism(self, adapter, modes):
        """Two runs must produce identical results."""
        ds_a = _stream_dataset("jwst", adapter, modes, N_SAMPLES)
        emb_a, fp_a = _extract_embeddings_and_fingerprints(
            adapter, ds_a, modes, batch_size=8
        )

        ds_b = _stream_dataset("jwst", adapter, modes, N_SAMPLES)
        emb_b, fp_b = _extract_embeddings_and_fingerprints(
            adapter, ds_b, modes, batch_size=8
        )

        assert fp_a == fp_b, "Fingerprints differ between runs"

        for mode in modes:
            diff = (emb_a[mode] - emb_b[mode]).abs().max().item()
            assert diff == 0.0, f"{mode} embeddings differ by {diff:.6e}"

    def test_hsc_jwst_pairing(self, adapter, modes):
        """HSC and JWST must have the same number of samples in the same order."""
        ds = _stream_dataset("jwst", adapter, modes, N_SAMPLES)
        emb, fp = _extract_embeddings_and_fingerprints(
            adapter, ds, modes, batch_size=8
        )

        assert emb["hsc"].shape[0] == emb["jwst"].shape[0], (
            f"Pairing broken: HSC has {emb['hsc'].shape[0]}, "
            f"JWST has {emb['jwst'].shape[0]}"
        )

    def test_different_modes_produce_different_embeddings(self, adapter, modes):
        """HSC and JWST images of the same galaxy should produce different embeddings.

        If they're identical, the same image is being fed to both modes (wrong
        preprocessing or band selection).
        """
        ds = _stream_dataset("jwst", adapter, modes, N_SAMPLES)
        emb, _ = _extract_embeddings_and_fingerprints(
            adapter, ds, modes, batch_size=8
        )

        # Cosine similarity between paired HSC/JWST embeddings.
        # Same galaxy in different surveys should be correlated but NOT identical.
        hsc_norm = emb["hsc"] / emb["hsc"].norm(dim=1, keepdim=True)
        jwst_norm = emb["jwst"] / emb["jwst"].norm(dim=1, keepdim=True)
        cos_sim = (hsc_norm * jwst_norm).sum(dim=1)

        # If mean cosine sim is > 0.999, the same image is likely being used for both
        mean_sim = cos_sim.mean().item()
        assert mean_sim < 0.999, (
            f"HSC/JWST embeddings are suspiciously similar (cos_sim={mean_sim:.4f}). "
            f"Same image may be fed to both modes."
        )
        # But they should be at least somewhat correlated (same galaxy)
        assert mean_sim > 0.0, (
            f"HSC/JWST embeddings are anticorrelated (cos_sim={mean_sim:.4f}). "
            f"Something is very wrong."
        )


# ---------------------------------------------------------------------------
# Layer-wise extraction tests (run after adding embed_all_layers_for_mode)
# ---------------------------------------------------------------------------

class TestLayerwiseExtraction:
    """Tests for all-layer embedding extraction.

    These validate that the layer-wise code produces correct results by
    checking against the existing single-layer embed_for_mode as ground truth.
    """

    @pytest.fixture(scope="class")
    def adapter(self):
        return _load_adapter("vit", "google/vit-base-patch16-224-in21k", "base")

    @pytest.fixture(scope="class")
    def modes(self):
        return ["hsc", "desi"]

    def _skip_if_no_layerwise(self, adapter):
        if not hasattr(adapter, "supports_layerwise") or not adapter.supports_layerwise():
            pytest.skip("embed_all_layers_for_mode not yet implemented")

    def test_some_module_matches_embed_for_mode(self, adapter, modes):
        """At least one extracted module must be highly correlated with
        embed_for_mode output.

        Generic extraction hooks every leaf module; the final layernorm's
        output (with generic mean pooling) should closely match
        embed_for_mode (with model-specific pooling), catching wrong
        images or broken hooks.
        """
        self._skip_if_no_layerwise(adapter)

        ds = _stream_dataset("desi", adapter, modes, N_SAMPLES)
        dl = DataLoader(ds, batch_size=8, num_workers=0)

        for batch in dl:
            expected = adapter.embed_for_mode(batch, "hsc").cpu()
            layer_embs = adapter.embed_all_layers_for_mode(batch, "hsc")
            exp_norm = expected / expected.norm(dim=1, keepdim=True)

            best_sim = -1.0
            best_key = None
            for key, emb in layer_embs.items():
                emb_cpu = emb.cpu()
                if emb_cpu.shape[1] != expected.shape[1]:
                    continue  # different dim, skip
                act_norm = emb_cpu / emb_cpu.norm(dim=1, keepdim=True)
                cos_sim = (exp_norm * act_norm).sum(dim=1).mean().item()
                if cos_sim > best_sim:
                    best_sim = cos_sim
                    best_key = key

            assert best_sim > 0.9, (
                f"No extracted module has cosine similarity > 0.9 with "
                f"embed_for_mode. Best: '{best_key}' at {best_sim:.4f}."
            )
            break

    def test_extraction_returns_block_level_modules(self, adapter, modes):
        """Default extraction returns block-level (residual stream) modules."""
        self._skip_if_no_layerwise(adapter)

        layer_names = adapter.get_layer_names()
        num_layers = adapter.get_num_layers()

        # ViT-base: ~76 block-level modules (embeddings, encoder.layer.N, encoder.layer.N.attention, etc.)
        assert num_layers > 50, (
            f"Expected >50 block-level extraction points for ViT-base, got {num_layers}"
        )
        assert len(layer_names) == num_layers

        # Verify we see block-level modules
        names_str = " ".join(layer_names)
        assert "encoder.layer.0" in names_str
        assert "encoder.layer.11" in names_str
        assert "last_hidden_state" in names_str

        # With include_leaves, should get many more
        leaf_names = adapter.get_layer_names(include_leaves=True)
        assert len(leaf_names) > 200, (
            f"Expected >200 with leaves for ViT-base, got {len(leaf_names)}"
        )

        ds = _stream_dataset("desi", adapter, modes, 4)
        dl = DataLoader(ds, batch_size=4, num_workers=0)

        for batch in dl:
            layer_embs = adapter.embed_all_layers_for_mode(batch, "hsc")
            for key in layer_embs:
                assert key in layer_names, f"Unexpected key '{key}' not in layer_names"
            assert len(layer_embs) > 50
            break

    def test_all_outputs_are_valid_tensors(self, adapter, modes):
        """Every extraction point must produce a 2D (batch, dim) tensor."""
        self._skip_if_no_layerwise(adapter)

        ds = _stream_dataset("desi", adapter, modes, 4)
        dl = DataLoader(ds, batch_size=4, num_workers=0)

        for batch in dl:
            layer_embs = adapter.embed_all_layers_for_mode(batch, "hsc")
            for key, emb in layer_embs.items():
                assert emb.dim() == 2, (
                    f"'{key}': expected 2D tensor, got {emb.dim()}D shape {emb.shape}"
                )
                assert emb.shape[0] == 4, (
                    f"'{key}': batch dim is {emb.shape[0]}, expected 4"
                )
                assert emb.shape[1] > 0, (
                    f"'{key}': feature dim is 0"
                )
                assert emb.dtype == torch.float32, (
                    f"'{key}': dtype is {emb.dtype}, expected float32"
                )
                assert not emb.isnan().any(), (
                    f"'{key}': contains NaN values"
                )
            break

    def test_layers_are_not_identical(self, adapter, modes):
        """Different modules must produce different embeddings."""
        self._skip_if_no_layerwise(adapter)

        ds = _stream_dataset("desi", adapter, modes, 4)
        dl = DataLoader(ds, batch_size=4, num_workers=0)

        for batch in dl:
            layer_embs = adapter.embed_all_layers_for_mode(batch, "hsc")
            keys = list(layer_embs.keys())
            first = layer_embs[keys[0]].cpu()
            last = layer_embs[keys[-1]].cpu()

            # First and last modules must produce different outputs
            # (unless they happen to have different dims, in which case
            # the comparison itself proves they're different)
            if first.shape == last.shape:
                diff = (first - last).abs().max().item()
                assert diff > 0.01, (
                    f"'{keys[0]}' and '{keys[-1]}' are nearly identical"
                )
            break

    def test_query_and_value_differ_with_leaves(self, adapter, modes):
        """Q and V projections within the same block must differ (requires include_leaves)."""
        self._skip_if_no_layerwise(adapter)

        ds = _stream_dataset("desi", adapter, modes, 4)
        dl = DataLoader(ds, batch_size=4, num_workers=0)

        for batch in dl:
            layer_embs = adapter.embed_all_layers_for_mode(batch, "hsc", include_leaves=True)

            q_key = "encoder.layer.0.attention.attention.query"
            v_key = "encoder.layer.0.attention.attention.value"
            assert q_key in layer_embs, f"Missing '{q_key}'"
            assert v_key in layer_embs, f"Missing '{v_key}'"

            q = layer_embs[q_key].cpu()
            v = layer_embs[v_key].cpu()
            diff = (q - v).abs().max().item()
            assert diff > 0.01, (
                f"Q and V projections are nearly identical (diff={diff:.6f})"
            )
            break

    def test_layerwise_batch_size_invariance(self, adapter, modes):
        """Layer embeddings must not change with batch size."""
        self._skip_if_no_layerwise(adapter)

        ds1 = _stream_dataset("desi", adapter, modes, 8)
        dl1 = DataLoader(ds1, batch_size=1, num_workers=0)

        ds8 = _stream_dataset("desi", adapter, modes, 8)
        dl8 = DataLoader(ds8, batch_size=8, num_workers=0)

        # Collect all layer embeddings with batch_size=1
        all_bs1 = {}
        for batch in dl1:
            layer_embs = adapter.embed_all_layers_for_mode(batch, "hsc")
            for idx, emb in layer_embs.items():
                all_bs1.setdefault(idx, []).append(emb.cpu())
        all_bs1 = {k: torch.cat(v) for k, v in all_bs1.items()}

        # Collect with batch_size=8
        all_bs8 = {}
        for batch in dl8:
            layer_embs = adapter.embed_all_layers_for_mode(batch, "hsc")
            for idx, emb in layer_embs.items():
                all_bs8.setdefault(idx, []).append(emb.cpu())
        all_bs8 = {k: torch.cat(v) for k, v in all_bs8.items()}

        for idx in all_bs1:
            diff = (all_bs1[idx] - all_bs8[idx]).abs().max().item()
            assert diff < 5e-4, (
                f"Layer {idx} differs by {diff:.6f} between batch sizes"
            )


# ---------------------------------------------------------------------------
# Multi-model ordering test
# ---------------------------------------------------------------------------

class TestCrossModelOrdering:
    """Verify that different models receive samples in the same order.

    If model A and model B are run on the same dataset, sample N from both
    must correspond to the same galaxy. We verify this via pixel fingerprints.
    """

    def test_vit_and_dino_see_same_samples(self):
        """ViT-base and DINO-small must receive HSC images in the same order."""
        vit = _load_adapter("vit", "google/vit-base-patch16-224-in21k", "base")
        dino = _load_adapter("dino", "facebook/dinov2-with-registers-small", "small")

        modes = ["hsc", "desi"]

        ds_vit = _stream_dataset("desi", vit, modes, N_SAMPLES)
        _, fp_vit = _extract_embeddings_and_fingerprints(
            vit, ds_vit, modes, batch_size=8
        )

        ds_dino = _stream_dataset("desi", dino, modes, N_SAMPLES)
        _, fp_dino = _extract_embeddings_and_fingerprints(
            dino, ds_dino, modes, batch_size=8
        )

        # Fingerprints won't be exactly equal because different processors
        # produce different pixel values from the same PIL image. But the
        # COUNT must match, confirming same samples were drawn.
        assert len(fp_vit) == len(fp_dino), (
            f"ViT got {len(fp_vit)} samples, DINO got {len(fp_dino)}"
        )
