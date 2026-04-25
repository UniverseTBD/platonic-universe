"""Coverage for two cluster-driven fixes:

1. transformers >=5 wraps ``CLIPModel.get_image_features`` in
   ``BaseModelOutputWithPooling`` instead of returning a tensor; the
   helper ``_clip_image_features`` must unwrap to the projected feature
   tensor (``pooler_output``).

2. llava-onevision returns multi-patch ``pixel_values`` shaped
   ``(B, P, C, H, W)``; ``VLMAdapter._prepare_vlm_inputs`` must accept
   that without crashing on ``permute(1, 2, 0)`` of a 4D slice.

Tests use stubs so they never download a model or touch HF.
"""
import torch
import torch.nn as nn

from pu.models.hf import VLMAdapter, _clip_image_features

# ── (1) _clip_image_features unwrap ──────────────────────────────────────

class _ClipModelStubTensor(nn.Module):
    """Mimics transformers 4.x: get_image_features returns a Tensor."""

    def __init__(self, B: int = 2, D: int = 512):
        super().__init__()
        self._B = B
        self._D = D

    def get_image_features(self, pixel_values):
        return torch.randn(self._B, self._D)


class _ClipModelStubBMOP(nn.Module):
    """Mimics transformers >=5: returns BaseModelOutputWithPooling."""

    def __init__(self, B: int = 2, D: int = 512):
        super().__init__()
        self._B = B
        self._D = D

    def get_image_features(self, pixel_values):
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        return BaseModelOutputWithPooling(
            last_hidden_state=torch.randn(self._B, 197, 768),
            pooler_output=torch.randn(self._B, self._D),
        )


def test_clip_image_features_passthrough_when_already_tensor():
    """transformers 4.x path — input already a tensor."""
    m = _ClipModelStubTensor(B=3, D=512)
    out = _clip_image_features(m, torch.randn(3, 3, 224, 224))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 512)


def test_clip_image_features_unwraps_bmop():
    """transformers >=5 path — picks pooler_output (the projected features)."""
    m = _ClipModelStubBMOP(B=2, D=512)
    out = _clip_image_features(m, torch.randn(2, 3, 224, 224))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 512), "should pick pooler_output, not last_hidden_state"


def test_clip_image_features_raises_on_unrecognized():
    """If the wrapper has no recognised tensor field, raise — don't return a
    ModelOutput that would silently break callers downstream."""

    class _Weird:
        def get_image_features(self, pixel_values):
            class _O:
                pass
            return _O()

    import pytest
    with pytest.raises(TypeError, match="recognizable image-feature tensor"):
        _clip_image_features(_Weird(), torch.randn(1, 3, 224, 224))


# ── (2) VLMAdapter._prepare_vlm_inputs handles 5D pixel_values ───────────

class _StubProcessor:
    """Returns the dict shape _prepare_vlm_inputs expects after re-encoding."""

    def __call__(self, images, text, return_tensors=None, padding=None):
        B = len(images)
        return {
            "input_ids":      torch.zeros(B, 5, dtype=torch.long),
            "attention_mask": torch.ones(B, 5, dtype=torch.long),
            "pixel_values":   torch.zeros(B, 3, 384, 384),
        }


def _make_stub_vlm(alias: str = "llava_ov_7b") -> VLMAdapter:
    """Build a VLMAdapter without going through HF model loading."""
    stub = VLMAdapter.__new__(VLMAdapter)
    stub.alias = alias
    stub.model = nn.Linear(4, 4)  # gives _prepare_vlm_inputs a device + dtype
    stub.processor = _StubProcessor()
    return stub


def test_prepare_vlm_inputs_accepts_5d_multipatch():
    """llava-onevision shape — was crashing on permute(1, 2, 0) of 4D slice."""
    stub = _make_stub_vlm("llava_ov_7b")
    pv = torch.randn(2, 5, 3, 384, 384)  # (B, P, C, H, W)
    ids, pv_enc, attn = stub._prepare_vlm_inputs({"hsc": pv}, "hsc")
    assert ids.shape == (2, 5)
    assert pv_enc.shape == (2, 3, 384, 384)
    assert attn.shape == (2, 5)


def test_prepare_vlm_inputs_still_handles_4d():
    """paligemma / llava-1.5 shape — the common case must not regress."""
    stub = _make_stub_vlm("paligemma_3b")
    pv = torch.randn(2, 3, 224, 224)
    ids, pv_enc, attn = stub._prepare_vlm_inputs({"hsc": pv}, "hsc")
    assert ids.shape == (2, 5)
    assert pv_enc.shape == (2, 3, 384, 384)
