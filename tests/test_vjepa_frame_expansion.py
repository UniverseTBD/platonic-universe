"""Reproducibility check: the tensor reaching the vjepa model is identical
before and after moving the 16× temporal repeat out of the preprocessor.

Main (pre-PR):
    proc_out["pixel_values_videos"].repeat(1, 16, 1, 1, 1).squeeze()
    -> DataLoader stacks -> (B, 16, 3, H, W) -> model

This PR:
    proc_out["pixel_values_videos"].squeeze(0)
    -> DataLoader stacks -> (B, 1, 3, H, W)
    -> HFAdapter._maybe_expand_video_frames -> (B, 16, 3, H, W) -> model

No model or processor download — we synthesize the processor's
`pixel_values_videos` output directly, which is the only contract the
preprocessor depends on.
"""
import torch

from pu.models.hf import HFAdapter


def _fake_proc_out(seed: int = 0, H: int = 256, W: int = 256) -> torch.Tensor:
    """Shape matches what AutoVideoProcessor returns for a single still image:
    (batch=1, T_native=1, C=3, H, W)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 1, 3, H, W, generator=g)


def _old_preprocess(proc_out: torch.Tensor) -> torch.Tensor:
    """Reproduces the pre-PR preprocess branch."""
    return proc_out.repeat(1, 16, 1, 1, 1).squeeze()


def _new_preprocess(proc_out: torch.Tensor) -> torch.Tensor:
    """Reproduces the post-PR preprocess branch."""
    return proc_out.squeeze(0)


def _stack_batch(per_sample_tensors):
    """DataLoader's default collate stacks along a new leading batch dim."""
    return torch.stack(per_sample_tensors, dim=0)


class _VjepaStub(HFAdapter):
    """Bypass HFAdapter.__init__'s model/processor loading — we only need
    _maybe_expand_video_frames, which depends solely on self.alias."""

    def __init__(self):
        self.alias = "vjepa"

    def load(self, compile_model: bool = False):
        raise NotImplementedError  # not used

    def get_preprocessor(self, *a, **k):
        raise NotImplementedError  # not used

    def embed_for_mode(self, *a, **k):
        raise NotImplementedError  # not used


def test_vjepa_model_input_unchanged():
    """Old path and new path produce bit-identical (B, 16, 3, H, W) tensors."""
    B = 3
    per_sample_old = [_old_preprocess(_fake_proc_out(seed=i)) for i in range(B)]
    per_sample_new = [_new_preprocess(_fake_proc_out(seed=i)) for i in range(B)]

    # Shapes after per-sample preprocessing:
    # old: (16, 3, 256, 256) — already repeated
    # new: (1, 3, 256, 256) — single frame
    assert per_sample_old[0].shape == (16, 3, 256, 256)
    assert per_sample_new[0].shape == (1, 3, 256, 256)

    batch_old = _stack_batch(per_sample_old)             # (B, 16, 3, 256, 256)
    batch_new = _stack_batch(per_sample_new)             # (B,  1, 3, 256, 256)
    assert batch_old.shape == (B, 16, 3, 256, 256)
    assert batch_new.shape == (B,  1, 3, 256, 256)

    # Old path is done — that's what the model sees on main.
    # New path: adapter's helper expands T=1 -> T=16 before the forward.
    adapter = _VjepaStub()
    batch_new_expanded = adapter._maybe_expand_video_frames(batch_new)
    assert batch_new_expanded.shape == (B, 16, 3, 256, 256)

    # Byte-identical tensors reach the model.
    assert torch.equal(batch_old, batch_new_expanded)


def test_expand_is_noop_for_non_vjepa():
    """The helper must not touch inputs for any other alias."""
    x = torch.randn(2, 3, 224, 224)
    for alias in ("vit", "dino", "dinov3", "clip", "convnext", "ijepa", "vit-mae"):
        stub = _VjepaStub()
        stub.alias = alias
        y = stub._maybe_expand_video_frames(x)
        assert y is x, f"alias {alias!r} unexpectedly mutated inputs"


def test_expand_leaves_already_expanded_tensor_alone():
    """If a future processor ever returns T>=16 natively, don't re-expand."""
    stub = _VjepaStub()
    x = torch.randn(2, 16, 3, 256, 256)
    y = stub._maybe_expand_video_frames(x)
    assert y.shape == x.shape
    assert torch.equal(x, y)
