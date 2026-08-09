"""Step 2b: end-to-end validation of VLMAdapter's new pre/post-fusion
extraction keys against a tiny synthetic LLaVA-1.5 model + a stub processor
that mimics LlavaProcessor's "<image>" placeholder expansion contract.

No real weights, no GPU, no internet — same tiny-model strategy as
tests/test_llava15_mechanics.py, now exercised through the actual
VLMAdapter.embed_all_layers_for_mode code path instead of hand-rolled
mechanics.
"""
import pytest
import torch
import torch.nn as nn
from transformers import (
    AutoModelForImageTextToText,
    CLIPVisionConfig,
    LlamaConfig,
    LlavaConfig,
)

from pu.models.hf import VLMAdapter

IMAGE_TOKEN_INDEX = 32000


class _StubLlavaProcessor:
    """Mimics LlavaProcessor's __call__ contract closely enough to drive
    VLMAdapter's input-construction code paths:

    - text containing "<image>" + images given -> expands the placeholder
      to n_patches image-token ids (matches real processor behavior).
    - text only (no images) -> plain token ids, none equal to the image
      token id.
    """

    def __init__(self, n_patches: int):
        self.n_patches = n_patches
        self._next_id = 1

    def _tokenize_plain(self, s: str, n: int) -> list:
        # Deterministic, non-image-token ids.
        return [((hash((s, i)) % (IMAGE_TOKEN_INDEX - 1)) + 1) for i in range(n)]

    def __call__(self, text, images=None, return_tensors=None, padding=None):
        rows = []
        for t in text:
            if images is not None and "<image>" in t:
                pre, post = t.split("<image>", 1)
                pre_ids = self._tokenize_plain(pre, len(pre.split())) if pre.strip() else []
                post_ids = self._tokenize_plain(post, len(post.split())) if post.strip() else []
                row = pre_ids + [IMAGE_TOKEN_INDEX] * self.n_patches + post_ids
            else:
                n_tokens = max(1, len(t.split()))
                row = self._tokenize_plain(t, n_tokens)
            rows.append(row)

        max_len = max(len(r) for r in rows)
        input_ids = torch.zeros(len(rows), max_len, dtype=torch.long)
        attn_mask = torch.zeros(len(rows), max_len, dtype=torch.long)
        for i, r in enumerate(rows):
            input_ids[i, :len(r)] = torch.tensor(r, dtype=torch.long)
            attn_mask[i, :len(r)] = 1

        out = {"input_ids": input_ids, "attention_mask": attn_mask}
        if images is not None:
            out["pixel_values"] = torch.stack([
                torch.from_numpy(__import__("numpy").array(im)).permute(2, 0, 1).float()
                for im in images
            ])
        return out


@pytest.fixture(scope="module")
def tiny_llava_model():
    vision_config = CLIPVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, image_size=224, patch_size=32,
    )
    text_config = LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, vocab_size=32001,
    )
    config = LlavaConfig(
        vision_config=vision_config, text_config=text_config,
        image_token_index=IMAGE_TOKEN_INDEX, vision_feature_layer=-2,
        vision_feature_select_strategy="default",
    )
    model = AutoModelForImageTextToText.from_config(config)
    model.eval()
    return model


@pytest.fixture
def adapter(tiny_llava_model):
    n_patches = (224 // 32) ** 2  # 49
    a = VLMAdapter.__new__(VLMAdapter)
    a.alias = "llava_15"
    a.model = tiny_llava_model
    a.processor = _StubLlavaProcessor(n_patches=n_patches)
    return a


@pytest.fixture
def batch():
    B = 2
    return {"hsc": torch.rand(B, 3, 224, 224)}


def test_get_layer_names_includes_all_new_keys(adapter):
    names = adapter.get_layer_names(granularity="blocks")
    for key in (
        "hidden_states_last", "fused_image_masked", "fused_text_masked",
        "image_only_unimodal", "text_only_unimodal",
    ):
        assert key in names
    assert any("vision_tower.encoder.layers" in n for n in names)
    assert any("language_model.layers" in n for n in names)


def test_embed_all_layers_for_mode_populates_all_keys(adapter, batch):
    results = adapter.embed_all_layers_for_mode(batch, "hsc", granularity="blocks")

    expected_keys = {
        "hidden_states_last", "fused_image_masked", "fused_text_masked",
        "image_only_unimodal", "text_only_unimodal",
    }
    assert expected_keys.issubset(results.keys())

    B = batch["hsc"].shape[0]
    for key, t in results.items():
        assert t.dim() == 2, f"{key} should be pooled to (B, D), got {t.shape}"
        assert t.shape[0] == B, f"{key} row count mismatch"

    vision_cols = [k for k in results if "vision_tower.encoder.layers" in k]
    lm_cols = [k for k in results if "language_model.layers" in k]
    assert len(vision_cols) == 3
    assert len(lm_cols) == 3


def test_fused_masked_variants_are_distinct(adapter, batch):
    results = adapter.embed_all_layers_for_mode(batch, "hsc", granularity="blocks")
    assert not torch.allclose(results["fused_image_masked"], results["fused_text_masked"])
    assert not torch.allclose(results["fused_image_masked"], results["hidden_states_last"])


def test_unimodal_baselines_differ_from_fused(adapter, batch):
    results = adapter.embed_all_layers_for_mode(batch, "hsc", granularity="blocks")
    assert not torch.allclose(results["image_only_unimodal"], results["fused_image_masked"])
    assert not torch.allclose(results["text_only_unimodal"], results["fused_text_masked"])


def test_no_nans_or_degenerate_outputs(adapter, batch):
    results = adapter.embed_all_layers_for_mode(batch, "hsc", granularity="blocks")
    for key, t in results.items():
        assert torch.isfinite(t).all(), f"{key} has non-finite values"
        assert not torch.allclose(t, torch.zeros_like(t)), f"{key} is all-zero"
