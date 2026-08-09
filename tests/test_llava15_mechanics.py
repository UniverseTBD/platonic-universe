"""
Builds a tiny ``LlavaForConditionalGeneration`` with
random weights so data-dependent forward-pass branches (image-token masking,
``pixel_values=None`` handling) actually execute. No GPU/ model weights needed.
Validates the following mechanics:

- hook registration reaches both the vision tower and language model blocks
  via the existing generic ``blocks`` granularity machinery
- image-token mask logic (``input_ids == image_token_index``) on edge cases
- ``pixel_values=None`` text-only forward pass
- image-only forward pass (minimal text prompt)
- fused forward pass, with masked pooling split into image vs. text tokens
"""
import pytest
import torch
from transformers import (
    AutoModelForImageTextToText,
    CLIPVisionConfig,
    LlamaConfig,
    LlavaConfig,
)

from pu.models.base import ModelAdapter

IMAGE_TOKEN_INDEX = 32000


class _StubAdapter(ModelAdapter):
    """Minimal concrete ModelAdapter so we can reuse hooking code
    (get_layer_names / _capture_module_outputs) without going
    through HFAdapter/VLMAdapter's HF-download-driven `load()`."""

    def load(self, compile_model: bool = False) -> None:
        pass

    def get_preprocessor(self, modes, resize=False, resize_mode="fill"):
        pass

    def embed_for_mode(self, batch, mode):
        pass


@pytest.fixture(scope="module")
def tiny_llava():
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


@pytest.fixture(scope="module")
def n_patches(tiny_llava):
    vc = tiny_llava.config.vision_config
    return (vc.image_size // vc.patch_size) ** 2  # 49 for 224/32


def _fused_inputs(n_patches, batch_size=2, n_text_tokens=4):
    """input_ids with an image-token block plus surrounding text tokens."""
    seq_len = n_patches + n_text_tokens
    input_ids = torch.randint(1, IMAGE_TOKEN_INDEX, (batch_size, seq_len))
    start = n_text_tokens // 2
    input_ids[:, start:start + n_patches] = IMAGE_TOKEN_INDEX
    attn_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    return input_ids, pixel_values, attn_mask


# ── hook registration ─────────────────────────────────────────────────────

def test_blocks_granularity_covers_vision_tower_and_language_model(tiny_llava):
    adapter = _StubAdapter.__new__(_StubAdapter)
    adapter.model = tiny_llava
    names = adapter.get_layer_names(granularity="blocks")

    vision_layers = [n for n in names if "vision_tower.encoder.layers" in n]
    lm_layers = [n for n in names if "language_model.layers" in n]

    assert len(vision_layers) == tiny_llava.config.vision_config.num_hidden_layers
    assert len(lm_layers) == tiny_llava.config.text_config.num_hidden_layers
    # Distinct ModuleLists, no merge/miss between the two subtrees.
    assert set(vision_layers).isdisjoint(lm_layers)


def test_capture_module_outputs_fires_on_fused_pass(tiny_llava, n_patches):
    adapter = _StubAdapter.__new__(_StubAdapter)
    adapter.model = tiny_llava
    input_ids, pixel_values, attn_mask = _fused_inputs(n_patches)

    def forward_fn():
        tiny_llava(
            input_ids=input_ids, pixel_values=pixel_values,
            attention_mask=attn_mask, return_dict=True,
        )

    results = adapter._capture_module_outputs(forward_fn, granularity="blocks")
    vision_keys = [k for k in results if "vision_tower.encoder.layers" in k]
    lm_keys = [k for k in results if "language_model.layers" in k]
    assert len(vision_keys) == 3
    assert len(lm_keys) == 3
    for t in results.values():
        assert t.dim() == 2
        assert t.shape[0] == input_ids.shape[0]


# ── image-token mask logic edge cases ─────────────────────────────────────

def test_mask_no_image_tokens_present():
    input_ids = torch.randint(1, IMAGE_TOKEN_INDEX, (2, 10))
    mask = input_ids == IMAGE_TOKEN_INDEX
    assert mask.sum().item() == 0
    # Masked-mean-pool must not divide by zero here (clamp_min(1) in the
    # pooling helper handles this — this test documents why it's needed).


def test_mask_image_tokens_at_sequence_start():
    input_ids = torch.randint(1, IMAGE_TOKEN_INDEX, (2, 10))
    input_ids[:, :5] = IMAGE_TOKEN_INDEX
    mask = input_ids == IMAGE_TOKEN_INDEX
    assert mask[:, :5].all()
    assert not mask[:, 5:].any()


def test_mask_image_tokens_at_sequence_end():
    input_ids = torch.randint(1, IMAGE_TOKEN_INDEX, (2, 10))
    input_ids[:, -5:] = IMAGE_TOKEN_INDEX
    mask = input_ids == IMAGE_TOKEN_INDEX
    assert mask[:, -5:].all()
    assert not mask[:, :-5].any()


def test_mask_split_across_batch_padding():
    # Row 0: image tokens then padding (attn_mask=0 past position 6).
    # Row 1: shorter image block, more padding.
    input_ids = torch.randint(1, IMAGE_TOKEN_INDEX, (2, 10))
    attn_mask = torch.ones(2, 10, dtype=torch.long)
    input_ids[0, 1:6] = IMAGE_TOKEN_INDEX
    attn_mask[0, 8:] = 0
    input_ids[1, 1:4] = IMAGE_TOKEN_INDEX
    attn_mask[1, 6:] = 0

    image_mask = (input_ids == IMAGE_TOKEN_INDEX) & attn_mask.bool()
    assert image_mask[0].sum().item() == 5
    assert image_mask[1].sum().item() == 3
    # Padding positions must never be counted even if they happen to equal
    # the image token id.
    input_ids[0, 9] = IMAGE_TOKEN_INDEX  # inside the padded region
    image_mask = (input_ids == IMAGE_TOKEN_INDEX) & attn_mask.bool()
    assert image_mask[0].sum().item() == 5


# ── pixel_values=None / text-only forward pass ────────────────────────────

def test_text_only_pass_tolerates_pixel_values_none(tiny_llava):
    """Resolves the Step 1 open question: LLaVA's forward code runs cleanly
    with pixel_values=None as long as no image tokens are in input_ids."""
    input_ids = torch.randint(1, IMAGE_TOKEN_INDEX, (2, 8))
    attn_mask = torch.ones(2, 8, dtype=torch.long)
    with torch.no_grad():
        out = tiny_llava(
            input_ids=input_ids, pixel_values=None,
            attention_mask=attn_mask, return_dict=True,
            output_hidden_states=True,
        )
    assert out.hidden_states[-1].shape == (2, 8, 32)


# ── image-only forward pass ────────────────────────────────────────────────

def test_image_only_pass_minimal_text_prompt(tiny_llava, n_patches):
    """Image-only pass: exactly n_patches image tokens plus a single leading
    text/bos token, matching what _prepare_vlm_inputs-style construction
    would produce for an "image-only" input variant."""
    batch_size = 2
    seq_len = n_patches + 1
    input_ids = torch.full((batch_size, seq_len), IMAGE_TOKEN_INDEX)
    input_ids[:, 0] = 1
    attn_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    pixel_values = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        out = tiny_llava(
            input_ids=input_ids, pixel_values=pixel_values,
            attention_mask=attn_mask, return_dict=True,
            output_hidden_states=True,
        )
    assert out.hidden_states[-1].shape == (batch_size, seq_len, 32)


# ── fused pass: masked pooling split by image vs. text positions ─────────

def test_fused_pass_masked_pooling_image_vs_text(tiny_llava, n_patches):
    input_ids, pixel_values, attn_mask = _fused_inputs(n_patches)
    with torch.no_grad():
        out = tiny_llava(
            input_ids=input_ids, pixel_values=pixel_values,
            attention_mask=attn_mask, return_dict=True,
            output_hidden_states=True,
        )
    hs = out.hidden_states[-1]  # (B, seq_len, D)

    image_mask = (input_ids == IMAGE_TOKEN_INDEX) & attn_mask.bool()
    text_mask = (~image_mask) & attn_mask.bool()
    assert image_mask.sum(dim=1).eq(n_patches).all()
    assert text_mask.sum(dim=1).gt(0).all()

    def masked_mean_pool(hs, mask):
        m = mask.float().unsqueeze(-1)
        return (hs * m).sum(1) / m.sum(1).clamp_min(1.0)

    image_pooled = masked_mean_pool(hs, image_mask)
    text_pooled = masked_mean_pool(hs, text_mask)
    unmasked_pooled = masked_mean_pool(hs, attn_mask.bool())

    assert image_pooled.shape == (2, 32)
    assert text_pooled.shape == (2, 32)
    assert unmasked_pooled.shape == (2, 32)
    # The three pooling variants must actually differ (sanity: not
    # accidentally pooling the same positions three times).
    assert not torch.allclose(image_pooled, text_pooled)
    assert not torch.allclose(image_pooled, unmasked_pooled)
