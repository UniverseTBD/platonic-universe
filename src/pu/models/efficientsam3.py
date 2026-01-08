import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from typing import Any, Dict, Iterable, Optional, Tuple

from pu.models.base import ModelAdapter
from pu.models.registry import register_adapter
from pu.preprocess import PreprocessEfficientSAM3

from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.data_misc import FindStage


_MODEL_REPO = "Simon7108528/EfficientSAM3"
_TINYVIT_CHECKPOINTS = {
    "5m": "stage1_all_converted/efficient_sam3_tinyvit_s.pt",
    "11m": "stage1_all_converted/efficient_sam3_tinyvit_m.pt",
    "21m": "stage1_all_converted/efficient_sam3_tinyvit_l.pt",
}
_TINYVIT_TEXT_CHECKPOINTS = {
    "5m": "stage1_all_converted/efficient_sam3_tinyvit_5m_mobileclip_s1.pth",
    "11m": "stage1_all_converted/efficient_sam3_tinyvit_11m_mobileclip_s1.pth",
    "21m": "stage1_all_converted/efficient_sam3_tinyvit_21m_mobileclip_s1.pth",
}
_PROMPT_KIND_TEXT = "text"
_PROMPT_KIND_GEOM = "geom"
_TEXT_PROMPTS = {
    "galaxy": "galaxy",
    "spiral-galaxy": "spiral galaxy",
    "elliptical-galaxy": "elliptical galaxy",
    "galaxy-core": "galaxy core",
    "spiral-arms": "spiral arms",
    "star": "star",
}
_GEOM_PROMPTS = {
    "center-box": {"box": [0.5, 0.5, 0.6, 0.6], "label": True},
}


def _parse_prompt_spec(size: str) -> Tuple[str, Optional[str], Optional[str]]:
    parts = size.split("-")
    base_size = parts[0]
    if len(parts) < 2:
        return base_size, None, None
    prompt_kind = parts[1]
    prompt_slug = "-".join(parts[2:]) if len(parts) > 2 else None
    return base_size, prompt_kind, prompt_slug


class EfficientSAM3Adapter(ModelAdapter):
    """
    Adapter for EfficientSAM3 (TinyViT family) image encoders.
    Uses the SAM3 vision backbone and pools spatial features to embeddings.
    """

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        self.model = None

    def load(self) -> None:
        if self.size not in _TINYVIT_CHECKPOINTS:
            raise ValueError(f"Unknown EfficientSAM3 size '{self.size}'.")

        checkpoint_path = hf_hub_download(
            repo_id=_MODEL_REPO,
            filename=_TINYVIT_CHECKPOINTS[self.size],
        )

        self.model = build_efficientsam3_image_model(
            checkpoint_path=checkpoint_path,
            backbone_type="tinyvit",
            model_name=self.model_name,
            enable_inst_interactivity=False,
        ).to("cuda").eval()

    def get_preprocessor(self, modes: Iterable[str]):
        return PreprocessEfficientSAM3(modes, resize=False)

    def _preprocess(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"Expected 4D batch tensor, got shape {inputs.shape}")

        if inputs.shape[1] != 3 and inputs.shape[-1] == 3:
            inputs = inputs.permute(0, 3, 1, 2)

        inputs = inputs.float() / 255.0
        inputs = F.interpolate(
            inputs, size=(1008, 1008), mode="bilinear", align_corners=False
        )
        inputs = (inputs - 0.5) / 0.5
        return inputs

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = batch[f"{mode}"].to("cuda", non_blocking=True)
        inputs = self._preprocess(inputs)

        with torch.no_grad():
            backbone_out = self.model.backbone.forward_image(inputs)
            features = backbone_out["vision_features"]
            emb = features.mean(dim=(2, 3)).detach()

        return emb


class EfficientSAM3PromptAdapter(ModelAdapter):
    """
    Adapter for prompt-conditioned EfficientSAM3 embeddings.
    Uses SAM3 prompt heads to generate masks, then pools vision features with the mask.
    """

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        self.model = None
        self.base_size = None
        self.prompt_kind = None
        self.prompt_slug = None
        self.prompt_text = None
        self.prompt_box = None
        self.prompt_label = True

    def load(self) -> None:
        base_size, prompt_kind, prompt_slug = _parse_prompt_spec(self.size)
        if prompt_kind is None:
            raise ValueError(
                "Prompt adapter requires size format '<base>-text-<slug>' or '<base>-geom-<slug>'."
            )
        if base_size not in _TINYVIT_TEXT_CHECKPOINTS:
            raise ValueError(f"Unknown EfficientSAM3 size '{base_size}'.")

        self.base_size = base_size
        self.prompt_kind = prompt_kind
        self.prompt_slug = prompt_slug

        checkpoint_path = hf_hub_download(
            repo_id=_MODEL_REPO,
            filename=_TINYVIT_TEXT_CHECKPOINTS[base_size],
        )

        self.model = build_efficientsam3_image_model(
            checkpoint_path=checkpoint_path,
            backbone_type="tinyvit",
            model_name=self.model_name,
            text_encoder_type="MobileCLIP-S1",
            enable_inst_interactivity=False,
        ).to("cuda").eval()

        if prompt_kind == _PROMPT_KIND_TEXT:
            if prompt_slug not in _TEXT_PROMPTS:
                raise ValueError(f"Unknown text prompt slug '{prompt_slug}'.")
            self.prompt_text = _TEXT_PROMPTS[prompt_slug]
        elif prompt_kind == _PROMPT_KIND_GEOM:
            if prompt_slug not in _GEOM_PROMPTS:
                raise ValueError(f"Unknown geometric prompt slug '{prompt_slug}'.")
            prompt_spec = _GEOM_PROMPTS[prompt_slug]
            self.prompt_box = prompt_spec["box"]
            self.prompt_label = bool(prompt_spec.get("label", True))
        else:
            raise ValueError(f"Unknown prompt kind '{prompt_kind}'.")

    def get_preprocessor(self, modes: Iterable[str]):
        return PreprocessEfficientSAM3(modes, resize=False)

    def _preprocess(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"Expected 4D batch tensor, got shape {inputs.shape}")

        if inputs.shape[1] != 3 and inputs.shape[-1] == 3:
            inputs = inputs.permute(0, 3, 1, 2)

        inputs = inputs.float() / 255.0
        inputs = F.interpolate(
            inputs, size=(1008, 1008), mode="bilinear", align_corners=False
        )
        inputs = (inputs - 0.5) / 0.5
        return inputs

    def _build_find_stage(self, batch_size: int) -> FindStage:
        device = "cuda"
        img_ids = torch.arange(batch_size, device=device, dtype=torch.long)
        text_ids = torch.arange(batch_size, device=device, dtype=torch.long)
        return FindStage(
            img_ids=img_ids,
            text_ids=text_ids,
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    def _embed_prompted_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        backbone_out = self.model.backbone.forward_image(inputs)

        if self.prompt_kind == _PROMPT_KIND_TEXT:
            prompts = [self.prompt_text] * batch_size
        else:
            prompts = ["visual"] * batch_size
        text_outputs = self.model.backbone.forward_text(prompts, device="cuda")
        backbone_out.update(text_outputs)

        geometric_prompt = self.model._get_dummy_prompt(num_prompts=batch_size)
        if self.prompt_kind == _PROMPT_KIND_GEOM:
            box = torch.tensor(
                self.prompt_box, device="cuda", dtype=torch.float32
            ).view(1, 1, 4)
            box = box.repeat(1, batch_size, 1)
            label_val = 1 if self.prompt_label else 0
            labels = torch.full(
                (1, batch_size), label_val, device="cuda", dtype=torch.long
            )
            geometric_prompt.append_boxes(box, labels)

        find_input = self._build_find_stage(batch_size)
        outputs = self.model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=geometric_prompt,
        )

        pred_logits = outputs["pred_logits"]
        pred_masks = outputs["pred_masks"]
        scores = pred_logits.sigmoid()
        presence = outputs.get("presence_logit_dec")
        if presence is not None:
            scores = scores * presence.sigmoid().unsqueeze(1)
        scores = scores.squeeze(-1)
        best_idx = scores.argmax(dim=1)
        masks = pred_masks.sigmoid()
        batch_idx = torch.arange(batch_size, device=pred_masks.device)
        chosen_masks = masks[batch_idx, best_idx].unsqueeze(1)

        features = backbone_out["vision_features"]
        if features.ndim == 3:
            features = features.unsqueeze(0)
        chosen_masks = F.interpolate(
            chosen_masks, size=features.shape[-2:], mode="bilinear", align_corners=False
        )
        weight_sum = chosen_masks.sum(dim=(2, 3)).clamp_min(1e-6)
        pooled = (features * chosen_masks).sum(dim=(2, 3)) / weight_sum
        return pooled.detach()

    def _stack_collated_list(self, inputs: list) -> torch.Tensor:
        channels = []
        for channel in inputs:
            rows = [torch.stack(row, dim=0) for row in channel]
            channels.append(torch.stack(rows, dim=0))
        stacked = torch.stack(channels, dim=0)
        return stacked.permute(3, 0, 1, 2)

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = batch[f"{mode}"]
        if isinstance(inputs, list):
            if len(inputs) == 0:
                raise ValueError("Empty batch received for EfficientSAM3 prompt adapter.")
            if isinstance(inputs[0], torch.Tensor):
                inputs = torch.stack(inputs, dim=0)
            elif (
                isinstance(inputs[0], list)
                and inputs[0]
                and isinstance(inputs[0][0], list)
                and inputs[0][0]
                and isinstance(inputs[0][0][0], list)
            ):
                # Collate produced C x H x W x B layout; move batch dim first.
                inputs = torch.tensor(inputs).permute(3, 0, 1, 2)
            elif (
                isinstance(inputs[0], list)
                and inputs[0]
                and isinstance(inputs[0][0], list)
                and inputs[0][0]
                and isinstance(inputs[0][0][0], torch.Tensor)
            ):
                inputs = self._stack_collated_list(inputs)
            else:
                inputs = torch.stack([torch.as_tensor(item) for item in inputs], dim=0)
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)

        if inputs.shape[1] != 3 and inputs.shape[-1] == 3:
            inputs = inputs.permute(0, 3, 1, 2)

        inputs = inputs.to("cuda", non_blocking=True)
        inputs = self._preprocess(inputs)
        with torch.no_grad():
            embeddings = self._embed_prompted_batch(inputs)
        return embeddings


register_adapter("efficientsam3", EfficientSAM3Adapter)
register_adapter("efficientsam3-prompt", EfficientSAM3PromptAdapter)
