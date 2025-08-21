MODEL_REGISTRY = {
    "ijepa-base": {
        "repo_id": "facebook/ijepa_vith14_1k",
        "source": "huggingface", # Be explicit
        "description": "I-JEPA base model from Hugging Face.",
    },
    "ijepa-vith14-22k": {
        "repo_id": "facebook/ijepa_vith14_22k",
        "source": "huggingface",
        "description": "I-JEPA ViT-H/14 model trained on ImageNet-22k.",
    },
    "ijepa-vitg16-22k": {
        "repo_id": "facebook/ijepa_vitg16_22k",
        "source": "huggingface",
        "description": "I-JEPA ViT-G/16 model trained on ImageNet-22k.",
    },
     "dinov2-small": {
        "repo_id": "vit_base_patch14_dinov2.lvd142m",
        "source": "timm",
        "description": "DINOv2 small model from the timm library.",
    },
     "dinov2-base": {
        "repo_id": "vit_base_patch14_dinov2.lvd142m",
        "source": "timm",
        "description": "DINOv2 base model from the timm library.",
    },
     "dinov2-large": {
        "repo_id": "vit_base_patch14_dinov2.lvd142m",
        "source": "timm",
        "description": "DINOv2 large model from the timm library.",
    },
    "dinov2-giant": {
        "repo_id": "vit_base_patch14_dinov2.lvd142m",
        "source": "timm",
        "description": "DINOv2 huge model from the timm library.",
    },
    "vit-base": {
        "repo_id": "google/vit-base-patch16-224-in21k",
        "source": "huggingface",
        "description": "Vision Transformer base model trained on ImageNet-21k.",
    },
    "vit-large": {
        "repo_id": "google/vit-large-patch16-224-in21k",
        "source": "huggingface", 
        "description": "Vision Transformer large model trained on ImageNet-21k.",
    },
    "vit-huge": {
        "repo_id": "google/vit-huge-patch14-224-in21k",
        "source": "huggingface",
        "description": "Vision Transformer huge model trained on ImageNet-21k.",
    },
}