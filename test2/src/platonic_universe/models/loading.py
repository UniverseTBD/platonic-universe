import torch
import logging
import timm
from ._registry import MODEL_REGISTRY

# Delay transformers imports until needed to allow environment setup first
def _import_transformers():
    """Lazy import of transformers to allow environment setup first."""
    global AutoProcessor, AutoModel, ViTImageProcessor, ViTModel, AutoImageProcessor
    if 'AutoProcessor' not in globals():
        from transformers import AutoProcessor, AutoModel, ViTImageProcessor, ViTModel, AutoImageProcessor
    return AutoProcessor, AutoModel, ViTImageProcessor, ViTModel, AutoImageProcessor

class LoadedModel:
    """A container for a loaded model and its specific preprocessor/transforms."""
    def __init__(self, model, device, processor=None, transforms=None):
        self.model = model
        self.processor = processor      # For Hugging Face transformers
        self.transforms = transforms    # For timm models
        self.device = device
        # Use a consistent way to get the model name
        model_name = getattr(model, 'name_or_path', getattr(model, 'default_cfg', {}).get('url', ''))
        logging.info(f"Model '{model_name}' loaded on device '{device}'.")

def list_available_models():
    """Returns a dictionary of available model aliases and their descriptions."""
    return {alias: info["description"] for alias, info in MODEL_REGISTRY.items()}

def load_model_from_alias(alias: str) -> LoadedModel:
    """Loads a pre-defined model from either Hugging Face or timm."""
    if alias not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model alias '{alias}'. Available options are: {available}")

    model_info = MODEL_REGISTRY[alias]
    repo_id = model_info["repo_id"]
    source = model_info.get("source", "huggingface") # Default to huggingface
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Loading model '{alias}' ({repo_id}) from '{source}' on device '{device}'...")

    try:
        if source == "timm":
            # --- TIMM-SPECIFIC LOADING LOGIC ---
            model = timm.create_model(repo_id, pretrained=True, num_classes=0).to(device)
            model = model.eval()
            data_config = timm.data.resolve_model_data_config(model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            return LoadedModel(model=model, device=device, transforms=transforms)
            
        elif source == "huggingface":
            # --- HUGGING FACE TRANSFORMERS LOGIC ---
            AutoProcessor, AutoModel, ViTImageProcessor, ViTModel, AutoImageProcessor = _import_transformers()
            
            # Use specific classes for different model types
            if 'ijepa' in repo_id.lower():
                model = AutoModel.from_pretrained(repo_id).to(device)
                processor = AutoProcessor.from_pretrained(repo_id)
                return LoadedModel(model=model, device=device, processor=processor)
            elif 'dinov2' in repo_id.lower():
                # Use AutoImageProcessor and AutoModel for DINOv2 with registers
                model = AutoModel.from_pretrained(repo_id).to(device)
                processor = AutoImageProcessor.from_pretrained(repo_id)
                return LoadedModel(model=model, device=device, processor=processor)
            elif 'vit' in repo_id.lower():
                model = ViTModel.from_pretrained(repo_id).to(device)
                processor = ViTImageProcessor.from_pretrained(repo_id)
                return LoadedModel(model=model, device=device, processor=processor)
            elif 'convnext' in repo_id.lower():
                model = AutoModel.from_pretrained(repo_id).to(device)
                processor = AutoImageProcessor.from_pretrained(repo_id)
                return LoadedModel(model=model, device=device, processor=processor)
            else:
                model = AutoModel.from_pretrained(repo_id).to(device)
                processor = AutoProcessor.from_pretrained(repo_id)
                return LoadedModel(model=model, device=device, processor=processor)

    except Exception as e:
        logging.error(f"Failed to load model '{alias}' from '{repo_id}'. Error: {e}")
        raise