"""Map the full module tree of any PyTorch model to a machine-readable JSON.

Every nn.Module in the tree is a valid hook point. This module:
1. Walks the full named_modules() graph
2. Probes each module with a dummy forward to get output shapes
3. Dumps a JSON file describing every extractable point

Usage:
    from pu.arch_map import map_architecture
    arch = map_architecture(model, dummy_input)
    # arch is a list of dicts, each with:
    #   name, class, output_shape, num_params, depth, is_leaf
"""

import json
from pathlib import Path

import torch
import torch.nn as nn


def map_architecture(model, dummy_input, device="cuda"):
    """Walk the full module tree and probe output shapes.

    Args:
        model: Any nn.Module (already on device, in eval mode).
        dummy_input: A tensor that can be passed to model(dummy_input).
                     For CLIP, pass pixel_values; for VLMs, pass a dict.
        device: Device string.

    Returns:
        List of dicts, one per named module (excluding root).
    """
    shapes = {}
    hooks = []

    def _make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                t = output[0]
            elif isinstance(output, dict):
                # Some modules return dicts (e.g., BaseModelOutput)
                t = next(iter(output.values())) if output else None
            else:
                t = output
            if isinstance(t, torch.Tensor):
                shapes[name] = list(t.shape)
            else:
                shapes[name] = None
        return hook

    # Register hooks on every module
    for name, mod in model.named_modules():
        if name:
            h = mod.register_forward_hook(_make_hook(name))
            hooks.append(h)

    # Forward pass to capture all shapes
    with torch.no_grad():
        try:
            if isinstance(dummy_input, dict):
                model(**dummy_input)
            else:
                model(dummy_input)
        except Exception as e:
            print(f"Warning: forward pass raised {e.__class__.__name__}: {e}")

    # Remove all hooks
    for h in hooks:
        h.remove()

    # Build architecture map
    arch = []
    for name, mod in model.named_modules():
        if not name:
            continue
        # Count depth by dots
        depth = name.count(".") + 1
        # Is leaf = has no children
        is_leaf = len(list(mod.children())) == 0
        # Parameter count (non-recursive to avoid double counting)
        num_params = sum(p.numel() for p in mod.parameters(recurse=False))

        entry = {
            "name": name,
            "class": mod.__class__.__name__,
            "output_shape": shapes.get(name),
            "num_params": num_params,
            "depth": depth,
            "is_leaf": is_leaf,
        }
        arch.append(entry)

    return arch


def map_all_models(output_dir="data/architectures", batch_size=2, image_size=224):
    """Map architectures for all registered models and save as JSON files.

    Loads each model, runs a dummy forward, and saves the full module tree.
    """
    from pu.models import get_adapter
    from pu.experiments_layerwise import MODEL_MAP

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dummy_img = torch.randn(batch_size, 3, image_size, image_size)

    for alias, (sizes, model_names) in MODEL_MAP.items():
        # Just map the first (smallest) size
        size, model_name = sizes[0], model_names[0]
        out_path = output_dir / f"{alias}_{size}.json"

        if out_path.exists():
            print(f"[skip] {out_path} already exists")
            continue

        print(f"\n[{alias} {size}] Loading {model_name}...")
        try:
            adapter_cls = get_adapter(alias)
            adapter = adapter_cls(model_name, size, alias=alias)
            adapter.load()
        except Exception as e:
            print(f"  [error] Could not load: {e}")
            continue

        model = adapter.model
        device = next(model.parameters()).device

        # Determine the right input for this model type
        if alias in ("clip",):
            dummy = dummy_img.to(device)
            # CLIP needs pixel_values kwarg for full model, but we map vision_model
            model_to_map = model.vision_model
            dummy_for_map = dummy
        else:
            model_to_map = model
            dummy_for_map = dummy_img.to(device)

        print(f"  Mapping {sum(1 for _ in model_to_map.named_modules()) - 1} modules...")
        arch = map_architecture(model_to_map, dummy_for_map, device=str(device))

        with open(out_path, "w") as f:
            json.dump({
                "model_alias": alias,
                "model_size": size,
                "model_name": model_name,
                "num_modules": len(arch),
                "num_leaf_modules": sum(1 for a in arch if a["is_leaf"]),
                "total_params": sum(a["num_params"] for a in arch),
                "modules": arch,
            }, f, indent=2)

        print(f"  Saved to {out_path} ({len(arch)} modules)")

        # Cleanup
        del adapter, model, model_to_map
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    map_all_models()
