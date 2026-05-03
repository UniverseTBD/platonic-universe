"""
Layer-wise embedding extraction pipeline.

Extracts embeddings from ALL layers of each model, saves to parquet,
and optionally uploads to HuggingFace Hub.
"""

import os
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pu.models import get_adapter
from pu.pu_datasets import get_dataset_adapter

# Survey alias → (HF dataset id, list of band-modes that will be passed to
# the model + saved as per-band columns). Use this for survey-agnostic
# layerwise extraction; the CLI picks rows from here, the legacy
# `extract_all_layers` Smith42-only path is preserved below it for backward
# compatibility.
SURVEY_REGISTRY: dict[str, tuple[str, list[str]]] = {
    "jwst":         ("Smith42/jwst_hsc_crossmatched",         ["hsc", "jwst"]),
    "legacysurvey": ("Smith42/legacysurvey_hsc_crossmatched", ["hsc", "legacysurvey"]),
    "desi":         ("Smith42/desi_hsc_crossmatched",         ["hsc", "desi"]),
    "sdss":         ("Smith42/sdss_hsc_crossmatched",         ["hsc", "sdss"]),
    "cosmosweb":    ("Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2", ["hsc", "jwst"]),
}

# Substrings that, if found in a layer's column name, cause the column to
# be dropped before parquet write. Default kills paligemma/llava vocab
# projections (`lm_head_*`, d ≈ 256k) which OOM downstream metrics and
# overflow pyarrow int32 list offsets at large N.
DEFAULT_EXCLUDE_SUBSTRINGS: tuple[str, ...] = ("lm_head",)

# Shared model map — same as experiments.py
MODEL_MAP = {
    "vit": (
        ["base", "large", "huge"],
        [
            "google/vit-base-patch16-224-in21k",
            "google/vit-large-patch16-224-in21k",
            "google/vit-huge-patch14-224-in21k",
        ],
    ),
    "clip": (
        ["base", "large"],
        [
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-large-patch14",
        ],
    ),
    "dino": (
        ["small", "base", "large", "giant"],
        [f"facebook/dinov2-with-registers-{s}" for s in ["small", "base", "large", "giant"]],
    ),
    "dinov3": (
        ["vits16", "vits16plus", "vitb16", "vitl16", "vith16plus", "vit7b16"],
        [
            "facebook/dinov3-vits16-pretrain-lvd1689m",
            "facebook/dinov3-vits16plus-pretrain-lvd1689m",
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "facebook/dinov3-vith16plus-pretrain-lvd1689m",
            "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        ],
    ),
    "convnext": (
        ["nano", "tiny", "base", "large"],
        [f"facebook/convnextv2-{s}-22k-224" for s in ["nano", "tiny", "base", "large"]],
    ),
    "ijepa": (
        ["huge", "giant"],
        ["facebook/ijepa_vith14_22k", "facebook/ijepa_vitg16_22k"],
    ),
    "vjepa": (
        ["large", "huge", "giant"],
        [
            "facebook/vjepa2-vitl-fpc64-256",
            "facebook/vjepa2-vith-fpc64-256",
            "facebook/vjepa2-vitg-fpc64-256",
        ],
    ),
    "astropt": (
        ["015M", "095M", "850M"],
        ["Smith42/astroPT_v2.0" for _ in range(3)],
    ),
    "vit-mae": (
        ["base", "large", "huge"],
        [f"facebook/vit-mae-{s}" for s in ["base", "large", "huge"]],
    ),
    "paligemma": (
        ["3b", "10b", "28b"],
        [
            "google/paligemma2-3b-mix-224",
            "google/paligemma2-10b-mix-224",
            "google/paligemma2-28b-mix-224",
        ],
    ),
    "paligemma_3b": (["3b"], ["google/paligemma2-3b-mix-224"]),
    "paligemma_10b": (["10b"], ["google/paligemma2-10b-mix-224"]),
    "paligemma_28b": (["28b"], ["google/paligemma2-28b-mix-224"]),
    "llava_15": (
        ["7b", "13b"],
        ["llava-hf/llava-1.5-7b-hf", "llava-hf/llava-1.5-13b-hf"],
    ),
    "llava_ov": (["7b"], ["llava-hf/llava-onevision-qwen2-7b-ov-hf"]),
}


def _default_filterfun(survey: str):
    """Survey-aware row filter. Keeps the original Smith42-jwst sanity check
    (drops rows whose JWST F444W slice is uniformly zero) and is a no-op for
    every other survey including cosmosweb."""
    def keep(idx):
        if survey != "jwst":
            return True
        try:
            im = idx["jwst_image"]["flux"][3]
        except (KeyError, IndexError, TypeError):
            return True
        v0, v1 = np.nanpercentile(im, 5), np.nanpercentile(im, 99)
        return v0 - v1 != 0
    return keep


def extract_layerwise_one(
    family: str,
    size: str,
    hf_id: str,
    survey: str,
    output_path: str | os.PathLike,
    *,
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: int | None = None,
    resize: bool = True,
    resize_mode: str = "match",
    granularity: str = "blocks",
    seed: int = 42,
    exclude_substrings: Iterable[str] = DEFAULT_EXCLUDE_SUBSTRINGS,
) -> Path:
    """Extract per-layer embeddings for ONE (family, size, survey) tuple.

    Loads the survey's HF dataset, runs the adapter's hookable forward pass
    over both bands defined in `SURVEY_REGISTRY[survey]`, drops any column
    whose name contains a substring in ``exclude_substrings`` (default:
    ``"lm_head"``), and writes a single parquet to ``output_path``.

    This is the unit of work for cooperative SLURM/vast.ai submissions —
    one parquet, one claim, one upload. ``extract_all_layers`` (the legacy
    Smith42-only entry point below) is now a thin loop over this.
    """
    from pu.models.base import set_seed
    set_seed(seed)

    if family == "specformer":
        raise NotImplementedError("SpecFormer does not support layerwise extraction")

    try:
        dataset_id, modes = SURVEY_REGISTRY[survey]
    except KeyError as e:
        raise ValueError(
            f"Unknown survey {survey!r}; known: {sorted(SURVEY_REGISTRY)}"
        ) from e

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adapter_cls = get_adapter(family)
    adapter = adapter_cls(hf_id, size, alias=family)
    adapter.load()
    if not adapter.supports_layerwise():
        raise NotImplementedError(
            f"Adapter for {family} {size} does not support layerwise extraction"
        )

    n_layers = adapter.get_num_layers(granularity=granularity)
    print(f"[{family} {size}] {n_layers} extraction points ({granularity}) "
          f"on {survey} bands={modes}")

    processor = adapter.get_preprocessor(modes, resize=resize, resize_mode=resize_mode)
    dataset_cls = get_dataset_adapter(survey)

    def _build_ds(*, ignore_max_samples: bool = False):
        ds_adapter = dataset_cls(dataset_id, modes[-1])
        ds_adapter.load()
        d = ds_adapter.prepare(processor, modes, _default_filterfun(survey))
        if max_samples is not None and not ignore_max_samples:
            d = d.take(max_samples)
        return d

    def _run_pass(bs: int, max_batches: int | None = None,
                  ignore_max_samples: bool = False):
        """Run the forward pass at batch=bs.

        If `max_batches` is set, stops after that many batches and returns
        whatever was collected — used by the calibration probe. Raises
        CUDA OOM up to the caller.
        """
        dl_kwargs = dict(
            batch_size=bs,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = 2
            dl_kwargs["persistent_workers"] = True
        dl = DataLoader(_build_ds(ignore_max_samples=ignore_max_samples), **dl_kwargs)

        layer_embs: dict[str, dict[str, list[torch.Tensor]]] = {}
        precomputed: dict[str, list[torch.Tensor]] = {}
        desc = f"{family} {size} ({survey}) bs={bs}"
        if max_batches is not None:
            desc += f" probe@{max_batches}"
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dl, desc=desc)):
                if max_batches is not None and i >= max_batches:
                    break
                for m in modes:
                    if m == "desi":
                        emb = batch["embeddings"]
                        if not isinstance(emb, torch.Tensor):
                            emb = torch.as_tensor(np.asarray(emb))
                        precomputed.setdefault(m, []).append(emb)
                    elif m == "sdss":
                        emb = batch["embedding"]
                        if not isinstance(emb, torch.Tensor):
                            emb = torch.as_tensor(np.asarray(emb))
                        precomputed.setdefault(m, []).append(emb)
                    else:
                        batch_layers = adapter.embed_all_layers_for_mode(
                            batch, m, granularity=granularity)
                        if m not in layer_embs:
                            layer_embs[m] = {k: [] for k in batch_layers}
                        for k, emb in batch_layers.items():
                            layer_embs[m][k].append(emb.cpu())
        return layer_embs, precomputed

    # ---- Adaptive batching ----------------------------------------------------
    # Pre-flight calibration: probe with a single batch, double if the GPU has
    # headroom, halve on OOM. Settles in the [grow, shrink) VRAM band. Then
    # the full pass runs at the calibrated bs with the same OOM-halve safety
    # net (in case mid-run memory is steeper than the probe).
    GROW_BELOW   = 0.70   # if peak < 70% of VRAM, double
    SHRINK_ABOVE = 0.85   # if peak > 85%, halve (rarely hit; OOM hits first)
    MAX_BS       = int(os.environ.get("PU_LW_MAX_BS", "256"))

    def _calibrate(initial_bs: int) -> int:
        if not torch.cuda.is_available():
            return max(1, initial_bs)
        total_vram = torch.cuda.get_device_properties(0).total_memory
        bs = max(1, initial_bs)
        last_good = 0
        direction: str | None = None  # "up" / "down" once we commit
        while True:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                _run_pass(bs, max_batches=1, ignore_max_samples=True)
                peak = torch.cuda.max_memory_allocated()
                usage = peak / total_vram
                print(f"  [calib] bs={bs} peak={peak/1e9:.2f}GB / "
                      f"{total_vram/1e9:.1f}GB ({usage:.0%})")
                last_good = bs
                if (usage < GROW_BELOW
                        and direction != "down"
                        and bs * 2 <= MAX_BS):
                    bs *= 2
                    direction = "up"
                    continue
                if usage > SHRINK_ABOVE and bs > 1:
                    bs = max(1, bs // 2)
                    direction = "down"
                    continue
                return bs
            except torch.cuda.OutOfMemoryError as e:
                print(f"  [calib] bs={bs} OOM ({type(e).__name__})")
                torch.cuda.empty_cache()
                import gc; gc.collect()
                if bs <= 1:
                    if last_good == 0:
                        raise
                    return last_good
                bs = max(1, bs // 2)
                direction = "down"

    calibrated_bs = _calibrate(int(batch_size))
    if calibrated_bs != batch_size:
        print(f"[{family} {size}] auto-batch calibrated to bs={calibrated_bs} "
              f"(requested {batch_size})")

    # Real pass at the calibrated batch size, with the same OOM-halve safety
    # net in case the calibration probe under-counted (rare).
    bs = calibrated_bs
    layer_embs = precomputed = None
    while True:
        try:
            layer_embs, precomputed = _run_pass(bs)
            break
        except torch.cuda.OutOfMemoryError as e:
            if bs <= 1:
                raise
            new_bs = max(1, bs // 2)
            print(f"  [oom@bs={bs}] {type(e).__name__}; retrying at bs={new_bs}")
            torch.cuda.empty_cache()
            import gc; gc.collect()
            bs = new_bs

    # Stitch per-batch tensors into one (n_samples, d) array per (layer, mode).
    columns: dict[str, np.ndarray] = {}
    for m, layers_dict in layer_embs.items():
        for layer_key in layers_dict:
            col_name = f"{layer_key}_{m}"
            columns[col_name] = torch.cat(layers_dict[layer_key]).numpy()
    for m, tensors in precomputed.items():
        columns[f"{m}_embedding"] = torch.cat(tensors).numpy()

    # Drop hooks whose output didn't have a batch dim (e.g. DinoV3
    # rope_embeddings outputs (seq_len, dim) → rows = batches × seq_len).
    if columns:
        counts = Counter(v.shape[0] for v in columns.values())
        expected = counts.most_common(1)[0][0]
        for k in [k for k, v in columns.items() if v.shape[0] != expected]:
            print(f"  [drop-shape] {k}: {columns[k].shape[0]} rows "
                  f"(expected {expected})")
            del columns[k]

    # Drop columns whose name contains any excluded substring (default:
    # lm_head). Vocab projections at d ≈ 256k OOM downstream and overflow
    # pyarrow int32 list offsets.
    excludes = tuple(exclude_substrings)
    if excludes:
        for k in [k for k in columns if any(sub in k for sub in excludes)]:
            print(f"  [drop-name]  {k}: matches exclude {excludes}")
            del columns[k]

    n_samples = next(iter(columns.values())).shape[0] if columns else 0
    print(f"[{family} {size}] {n_samples} samples, {len(columns)} columns -> "
          f"{output_path}")

    df = pl.DataFrame({k: pl.Series(k, v) for k, v in columns.items()})
    df.write_parquet(output_path, compression="zstd")

    del adapter, layer_embs, precomputed, columns, df
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_path


def extract_all_layers(
    model_alias,
    mode,
    batch_size=64,
    num_workers=0,
    max_samples=None,
    resize=True,
    resize_mode="match",
    output_dir="data",
    hf_repo=None,
    hf_token=None,
    upload=True,
    delete_after_upload=False,
    granularity="blocks",
    seed=42,
):
    """Extract embeddings from all layers of a model across a dataset.

    For each model size, extracts per-layer embeddings for the HSC baseline
    and (for JWST/LegacySurvey) the comparison mode. For DESI/SDSS the
    comparison embeddings are pre-computed and stored alongside.

    Saves one parquet per (model, size, dataset) to output_dir.
    Optionally uploads to HuggingFace Hub.

    Args:
        granularity: Extraction granularity level:
            - "blocks": Top-level blocks only (~14 for ViT-base). Default, matches upstream PRH.
            - "residual": All non-leaf modules (~76 for ViT-base).
            - "leaves": Leaf modules only (~137 for ViT-base).
            - "all": Everything (~213 for ViT-base).
        seed: Random seed for reproducibility. Default 42.
    """
    from pu.models.base import set_seed
    set_seed(seed)

    comp_mode = mode
    is_spectral_model = model_alias == "specformer"
    if is_spectral_model:
        print("[skip] SpecFormer does not support layerwise extraction")
        return

    modes = ["hsc", comp_mode]
    hf_ds = f"Smith42/{comp_mode}_hsc_crossmatched"

    def filterfun(idx):
        if "jwst" != comp_mode:
            return True
        im = idx["jwst_image"]["flux"][3]
        v0, v1 = np.nanpercentile(im, 5), np.nanpercentile(im, 99)
        return v0 - v1 != 0

    try:
        sizes, model_names = MODEL_MAP[model_alias]
    except KeyError:
        raise NotImplementedError(f"Model '{model_alias}' not in MODEL_MAP.")

    adapter_cls = get_adapter(model_alias)

    for size, model_name in zip(sizes, model_names):
        print(f"\n{'='*60}")
        print(f"[{model_alias} {size}] Loading model...")
        adapter = adapter_cls(model_name, size, alias=model_alias)
        adapter.load()

        if not adapter.supports_layerwise():
            print(f"[skip] {model_alias} {size} does not support layerwise extraction")
            continue

        num_layers = adapter.get_num_layers(granularity=granularity)
        print(f"[{model_alias} {size}] {num_layers} extraction points ({granularity}) on {comp_mode}...")

        processor = adapter.get_preprocessor(modes, resize=resize, resize_mode=resize_mode)

        ds_alias = comp_mode
        dataset_cls = get_dataset_adapter(ds_alias)
        ds_adapter = dataset_cls(hf_ds, comp_mode)
        ds_adapter.load()
        ds = ds_adapter.prepare(processor, modes, filterfun)

        if max_samples is not None:
            ds = ds.take(max_samples)

        dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

        # Accumulate per-layer embeddings for each mode
        # For modes where the model runs (HSC, JWST, LegacySurvey):
        #   layer_embs[mode] = {layer_idx: [tensor, ...]}
        # For pre-computed modes (DESI, SDSS):
        #   precomputed[mode] = [tensor, ...]
        layer_embs = {}
        precomputed = {}

        with torch.no_grad():
            for batch in tqdm(dl, desc=f"{model_alias} {size}"):
                for m in modes:
                    if m == "desi":
                        emb = batch["embeddings"]
                        if not isinstance(emb, torch.Tensor):
                            emb = torch.as_tensor(np.asarray(emb))
                        precomputed.setdefault(m, []).append(emb)
                    elif m == "sdss":
                        emb = batch["embedding"]
                        if not isinstance(emb, torch.Tensor):
                            emb = torch.as_tensor(np.asarray(emb))
                        precomputed.setdefault(m, []).append(emb)
                    else:
                        batch_layers = adapter.embed_all_layers_for_mode(batch, m, granularity=granularity)
                        if m not in layer_embs:
                            layer_embs[m] = {k: [] for k in batch_layers}
                        for k, emb in batch_layers.items():
                            layer_embs[m][k].append(emb.cpu())

        # Build polars DataFrame — use first batch's key order (DFS module order)
        columns = {}

        for m, layers_dict in layer_embs.items():
            for layer_key in layers_dict:
                col_name = f"{layer_key}_{m}"
                cat = torch.cat(layers_dict[layer_key]).numpy()
                columns[col_name] = cat

        for m, tensors in precomputed.items():
            col_name = f"{m}_embedding"
            cat = torch.cat(tensors).numpy()
            columns[col_name] = cat

        # Drop hooks that returned shapes without a batch dim (e.g. DinoV3's
        # rope_embeddings outputs (seq_len, dim), so rows = batches × seq_len).
        # The majority row count is the correct one — anything else is garbage.
        if columns:
            from collections import Counter
            counts = Counter(v.shape[0] for v in columns.values())
            expected = counts.most_common(1)[0][0]
            dropped = [k for k, v in columns.items() if v.shape[0] != expected]
            for k in dropped:
                print(f"[drop] {k}: {columns[k].shape[0]} rows (expected {expected})")
                del columns[k]

        n_samples = next(iter(columns.values())).shape[0] if columns else 0
        print(f"[{model_alias} {size}] {n_samples} samples, {len(columns)} columns")

        df = pl.DataFrame({
            k: pl.Series(k, v) for k, v in columns.items()
        })

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{comp_mode}_{model_alias}_{size}_layerwise.parquet")
        df.write_parquet(out_path)
        print(f"[{model_alias} {size}] Saved to {out_path}")

        # Upload to HuggingFace Hub and optionally delete local file
        if upload and hf_repo:
            from pu.hub import push_parquet
            push_parquet(out_path, hf_repo, token=hf_token)
            if delete_after_upload:
                os.remove(out_path)
                print(f"[{model_alias} {size}] Deleted local file after upload")

        # Free memory before next size
        del layer_embs, precomputed, columns, df
        del adapter
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
