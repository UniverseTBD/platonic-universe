# Extract tier

Re-extract layer-wise embeddings from raw galaxy images using the model
adapters in `pu/models/`. The result is what `compute/` consumes.

## Cost

Approximate, per (model × survey) tuple, on one A100:

- small models (≤300M params, e.g. ViT-base, DINO-small): a few minutes each
- medium (1B): tens of minutes
- large (7B–13B VLMs): a few hours
- 28B (paligemma-28B): half a day to a day per tuple

Total for the full 36-model × 4-survey grid: GPU-days. Skip this tier and
read precomputed embeddings from the public dataset instead unless you
specifically need to re-extract.

## Usage

```bash
pip install -e ".[platonic-extract]"
export HF_TOKEN=<your-hf-token>      # required for any gated source dataset

python experiments/platonic/extract/extract_layerwise.py \
    --models  vit_base dino_giant convnext_large vit-mae_huge \
    --surveys jwst legacysurvey desi sdss \
    --out-dir ./embeddings_out \
    --upload-to <owner>/<dataset_repo>
```

`--skip-existing` lets a re-run resume after a crash.

## Notes

- The model and survey aliases must already be registered in
  `pu/models/__init__.py` and `pu/pu_datasets/__init__.py`. To list them:

  ```python
  from pu.models import list_adapters; print(list_adapters())
  from pu.pu_datasets import list_datasets; print(list_datasets())
  ```

- This script is a thin batch wrapper. Per-model preprocessing,
  tokenisation, hooking, and serialisation logic all live in `pu/`.
