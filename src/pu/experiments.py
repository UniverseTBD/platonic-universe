import json
import os
import numpy as np
import polars as pl
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import tempfile


from pu.models import get_adapter
from pu.pu_datasets import get_dataset_adapter
from pu.metrics import mknn, compare, compute_cka_mmap
from pu.utils import write_bin, plot_sample_galaxies

def run_experiment(model_alias, mode, output_dataset=None, batch_size=128, num_workers=0, knn_k=10, resize=True, resize_mode="match", all_metrics=False, max_samples=None, plot_samples=False):
    """Runs the embedding generation experiment based on the provided arguments.

    Args:
        model_alias: Model to run (e.g., 'vit', 'dino')
        mode: Dataset mode (e.g., 'jwst', 'legacysurvey')
        output_dataset: Optional HuggingFace dataset to upload to
        batch_size: Batch size for processing
        num_workers: Number of data loader workers
        knn_k: K value for MKNN calculation
        resize: If True, enable galaxy resizing
        resize_mode: 'match' to align to compared survey framing, 'fill' for adaptive per-galaxy cropping
        all_metrics: If True, compute all available metrics instead of just MKNN and CKA
        max_samples: If set, limit the dataset to this many samples (e.g. 1000 for a quick test run)
    """

    comp_mode = mode
    is_spectral_model = model_alias == "specformer"
    if is_spectral_model:
        # Spectral-only models process spectra directly; no HSC image pairing
        modes = [comp_mode]
    else:
        modes = ["hsc", comp_mode]
    hf_ds = f"Smith42/{comp_mode}_hsc_crossmatched"
    upload_ds = output_dataset

    def filterfun(idx):
        if "jwst" != comp_mode:
            return True
        else:
            im = idx["jwst_image"]["flux"][3]
            v0, v1 = np.nanpercentile(im, 5), np.nanpercentile(im, 99)
            if v0 - v1 == 0:
                return False
            else:
                return True

    model_map = {
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
        "dinov3":(
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
            [f"Smith42/astroPT_v2.0" for _ in range(3)],
        ),
        "sam2": (
            ["tiny", "small", "base-plus", "large"],
            [
                "facebook/sam2.1-hiera-tiny",
                "facebook/sam2.1-hiera-small",
                "facebook/sam2.1-hiera-base-plus",
                "facebook/sam2.1-hiera-large",
            ] ),
        "vit-mae": (
            ["base", "large", "huge"],
            [f"facebook/vit-mae-{s}" for s in ["base", "large", "huge"]],
        ),
        "hiera": (
            ["tiny", "small", "base-plus", "large"],
            [f"facebook/hiera-{s}-224-hf" for s in ["tiny", "small", "base-plus", "large"]],
        ),
        "specformer": (
            ["43M"],
            ["polymathic-ai/specformer"],
        ),
    }

    try:
        sizes, model_names = model_map[model_alias]
    except KeyError:
        raise NotImplementedError(f"Model '{model_alias}' not implemented.")

    if plot_samples:
        plot_sample_galaxies(hf_ds, modes, comp_mode, resize=resize, resize_mode=resize_mode)

    adapter_cls = get_adapter(model_alias)
    for size, model_name in zip(sizes, model_names):
        size_df = pl.DataFrame()
        adapter = adapter_cls(model_name, size, alias=model_alias)
        adapter.load()
        processor = adapter.get_preprocessor(modes, resize=resize, resize_mode=resize_mode)

        # Use dataset adapter to prepare the dataset (centralises dataset-specific logic)
        dataset_adapter_cls = get_dataset_adapter(comp_mode)
        dataset_adapter = dataset_adapter_cls(hf_ds, comp_mode)
        dataset_adapter.load()
        ds = dataset_adapter.prepare(processor, modes, filterfun)

        if max_samples is not None:
            ds = ds.take(max_samples)

        dl = iter(DataLoader(ds, batch_size=batch_size, num_workers=num_workers))

        zs = {mode: [] for mode in modes}
        with torch.no_grad():
            for B in tqdm(dl):
                for mode in modes:
                    if mode == "sdss":
                        zs[mode].append(torch.tensor(np.array(B["embedding"])).T)
                    elif mode == "desi":
                        zs[mode].append(torch.tensor(np.array(B["embeddings"])).T)
                    else:
                        # Delegate embedding to the adapter implementation
                        outputs = adapter.embed_for_mode(B, mode)
                        zs[mode].append(outputs)


        zs = {mode: torch.cat(embs) for mode, embs in zs.items()}
        Z1 = zs[modes[0]].cpu().numpy()
        Z2 = zs[modes[1]].cpu().numpy()

        temp1 = tempfile.NamedTemporaryFile(delete=False)
        temp2 = tempfile.NamedTemporaryFile(delete=False)
        temp1.close(); temp2.close()

        # build kernels
        k1 = Z1 @ Z1.T
        k2 = Z2 @ Z2.T

        write_bin(k1, str(temp1.name))
        write_bin(k2, str(temp2.name))

        # use kernel dimensions (square)
        cka_mmap_score = compute_cka_mmap(str(temp1.name), str(temp2.name), k1.shape[0], k1.shape[1])

        if all_metrics:
            # Use the compare() function to compute all metrics
            print(f"\n[{model_alias} {size}] Computing all metrics...")
            metrics_results = compare(
                Z1, Z2,
                metrics=["all"],
                mknn__k=knn_k,
                jaccard__k=knn_k,
            )

            # Add memory-mapped CKA to results
            metrics_results["cka_mmap"] = cka_mmap_score

            # Print all metrics
            print(f"\n{'='*60}")
            print(f"METRICS for {model_alias} {size}")
            print(f"{'='*60}")
            for metric_name, value in metrics_results.items():
                if value is not None:
                    print(f"  {metric_name:<25}: {value:.8f}")
                else:
                    print(f"  {metric_name:<25}: FAILED")
            print(f"{'='*60}\n")

            # Save detailed results
            os.makedirs("data", exist_ok=True)
            with open(f"data/{comp_mode}_{model_alias}_{size}_all_metrics.json", "w") as f:
                json.dump({
                    "model": model_alias,
                    "size": size,
                    "mode": comp_mode,
                    "n_samples": len(Z1),
                    "metrics": {k: float(v) if v is not None else None for k, v in metrics_results.items()}
                }, f, indent=2)

            mknn_score = metrics_results.get("mknn", 0.0)
            cka_score = cka_mmap_score
        else:
            # Original behavior: just MKNN and CKA
            mknn_score = mknn(Z1, Z2, knn_k)
            cka_score = cka_mmap_score
            print(f"\ncka {model_alias}, {size}: {cka_score:.8f}")
            print(f"\nmknn {model_alias}, {size}: {mknn_score:.8f}")

            # Create the directory if it doesn't exist
            os.makedirs("data", exist_ok=True)  
            # Creating the file to store mknn results
            with open(f"data/{comp_mode}_{model_alias}_scores.txt", "a") as fi:
                fi.write(f"{model_alias} {size},mknn : {mknn_score:.8f}, cka : {cka_score:.8f}\n")

        size_df = size_df.with_columns(
            [
                pl.Series(
                    f"{model_alias}_{size.lstrip('0')}_{mode}".lower(),
                    embs.cpu().numpy(),
                )
                for mode, embs in zs.items()
            ]
        )

        size_df.write_parquet(f"data/{comp_mode}_{model_alias}_{size}.parquet")
