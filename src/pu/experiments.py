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
#from astroclip.models.specformer import SpecFormer
from pu.utils import write_bin

def run_experiment(model_alias, mode, output_dataset=None, batch_size=128, num_workers=0, knn_k=10, all_metrics=False):
    """Runs the embedding generation experiment based on the provided arguments.

    Args:
        model_alias: Model to run (e.g., 'vit', 'dino')
        mode: Dataset mode (e.g., 'jwst', 'legacysurvey')
        output_dataset: Optional HuggingFace dataset to upload to
        batch_size: Batch size for processing
        num_workers: Number of data loader workers
        knn_k: K value for MKNN calculation
        all_metrics: If True, compute all available metrics instead of just MKNN and CKA
    """
    """Runs the embedding generation experiment based on the provided arguments."""

    comp_mode = mode
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
        "dino": (
            ["small", "base", "large", "giant"],
            [f"facebook/dinov2-with-registers-{s}" for s in ["small", "base", "large", "giant"]],
        ),
        "dinov3":(
            [
                "vits16", "vits16plus", "vitb16", "vitl16", "vith16plus", "vit7b16",
                "convnext-base", "convnext-large", "convnext-small", "convnext-tiny",
                "vitl16-sat493m", "vit7b16-sat493m",
            ],
            [
                "facebook/dinov3-vits16-pretrain-lvd1689m",
                "facebook/dinov3-vits16plus-pretrain-lvd1689m",
                "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "facebook/dinov3-vith16plus-pretrain-lvd1689m",
                "facebook/dinov3-vit7b16-pretrain-lvd1689m",
                "facebook/dinov3-convnext-base-pretrain-lvd1689m",
                "facebook/dinov3-convnext-large-pretrain-lvd1689m",
                "facebook/dinov3-convnext-small-pretrain-lvd1689m",
                "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
                "facebook/dinov3-vitl16-pretrain-sat493m",
                "facebook/dinov3-vit7b16-pretrain-sat493m",
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
    }

    try:
        sizes, model_names = model_map[model_alias]
    except KeyError:
        raise NotImplementedError(f"Model '{model_alias}' not implemented.")

    df = pl.DataFrame()
    adapter_cls = get_adapter(model_alias)
    for size, model_name in zip(sizes, model_names):
        adapter = adapter_cls(model_name, size, alias=model_alias)
        adapter.load()
        processor = adapter.get_preprocessor(modes)

        # Use dataset adapter to prepare the dataset (centralises dataset-specific logic)
        dataset_adapter_cls = get_dataset_adapter(comp_mode)
        dataset_adapter = dataset_adapter_cls(hf_ds, comp_mode)
        dataset_adapter.load()
        ds = dataset_adapter.prepare(processor, modes, filterfun)


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

        df = df.with_columns(
            [
                pl.Series(
                    f"{model_alias}_{size.lstrip('0')}_{mode}".lower(),
                    embs.cpu().numpy(),
                )
                for mode, embs in zs.items()
            ]
        )

        df.write_parquet(f"data/{comp_mode}_{model_alias}_{size}.parquet")

        # if upload_ds is not None:
        #     Dataset.from_polars(df).push_to_hub(upload_ds)


# def get_specformer_embeddings(dataset_name="Smith42/desi_hsc_crossmatched", batch_size=128, num_workers=0):
#     """Generates embeddings using the SpecFormer model for the given dataset."""

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     def _process_galaxy_wrapper(idx):
#         spectra = np.array(idx["spectrum"]["flux"], dtype=np.float32)[..., np.newaxis]
#         return {
#             "spectra": spectra,
#         }

#     def load_specformer_model(checkpoint_path):
#         """Load SpecFormer model from checkpoint."""
#         checkpoint = torch.load(checkpoint_path, weights_only=False)
#         model = SpecFormer(**checkpoint["hyper_parameters"])
#         model.load_state_dict(checkpoint["state_dict"])
#         model.eval()
#         return model
#     # Load model
#     checkpoint_path = "specformer.ckpt"
#     model = load_specformer_model(checkpoint_path).to(device)
#     ds = (
#         load_dataset(dataset_name, split="train", streaming=True)
#         .select_columns(("spectrum"))
#         .map(_process_galaxy_wrapper)
#         .remove_columns(("spectrum"))
#     )

#     dl = iter(DataLoader(ds, batch_size=batch_size, num_workers=num_workers))

#     embedddings = []

#     with torch.no_grad():
#         for B in tqdm(dl):
#             S = B["spectra"].to(device)
#             output = model(S)
#             # Extract the embedding (not the reconstruction)
#             batch_embeddings = output["embedding"].detach().cpu().numpy()
#             embedddings.append(batch_embeddings[:, 1:, :].mean(axis=1))
#     embedddings = np.concatenate(embedddings, axis=0)

#     print(f"Output embeddings shape: {embedddings.shape}")
    
#     os.makedirs("data", exist_ok=True)
#     np.save(f"data/specformer_{dataset_name.split('/')[-1].replace('_', '-')}_embeddings.npy", embedddings)
