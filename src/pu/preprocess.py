import json
import os
from functools import partial

import numpy as np
import torch
from astropt.local_datasets import GalaxyImageDataset
from torchvision import transforms

from pu.zoom import resize_galaxy_to_fit

# Default percentiles file path (relative to package root or cwd)
_PERCENTILES_PATH = os.path.join("data", "percentiles.json")
_percentiles_cache = None


def _load_percentiles():
    """Load percentiles from JSON file, with caching."""
    global _percentiles_cache
    if _percentiles_cache is not None:
        return _percentiles_cache

    path = os.environ.get("PU_PERCENTILES_PATH", _PERCENTILES_PATH)
    with open(path) as f:
        _percentiles_cache = json.load(f)
        return _percentiles_cache


class PreprocessHF:
    """Preprocessor that converts galaxy images to the format expected by Dino and ViT models"""

    def __init__(self, modes, autoproc, resize=True, resize_mode="match", alias=None):
        self.modes = modes
        self.autoproc = autoproc
        self.alias = alias
        self.f2p = partial(
            flux_to_pil, resize=resize, resize_mode=resize_mode
        )

    def __call__(self, idx):
        result = {}
        for mode in self.modes:
            if (mode == "desi") or (mode == "sdss"):
                continue
            else:
                im = self.f2p(idx[f"{mode}_image"], mode, self.modes)
                if self.alias == "clip":
                    proc_out = self.autoproc(images=im, return_tensors="pt")
                else:
                    proc_out = self.autoproc(im, return_tensors="pt")
                if "pixel_values" in proc_out:
                    # first try for image models
                    result[f"{mode}"] = proc_out["pixel_values"].squeeze()
                elif "pixel_values_videos" in proc_out:
                    # then try for video models
                    result[f"{mode}"] = self.autoproc(im, return_tensors="pt")[
                        "pixel_values_videos"
                    ].repeat(1, 16, 1, 1, 1).squeeze()
                else:
                    # finally bail if there is an issue
                    raise KeyError("autoproc does not have 'pixel_values' or 'pixel_values_videos' in its dict")

        return result
    

class PreprocessSAM2:
    """Preprocessor that converts galaxy images to the format expected by SAM2 models"""

    def __init__(
        self,
        modes,
        sam2_transforms,
        resize=True,
        resize_mode="match",
    ):
        self.modes = modes
        self.sam2_transforms = sam2_transforms
        self.f2p = partial(
            flux_to_pil, resize=resize, resize_mode=resize_mode
        )

    def __call__(self, idx):
        result = {}
        for mode in self.modes:
            if (mode == "desi") or (mode == "sdss"):
                continue
            else:
                # Convert flux to PIL-like array
                im = self.f2p(idx[f"{mode}_image"], mode, self.modes)
                # Apply SAM2 transforms
                transformed = self.sam2_transforms(im)
                result[f"{mode}"] = transformed

        return result


class PreprocessAstropt:
    """Preprocessor that converts galaxy images to the format expected by AstroPT models"""

    @staticmethod
    def normalise_for_astropt(x):
        std, mean = torch.std_mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + 1e-8)

    @classmethod
    def data_transforms(cls):
        return transforms.Compose([transforms.Lambda(cls.normalise_for_astropt)])

    def __init__(
        self,
        modality_registry,
        modes,
        resize=True,
        resize_mode="match",
    ):
        self.galproc = GalaxyImageDataset(
            None,
            spiral=True,
            transform={"images": self.data_transforms()},
            modality_registry=modality_registry,
        )
        self.modes = modes
        self.f2p = partial(
            flux_to_pil, resize=resize, resize_mode=resize_mode
        )

    def __call__(self, idx):
        result = {}
        for mode in self.modes:
            if (mode == "desi") or (mode == "sdss"):
                continue
            else:
                im = self.f2p(idx[f"{mode}_image"], mode, self.modes).swapaxes(0, 2)
                im = self.galproc.process_galaxy(
                    torch.from_numpy(im).to(torch.float)
                ).to(torch.float)
                result[f"{mode}_images"] = im
                result[f"{mode}_positions"] = torch.arange(0, len(im), dtype=torch.long)

        return result


def _get_norm_consts(mode, band_names):
    """Load norm constants from percentiles.json."""
    percentiles = _load_percentiles()
    if percentiles is None:
        raise FileNotFoundError(
            f"Percentiles file not found. Run 'pu percentiles' first to generate it."
        )
    mode_data = percentiles[mode]
    return {
        band: (mode_data[band]["p1"], mode_data[band]["p99"])
        for band in band_names
    }


def flux_to_pil(blob, mode, modes, resize=True, percentile_norm=True, resize_mode="match"):
    """
    Convert raw fluxes to PIL imagery
    """

    def _norm(chan, percentiles=None):
        if percentiles is not None:
            # if percentiles are present norm by them
            v0, v1 = percentiles
            chan = ((chan - v0) / (v1 - v0)).clip(0, 1)
        else:
            # else assume we norm per image
            scale = np.percentile(chan, 99) - np.percentile(chan, 1)
            chan = np.arcsinh((chan - np.percentile(chan, 1)) / scale)
            chan = (chan - chan.min()) / (chan.max() - chan.min())
        return chan

    arr = np.asarray(blob["flux"], np.float32)
    if mode == "hsc": #160x160 pixels in MMU dataset
        if arr.ndim == 3:
            arr = np.stack([arr[0], arr[1], arr[3]], axis=-1)  # grz
        elif arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        else:
            raise ValueError(f"Array shape {arr.shape} for {mode} not recognised")

        if resize:
            if resize_mode == "fill":
                arr = resize_galaxy_to_fit(
                    arr, target_size=96
                )
            else: # match
                arr = resize_galaxy_to_fit(
                    arr, force_extent=(68, 92, 68, 92), target_size=96
                )

        if percentile_norm:
            norm_consts = _get_norm_consts("hsc", ("g", "r", "z"))
            arr = np.stack(
                [
                    _norm(arr[..., ii], norm_consts[band])
                    for ii, band in enumerate(("g", "r", "z"))
                ],
                axis=-1,
            )

    if mode == "jwst":  # 0.04 pixel per arcsec, 96x96 pixels in MMU dataset
        if arr.ndim == 3:
            arr = np.stack([arr[0], arr[4], arr[6]], axis=-1)
        elif arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        else:
            raise ValueError(f"Array shape {arr.shape} for {mode} not recognised")

        if percentile_norm:
            norm_consts = _get_norm_consts("jwst", ("f090w", "f277w", "f444w"))
            arr = np.stack(
                [
                    _norm(arr[..., ii], norm_consts[band])
                    for ii, band in enumerate(("f090w", "f277w", "f444w"))
                ],
                axis=-1,
            )

    if mode == "legacysurvey":
        if arr.ndim == 3:
            arr = np.stack([arr[0], arr[1], arr[3]], axis=-1)  # grz
        elif arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        else:
            raise ValueError(f"Array shape {arr.shape} for {mode} not recognised")

        if resize:
            # we always resize legacy to match hsc for our use-case
            if resize_mode == "fill":
                arr = resize_galaxy_to_fit(
                    arr, target_size=96
                )
            else: # match
                arr = resize_galaxy_to_fit(
                    arr, force_extent=(72, 88, 72, 88), target_size=96
                )

        if percentile_norm:
            norm_consts = _get_norm_consts("legacysurvey", ("g", "r", "z"))
            arr = np.stack(
                [
                    _norm(arr[..., ii], norm_consts[band])
                    for ii, band in enumerate(("g", "r", "z"))
                ],
                axis=-1,
            )

    if not percentile_norm:
        arr = _norm(arr)
    arr = (arr[..., ::-1] * 255).astype(np.uint8)

    return arr
