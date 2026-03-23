import json
import os

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from pu.zoom import resize_galaxy_to_fit

BAND_CONFIG = {
    "hsc": {"indices": [0, 1, 3], "names": ["g", "r", "z"]},
    "jwst": {"indices": [0, 4, 6], "names": ["f090w", "f277w", "f444w"]},
    "legacysurvey": {"indices": [0, 1, 3], "names": ["g", "r", "z"]},
}

RESIZE_CONFIG = {
    "hsc": (68, 92, 68, 92),
    "legacysurvey": (72, 88, 72, 88),
}


def _process_image(blob, mode, resize_mode="match"):
    """Extract bands and apply resize, returning (H, W, 3) float32 array."""
    arr = np.asarray(blob["flux"], np.float32)
    config = BAND_CONFIG[mode]
    if arr.ndim == 3:
        arr = np.stack([arr[i] for i in config["indices"]], axis=-1)
    elif arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    else:
        raise ValueError(f"Array shape {arr.shape} for {mode} not recognised")

    extent = RESIZE_CONFIG.get(mode)
    if extent is not None:
        if resize_mode == "fill":
            arr = resize_galaxy_to_fit(arr, target_size=96)
        else:
            arr = resize_galaxy_to_fit(arr, force_extent=extent, target_size=96)
    return arr


def _percentiles_for_bands(pixels, band_names):
    """Compute p1/p99 per band from a (N, 3) pixel array."""
    return {
        name: {
            "p1": float(np.percentile(pixels[:, i], 1)),
            "p99": float(np.percentile(pixels[:, i], 99)),
        }
        for i, name in enumerate(band_names)
    }


def compute_percentiles(max_samples=10000, resize_mode="match", output_path="data/percentiles.json"):
    """Compute 1st and 99th percentiles for HSC, JWST, and Legacy Survey bands.

    HSC and Legacy Survey percentiles come from legacysurvey_hsc_crossmatched.
    JWST percentiles come from jwst_hsc_crossmatched (filtered for band-3 dynamic range).
    """
    results = {}

    # --- HSC + Legacy Survey ---
    print(f"Loading legacysurvey_hsc_crossmatched (up to {max_samples} samples)...")
    ds = (
        load_dataset("Smith42/legacysurvey_hsc_crossmatched", split="train", streaming=True)
        .select_columns(["hsc_image", "legacysurvey_image"])
        .take(max_samples)
    )

    pixels = {"hsc": [], "legacysurvey": []}
    n_ls = 0
    for sample in tqdm(ds, total=max_samples, desc="HSC + Legacy Survey"):
        for mode in ("hsc", "legacysurvey"):
            arr = _process_image(sample[f"{mode}_image"], mode, resize_mode)
            pixels[mode].append(arr.reshape(-1, 3))
        n_ls += 1

    for mode in ("hsc", "legacysurvey"):
        all_px = np.concatenate(pixels[mode])
        results[mode] = _percentiles_for_bands(all_px, BAND_CONFIG[mode]["names"])
        for band, vals in results[mode].items():
            print(f"  {mode}/{band}: p1={vals['p1']:.6f}, p99={vals['p99']:.6f}")
    print(f"  Processed {n_ls} galaxies for HSC + Legacy Survey")
    del pixels

    # --- JWST ---
    print(f"\nLoading jwst_hsc_crossmatched (up to {max_samples} samples)...")
    ds = (
        load_dataset("Smith42/jwst_hsc_crossmatched", split="train", streaming=True)
        .select_columns(["jwst_image"])
    )

    jwst_pixels = []
    n_jwst = 0
    for sample in tqdm(ds, total=max_samples, desc="JWST"):
        if n_jwst >= max_samples:
            break
        blob = sample["jwst_image"]
        im = np.asarray(blob["flux"][3], np.float32)
        if np.nanpercentile(im, 5) == np.nanpercentile(im, 99):
            continue
        arr = _process_image(blob, "jwst", resize_mode)
        jwst_pixels.append(arr.reshape(-1, 3))
        n_jwst += 1

    all_px = np.concatenate(jwst_pixels)
    results["jwst"] = _percentiles_for_bands(all_px, BAND_CONFIG["jwst"]["names"])
    for band, vals in results["jwst"].items():
        print(f"  jwst/{band}: p1={vals['p1']:.6f}, p99={vals['p99']:.6f}")
    print(f"  Processed {n_jwst} galaxies for JWST")

    results["metadata"] = {
        "n_samples_hsc_legacysurvey": n_ls,
        "n_samples_jwst": n_jwst,
        "max_samples": max_samples,
        "resize_mode": resize_mode,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved percentiles to {output_path}")

    return results
