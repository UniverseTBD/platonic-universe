import numpy as np
import json
import os
from PIL import Image

def flux_to_pil(blob: dict) -> Image.Image | None:
    """
    Converts a raw flux data blob into a PIL Image.
    This function performs contrast stretching and converts the data to an 8-bit RGB image.
    
    Args:
        blob (dict): A dictionary-like object containing a "flux" key with image data.

    Returns:
        Image.Image or None: The converted PIL image, or None if the data has no contrast.
    """
    # Extract the flux array as a float32 NumPy array
    arr = np.asarray(blob["flux"], dtype=np.float32)

    # If the array has multiple bands, select the middle one
    if arr.ndim == 3:
        arr = arr[arr.shape[0] // 2]

    # Calculate 5th and 99th percentiles for contrast stretching, ignoring NaNs
    v0, v1 = np.nanpercentile(arr, 5), np.nanpercentile(arr, 99)

    # If contrast is invalid (e.g., all pixels are the same), return None
    if v1 <= v0:
        return None

    # Normalize the array to a 0-1 range, clipping any outliers
    img_normalized = ((arr - v0) / (v1 - v0)).clip(0, 1)
    
    # Scale to 8-bit (0-255) and convert to an unsigned integer array
    img_uint8 = (img_normalized * 255).astype(np.uint8)

    # If the image is grayscale, duplicate the channel to create an RGB image
    if img_uint8.ndim == 2:
        img_rgb = np.repeat(img_uint8[:, :, np.newaxis], 3, axis=2)
    else:
        img_rgb = img_uint8

    # Return the final array as a PIL Image in "RGB" mode
    return Image.fromarray(img_rgb, "RGB")


def load_normalization_lut(lut_path: str = None) -> dict:
    """
    Load normalization lookup table with 5th-99th percentile values.
    
    Args:
        lut_path: Path to LUT JSON file. If None, uses default package location.
        
    Returns:
        dict: Normalization values by survey and band
    """
    if lut_path is None:
        # Try package data location first (when installed)
        try:
            # Python 3.9+ preferred method
            import importlib.resources as pkg_resources
            with pkg_resources.open_text('platonic_universe', 'normalization_lut.json') as f:
                return json.load(f)
        except (ImportError, FileNotFoundError, ModuleNotFoundError):
            try:
                # Python 3.7-3.8 fallback
                import pkg_resources
                lut_data = pkg_resources.resource_string('platonic_universe', 'normalization_lut.json')
                return json.loads(lut_data.decode('utf-8'))
            except (ImportError, FileNotFoundError, ModuleNotFoundError):
                # Final fallback to package directory or development location
                current_dir = os.path.dirname(__file__)
                
                # Try package directory first (after moving file there)
                pkg_lut_path = os.path.join(current_dir, "../normalization_lut.json")
                if os.path.exists(pkg_lut_path):
                    lut_path = pkg_lut_path
                else:
                    # Development location fallback
                    lut_path = os.path.join(current_dir, "../../../normalization_lut.json")
                    
                    # Check if the file exists before trying to open it
                    if not os.path.exists(lut_path):
                        raise FileNotFoundError(
                            f"normalization_lut.json not found. Tried package data, {pkg_lut_path}, and {lut_path}. "
                            f"Make sure the package was built and installed correctly."
                        )
    
    with open(lut_path, "r") as f:
        return json.load(f)


def _slice_band_from_image(image_dict: dict, want_band: str) -> np.ndarray | None:
    """
    Extract specific band from multi-band image.
    
    image_dict has 'band' list and 'flux' array shaped (B,H,W) or (H,W,B).
    Returns 2D float32 array for requested band, or None.
    """
    if not image_dict or "flux" not in image_dict or "band" not in image_dict:
        return None
    bands = [str(b).lower() for b in image_dict["band"]]
    want = want_band.lower()
    if want not in bands:
        tag = want.split("-")[-1]
        idxs = [i for i, b in enumerate(bands) if b.endswith("-" + tag) or b == tag]
        if not idxs:
            return None
        idx = idxs[0]
    else:
        idx = bands.index(want)

    flux = np.asarray(image_dict["flux"], dtype=np.float32)
    if flux.ndim == 3 and flux.shape[0] == len(bands):
        slab = flux[idx, ...]
    elif flux.ndim == 3 and flux.shape[-1] == len(bands):
        slab = flux[..., idx]
    elif flux.ndim == 2 and len(bands) == 1 and idx == 0:
        slab = flux
    else:
        return None
    return np.asarray(slab, dtype=np.float32)


def _center_crop_to_smallest(*arrs):
    """Center crop all arrays to the smallest common dimensions."""
    h = min(x.shape[0] for x in arrs)
    w = min(x.shape[1] for x in arrs)
    out = []
    for x in arrs:
        dh = (x.shape[0] - h) // 2
        dw = (x.shape[1] - w) // 2
        out.append(x[dh:dh+h, dw:dw+w])
    return out


def _scale_with_lut(a, lo, hi):
    """Scale array to 0-255 range using LUT bounds."""
    v0, v1 = float(lo), float(hi)
    a = np.asarray(a, dtype=np.float32)
    if not np.isfinite(v0) or not np.isfinite(v1) or v1 <= v0:
        return None
    img = ((a - v0) / (v1 - v0)).clip(0, 1) * 255.0
    return img.astype(np.uint8)


def hsc_to_pil_using_lut(sample: dict, survey: str = "HSC", lut_path: str = None) -> Image.Image | None:
    """
    Create RGB PIL image from HSC g,r,z bands using lookup table normalization.
    
    Args:
        sample: Dataset sample containing 'image' or 'hsc_image' key with band data
        survey: Survey name in LUT ("HSC", "JWST", "LEGACY")
        lut_path: Path to normalization LUT file
        
    Returns:
        PIL RGB image with R=z, G=r, B=g channels, or None if processing fails
    """
    # Load normalization values
    lut = load_normalization_lut(lut_path)
    if survey not in lut:
        raise ValueError(f"Survey '{survey}' not found in LUT. Available: {list(lut.keys())}")
    
    survey_lut = lut[survey]
    
    # Extract image data
    img = sample.get("image") or sample.get("hsc_image") or sample.get("cutout")
    if not isinstance(img, dict):
        return None
    
    # Extract g, r, z bands
    g2 = _slice_band_from_image(img, "hsc-g")
    r2 = _slice_band_from_image(img, "hsc-r")  
    z2 = _slice_band_from_image(img, "hsc-z")
    if g2 is None or r2 is None or z2 is None:
        return None
    
    # Apply LUT normalization
    g8 = _scale_with_lut(g2, *survey_lut["g"])
    r8 = _scale_with_lut(r2, *survey_lut["r"])
    z8 = _scale_with_lut(z2, *survey_lut["z"])
    if g8 is None or r8 is None or z8 is None:
        return None
    
    # Center crop and create RGB
    z8c, r8c, g8c = _center_crop_to_smallest(z8, r8, g8)
    rgb = np.stack([z8c, r8c, g8c], axis=2)  # R=z, G=r, B=g
    return Image.fromarray(rgb, "RGB")


def jwst_to_pil_using_lut(sample: dict, lut_path: str = None) -> Image.Image | None:
    """
    Create RGB PIL image from JWST f444w,f277w,f090w bands using lookup table normalization.
    
    Args:
        sample: Dataset sample containing 'jwst_image' key with band data
        lut_path: Path to normalization LUT file
        
    Returns:
        PIL RGB image with R=f444w, G=f277w, B=f090w channels, or None if processing fails
    """
    # Load normalization values
    lut = load_normalization_lut(lut_path)
    jwst_lut = lut["JWST"]
    
    # Extract image data
    img = sample.get("jwst_image") or sample.get("image")
    if not isinstance(img, dict):
        return None
    
    # Extract JWST bands
    f090w = _slice_band_from_image(img, "f090w")
    f277w = _slice_band_from_image(img, "f277w")
    f444w = _slice_band_from_image(img, "f444w")
    if f090w is None or f277w is None or f444w is None:
        return None
    
    # Apply LUT normalization
    f090w_norm = _scale_with_lut(f090w, *jwst_lut["f090w"])
    f277w_norm = _scale_with_lut(f277w, *jwst_lut["f277w"])
    f444w_norm = _scale_with_lut(f444w, *jwst_lut["f444w"])
    if f090w_norm is None or f277w_norm is None or f444w_norm is None:
        return None
    
    # Center crop and create RGB
    f444w_c, f277w_c, f090w_c = _center_crop_to_smallest(f444w_norm, f277w_norm, f090w_norm)
    rgb = np.stack([f444w_c, f277w_c, f090w_c], axis=2)  # R=f444w, G=f277w, B=f090w
    return Image.fromarray(rgb, "RGB")


def legacy_to_pil_using_lut(sample: dict, lut_path: str = None) -> Image.Image | None:
    """
    Create RGB PIL image from Legacy Survey g,r,z bands using lookup table normalization.
    
    Args:
        sample: Dataset sample containing 'legacy_image' key with band data
        lut_path: Path to normalization LUT file
        
    Returns:
        PIL RGB image with R=z, G=r, B=g channels, or None if processing fails
    """
    import logging
    
    try:
        # Load normalization values
        lut = load_normalization_lut(lut_path)
        legacy_lut = lut["LEGACY"]
        
        # Extract image data - handle different naming conventions
        img = (sample.get("legacy_image") or 
               sample.get("legacysurvey_image") or 
               sample.get("image"))
        if not isinstance(img, dict):
            logging.debug(f"Legacy LUT: No valid image dict found. Sample keys: {list(sample.keys())}")
            return None
        
        logging.debug(f"Legacy LUT: Found image dict with keys: {list(img.keys())}")
        
        # Extract g, r, z bands (Legacy uses same bands as HSC)
        g2 = _slice_band_from_image(img, "g")
        r2 = _slice_band_from_image(img, "r")
        z2 = _slice_band_from_image(img, "z")
        
        logging.debug(f"Legacy LUT: Band extraction - g: {g2 is not None}, r: {r2 is not None}, z: {z2 is not None}")
        
        if g2 is None or r2 is None or z2 is None:
            return None
        
        # Apply LUT normalization
        g8 = _scale_with_lut(g2, *legacy_lut["g"])
        r8 = _scale_with_lut(r2, *legacy_lut["r"])
        z8 = _scale_with_lut(z2, *legacy_lut["z"])
        
        logging.debug(f"Legacy LUT: Scaling - g: {g8 is not None}, r: {r8 is not None}, z: {z8 is not None}")
        
        if g8 is None or r8 is None or z8 is None:
            return None
        
        # Center crop and create RGB
        z8c, r8c, g8c = _center_crop_to_smallest(z8, r8, g8)
        rgb = np.stack([z8c, r8c, g8c], axis=2)  # R=z, G=r, B=g
        return Image.fromarray(rgb, "RGB")
        
    except Exception as e:
        logging.debug(f"Legacy LUT processing failed: {e}")
        return None


def extract_desi_flux(sample: dict):
    """
    Extract DESI spectrum flux from dataset sample.
    
    Args:
        sample: Dataset sample containing spectrum data
        
    Returns:
        Flux array or None if not found
    """
    spec = sample.get("spectrum")
    if spec is None:
        return None
    if isinstance(spec, dict) and "flux" in spec:
        return spec["flux"]
    if isinstance(spec, (list, tuple, np.ndarray)):
        return spec
    return None


def spectrum_to_pil(flux, width=224, height=224):
    """
    Convert DESI 1-D spectrum to 224×224 RGB strip image.
    
    Args:
        flux: 1D spectrum array
        width: Output image width
        height: Output image height
        
    Returns:
        PIL RGB image or None if processing fails
    """
    try:
        f = np.asarray(flux).astype(np.float32).ravel()
        if f.size < 16:
            return None
        xs = np.linspace(0, f.size - 1, width)
        f = np.interp(xs, np.arange(f.size), f)
        f = (f - np.nanmin(f)) / (np.nanmax(f) - np.nanmin(f) + 1e-8)
        img = np.tile(f[None, :], (height, 1))
        yy = np.linspace(0, 2*np.pi, height, endpoint=False)
        img *= (0.9 + 0.1*np.sin(yy)[:, None])
        img = (img * 255).clip(0, 255).astype(np.uint8)
        rgb = np.stack([img, img, img], 2)
        return Image.fromarray(rgb, "RGB")
    except Exception:
        return None


def image_to_pil_with_lut(sample: dict, column: str, dataset_alias: str) -> Image.Image | None:
    """
    Auto-detect survey type and apply appropriate processing.
    
    Args:
        sample: Dataset sample
        column: Column name containing image/spectrum data
        dataset_alias: Dataset alias to determine survey type
        
    Returns:
        PIL RGB image or None if processing fails
    """
    # Handle DESI spectra
    if "desi" in dataset_alias.lower() and column == "spectrum":
        flux = extract_desi_flux(sample)
        if flux is not None:
            return spectrum_to_pil(flux)
        return None
    
    # Determine survey type from dataset alias and column for images
    if "hsc" in dataset_alias.lower() and ("hsc" in column or column == "image"):
        return hsc_to_pil_using_lut(sample, survey="HSC")
    elif "jwst" in dataset_alias.lower() and "jwst" in column:
        return jwst_to_pil_using_lut(sample)
    elif "legacy" in dataset_alias.lower() and "legacy" in column:
        return legacy_to_pil_using_lut(sample)
    else:
        # Fallback to original flux_to_pil for unknown types
        return flux_to_pil(sample[column])