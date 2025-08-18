"""
Dataset loading utilities for astronomical data.
"""

from typing import List, Tuple, Optional, Dict, Any, Union
import pathlib
import tqdm
from PIL import Image

try:
    from datasets import load_dataset, DownloadConfig
    from huggingface_hub import snapshot_download
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

from .preprocessors import flux_to_pil, spectrum_to_pil


class DatasetLoader:
    """Base class for loading astronomical datasets."""
    
    def __init__(self, cache_dir: Optional[Union[str, pathlib.Path]] = None):
        """
        Initialize dataset loader.
        
        Args:
            cache_dir: Directory for caching downloaded datasets
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library is required for data loading")
        
        self.cache_dir = str(cache_dir) if cache_dir else None
        self._download_config = None
        if self.cache_dir:
            self._download_config = DownloadConfig(cache_dir=self.cache_dir)
    
    def load_streaming(self, repo_id: str, split: str = "train", 
                      **kwargs) -> Any:
        """
        Load dataset in streaming mode.
        
        Args:
            repo_id: HuggingFace dataset repository ID
            split: Dataset split to load
            **kwargs: Additional arguments for load_dataset
            
        Returns:
            Streaming dataset
        """
        return load_dataset(
            repo_id,
            split=split,
            streaming=True,
            download_config=self._download_config,
            **kwargs
        )
    
    def load_snapshot(self, repo_id: str, local_dir: Optional[str] = None) -> pathlib.Path:
        """
        Download dataset snapshot to local directory.
        
        Args:
            repo_id: HuggingFace dataset repository ID
            local_dir: Local directory to download to
            
        Returns:
            Path to downloaded snapshot
        """
        if local_dir is None:
            local_dir = f"./datasets/{repo_id.replace('/', '_')}"
        
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        
        return pathlib.Path(snapshot_path)


def load_hsc_sdss(max_samples: int = 0, 
                 cache_dir: Optional[str] = None,
                 log_every: int = 1000) -> Tuple[List[Image.Image], List[Image.Image]]:
    """
    Load HSC × SDSS cross-matched dataset.
    
    Args:
        max_samples: Maximum number of samples to load (0 = all)
        cache_dir: Cache directory for datasets
        log_every: How often to log progress
        
    Returns:
        Tuple of (HSC_images, SDSS_images) as PIL Images
    """
    print("📥 Loading HSC × SDSS cross-matched dataset...")
    
    loader = DatasetLoader(cache_dir)
    ds = loader.load_streaming("Smith42/hsc_sdss_crossmatched")
    
    hsc_images, sdss_images = [], []
    kept, seen = 0, 0
    
    for ex in tqdm.tqdm(ds, desc="Loading HSC-SDSS"):
        seen += 1
        
        # Extract HSC image
        hsc_blob = ex.get("image") or ex.get("hsc_image") or ex.get("cutout")
        if isinstance(hsc_blob, dict) and "flux" in hsc_blob:
            hsc_blob = hsc_blob["flux"]
        hsc_img = flux_to_pil(hsc_blob) if hsc_blob is not None else None
        
        # Extract SDSS spectrum and convert to image
        spec = ex.get("spectrum") or ex.get("sdss_spectrum")
        if isinstance(spec, dict):
            spec = spec.get("flux", None)
        sdss_img = spectrum_to_pil(spec) if spec is not None else None
        
        # Keep valid pairs
        if hsc_img is not None and sdss_img is not None:
            hsc_images.append(hsc_img)
            sdss_images.append(sdss_img)
            kept += 1
        
        # Progress logging
        if log_every and seen % log_every == 0:
            print(f"   Processed {seen:,} rows, kept {kept:,} valid pairs")
        
        # Stop if we have enough samples
        if max_samples and kept >= max_samples:
            break
    
    print(f"✅ Loaded {kept:,} HSC-SDSS pairs from {seen:,} total rows")
    
    if kept == 0:
        raise RuntimeError("No valid HSC-SDSS pairs found")
    
    return hsc_images, sdss_images


def load_hsc_jwst(max_samples: int = 0,
                 cache_dir: Optional[str] = None, 
                 log_every: int = 2000,
                 use_snapshot: bool = False) -> Tuple[List[Image.Image], List[Image.Image]]:
    """
    Load HSC × JWST cross-matched dataset.
    
    Args:
        max_samples: Maximum number of samples to load (0 = all)
        cache_dir: Cache directory for datasets
        log_every: How often to log progress
        use_snapshot: Whether to download full snapshot or stream
        
    Returns:
        Tuple of (HSC_images, JWST_images) as PIL Images
    """
    print("📥 Loading HSC × JWST cross-matched dataset...")
    
    loader = DatasetLoader(cache_dir)
    
    if use_snapshot:
        # Download full snapshot
        snapshot_path = loader.load_snapshot("Smith42/jwst_hsc_crossmatched")
        ds = load_dataset(str(snapshot_path), split="train")
    else:
        # Stream dataset
        ds = loader.load_streaming("Smith42/jwst_hsc_crossmatched")
    
    hsc_images, jwst_images = [], []
    kept, seen = 0, 0
    
    for ex in tqdm.tqdm(ds, desc="Loading HSC-JWST"):
        seen += 1
        
        # Extract HSC image
        hsc_blob = ex.get("hsc_image") or ex.get("HSC_image") or ex.get("image")
        if isinstance(hsc_blob, dict) and "flux" in hsc_blob:
            hsc_blob = hsc_blob["flux"]
        hsc_img = flux_to_pil(hsc_blob) if hsc_blob is not None else None
        
        # Extract JWST image
        jwst_blob = ex.get("jwst_image") or ex.get("JWST_image")
        if isinstance(jwst_blob, dict) and "flux" in jwst_blob:
            jwst_blob = jwst_blob["flux"]
        jwst_img = flux_to_pil(jwst_blob) if jwst_blob is not None else None
        
        # Keep valid pairs
        if hsc_img is not None and jwst_img is not None:
            hsc_images.append(hsc_img)
            jwst_images.append(jwst_img)
            kept += 1
        
        # Progress logging
        if log_every and seen % log_every == 0:
            print(f"   Processed {seen:,} rows, kept {kept:,} valid pairs")
        
        # Stop if we have enough samples
        if max_samples and kept >= max_samples:
            break
    
    print(f"✅ Loaded {kept:,} HSC-JWST pairs from {seen:,} total rows")
    
    if kept == 0:
        raise RuntimeError("No valid HSC-JWST pairs found")
    
    return hsc_images, jwst_images


def load_desi_hsc(max_samples: int = 0,
                 cache_dir: Optional[str] = None,
                 log_every: int = 2000,
                 prefer_desi_images: bool = True) -> Tuple[List[Image.Image], List[Image.Image]]:
    """
    Load DESI × HSC cross-matched dataset.
    
    Args:
        max_samples: Maximum number of samples to load (0 = all)
        cache_dir: Cache directory for datasets  
        log_every: How often to log progress
        prefer_desi_images: Whether to prefer DESI images over spectra
        
    Returns:
        Tuple of (HSC_images, DESI_images) as PIL Images
    """
    print("📥 Loading DESI × HSC cross-matched dataset...")
    
    loader = DatasetLoader(cache_dir)
    ds = loader.load_streaming("Smith42/desi_hsc_crossmatched")
    
    hsc_images, desi_images = [], []
    kept, seen = 0, 0
    
    # DESI image field names to try
    desi_image_keys = [
        "desi_image", "desi_cutout", "desi_rgb", "legacy_image", 
        "ls_image", "desi_dr9_image", "desi_dr10_image", "legacy_survey_image"
    ]
    
    for ex in tqdm.tqdm(ds, desc="Loading DESI-HSC"):
        seen += 1
        
        # Extract HSC image
        hsc_blob = ex.get("hsc_image") or ex.get("image") or ex.get("hsc")
        if isinstance(hsc_blob, dict) and "flux" in hsc_blob:
            hsc_blob = hsc_blob["flux"]
        hsc_img = flux_to_pil(hsc_blob) if hsc_blob is not None else None
        
        # Extract DESI data (prefer images if available)
        desi_img = None
        
        if prefer_desi_images:
            # Try DESI images first
            for key in desi_image_keys:
                if key in ex:
                    val = ex[key]
                    if isinstance(val, dict) and "flux" in val:
                        val = val["flux"]
                    desi_img = flux_to_pil(val)
                    if desi_img is not None:
                        break
        
        # Fallback to DESI spectrum if no image found
        if desi_img is None:
            spec_blob = ex.get("desi_spectrum") or ex.get("spectrum") or ex.get("desi")
            if isinstance(spec_blob, dict) and "flux" in spec_blob:
                spec_blob = spec_blob["flux"]
            if spec_blob is not None:
                desi_img = spectrum_to_pil(spec_blob)
        
        # Keep valid pairs
        if hsc_img is not None and desi_img is not None:
            hsc_images.append(hsc_img)
            desi_images.append(desi_img)
            kept += 1
        
        # Progress logging
        if log_every and seen % log_every == 0:
            print(f"   Processed {seen:,} rows, kept {kept:,} valid pairs")
        
        # Stop if we have enough samples
        if max_samples and kept >= max_samples:
            break
    
    print(f"✅ Loaded {kept:,} DESI-HSC pairs from {seen:,} total rows")
    
    if kept == 0:
        raise RuntimeError("No valid DESI-HSC pairs found")
    
    return hsc_images, desi_images


def load_custom_dataset(repo_id: str,
                       max_samples: int = 0,
                       cache_dir: Optional[str] = None,
                       log_every: int = 1000,
                       image_field_a: str = "image_a",
                       image_field_b: str = "image_b",
                       flux_field_a: str = "flux_a", 
                       flux_field_b: str = "flux_b") -> Tuple[List[Image.Image], List[Image.Image]]:
    """
    Load a custom cross-matched dataset from HuggingFace.
    
    Args:
        repo_id: HuggingFace dataset repository ID
        max_samples: Maximum number of samples to load (0 = all)
        cache_dir: Cache directory for datasets
        log_every: How often to log progress
        image_field_a: Field name for first set of images
        image_field_b: Field name for second set of images  
        flux_field_a: Field name for first set of flux data (fallback)
        flux_field_b: Field name for second set of flux data (fallback)
        
    Returns:
        Tuple of (images_a, images_b) as PIL Images
    """
    print(f"📥 Loading custom dataset: {repo_id}")
    
    loader = DatasetLoader(cache_dir)
    ds = loader.load_streaming(repo_id)
    
    images_a, images_b = [], []
    kept, seen = 0, 0
    
    for ex in tqdm.tqdm(ds, desc=f"Loading {repo_id}"):
        seen += 1
        
        # Extract first data type
        data_a = ex.get(image_field_a) or ex.get(flux_field_a)
        if isinstance(data_a, dict) and "flux" in data_a:
            data_a = data_a["flux"]
        img_a = flux_to_pil(data_a) if data_a is not None else None
        
        # Extract second data type
        data_b = ex.get(image_field_b) or ex.get(flux_field_b)
        if isinstance(data_b, dict) and "flux" in data_b:
            data_b = data_b["flux"]
        img_b = flux_to_pil(data_b) if data_b is not None else None
        
        # Keep valid pairs
        if img_a is not None and img_b is not None:
            images_a.append(img_a)
            images_b.append(img_b)
            kept += 1
        
        # Progress logging
        if log_every and seen % log_every == 0:
            print(f"   Processed {seen:,} rows, kept {kept:,} valid pairs")
        
        # Stop if we have enough samples
        if max_samples and kept >= max_samples:
            break
    
    print(f"✅ Loaded {kept:,} pairs from {repo_id}")
    
    if kept == 0:
        raise RuntimeError(f"No valid pairs found in {repo_id}")
    
    return images_a, images_b