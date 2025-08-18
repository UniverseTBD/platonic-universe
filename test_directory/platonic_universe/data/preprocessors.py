"""
Data preprocessing utilities for images and spectra.
"""

from typing import List, Optional, Union, Tuple
import numpy as np
from PIL import Image

try:
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import astropy.units as u
    from specutils import Spectrum1D
    from specutils.manipulation import LinearInterpolatedResampler
    SPECUTILS_AVAILABLE = True
except ImportError:
    SPECUTILS_AVAILABLE = False


def flux_to_pil(arr: Union[np.ndarray, list], 
               p_lo: float = 5, 
               p_hi: float = 99,
               target_size: Optional[int] = None) -> Optional[Image.Image]:
    """
    Convert astronomical flux array to PIL Image.
    
    Args:
        arr: Flux array (can be 1D, 2D, or 3D)
        p_lo: Lower percentile for scaling
        p_hi: Upper percentile for scaling
        target_size: Target size for output image (square)
        
    Returns:
        PIL Image or None if conversion fails
    """
    try:
        a = np.asarray(arr).squeeze()
        
        # Handle different input dimensions
        if a.ndim == 1:
            # Reshape 1D to 2D (assume square)
            s = int(np.sqrt(a.size))
            if s * s != a.size:
                # Pad or truncate to make square
                new_size = s * s
                if a.size > new_size:
                    a = a[:new_size]
                else:
                    a = np.pad(a, (0, new_size - a.size), mode='constant')
            a = a.reshape(s, s)
        elif a.ndim > 2:
            # Take middle slice or first channel
            if a.shape[0] < a.shape[1]:
                a = a[a.shape[0] // 2]  # Middle band
            else:
                a = a[:, :, 0]  # First channel
        
        if a.size == 0:
            return None
        
        # Robust percentile scaling
        v0, v1 = np.nanpercentile(a, p_lo), np.nanpercentile(a, p_hi)
        if not np.isfinite(v0) or not np.isfinite(v1) or v1 <= v0:
            v0, v1 = float(np.nanmin(a)), float(np.nanmax(a))
            if not np.isfinite(v0) or not np.isfinite(v1) or v1 <= v0:
                return None
        
        # Scale to 0-255
        img = ((a - v0) / (v1 - v0)).clip(0, 1)
        img = (img * 255).astype(np.uint8)
        
        # Convert to RGB
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, 2)
        
        pil_img = Image.fromarray(img, "RGB")
        
        # Resize if requested
        if target_size is not None:
            pil_img = pil_img.resize((target_size, target_size), Image.LANCZOS)
        
        return pil_img
        
    except Exception:
        return None


def spectrum_to_pil(flux: Union[np.ndarray, list], 
                   width: int = 224, 
                   height: int = 224,
                   enhanced: bool = True) -> Optional[Image.Image]:
    """
    Convert 1D spectrum to PIL Image.
    
    Args:
        flux: 1D flux array
        width: Output image width
        height: Output image height  
        enhanced: Whether to create enhanced RGB representation
        
    Returns:
        PIL Image or None if conversion fails
    """
    try:
        f = np.asarray(flux).astype(np.float32).ravel()
        if f.size < 16:
            return None
        
        # Interpolate to desired width
        xs = np.linspace(0, f.size - 1, width)
        f = np.interp(xs, np.arange(f.size), f)
        
        # Normalize
        f = (f - np.nanmin(f)) / (np.nanmax(f) - np.nanmin(f) + 1e-8)
        
        if enhanced and SCIPY_AVAILABLE:
            # Enhanced RGB representation
            # R: raw flux
            # G: smoothed flux  
            # B: gradient magnitude
            
            raw = f
            smoothed = gaussian_filter1d(f, sigma=2.0)
            gradient = np.abs(np.gradient(smoothed))
            
            # Normalize each channel
            def normalize(x):
                x = x - np.nanmin(x)
                denom = np.nanmax(x) - np.nanmin(x) + 1e-6
                return (x / denom).clip(0, 1)
            
            R = normalize(raw)
            G = normalize(smoothed)
            B = normalize(gradient)
            
            # Create vertical strips with slight modulation
            rows = []
            yy = np.linspace(0, 2*np.pi, height, endpoint=False)
            for i, y in enumerate(yy):
                modulation = 0.95 + 0.05 * np.sin(y)
                row = np.stack([R, modulation * G, (0.95 + 0.05*np.cos(y)) * B], axis=0)
                rows.append(row)
            
            rgb = np.stack(rows, axis=0)  # (H, 3, W)
            rgb = (rgb.transpose(0, 2, 1) * 255).astype(np.uint8)  # (H, W, 3)
            
        else:
            # Simple grayscale to RGB
            img = np.tile(f[None, :], (height, 1))
            
            # Add slight vertical modulation to avoid degenerate rows
            yy = np.linspace(0, 2*np.pi, height, endpoint=False)
            img *= (0.9 + 0.1*np.sin(yy)[:, None])
            
            img = (img * 255).clip(0, 255).astype(np.uint8)
            rgb = np.stack([img, img, img], axis=2)
        
        return Image.fromarray(rgb, "RGB")
        
    except Exception:
        return None


def interpolate_spectrum(wavelengths: np.ndarray, 
                        flux: np.ndarray,
                        target_wavelengths: np.ndarray,
                        method: str = "linear") -> np.ndarray:
    """
    Interpolate spectrum to new wavelength grid.
    
    Args:
        wavelengths: Original wavelength array
        flux: Original flux array
        target_wavelengths: Target wavelength grid
        method: Interpolation method ('linear' or 'specutils')
        
    Returns:
        Interpolated flux array
    """
    if method == "specutils" and SPECUTILS_AVAILABLE:
        try:
            # Use specutils for proper spectral interpolation
            spectrum = Spectrum1D(
                flux=flux * u.dimensionless_unscaled,
                spectral_axis=wavelengths * u.Angstrom
            )
            
            target_spectral_axis = target_wavelengths * u.Angstrom
            resampler = LinearInterpolatedResampler()
            resampled = resampler(spectrum, target_spectral_axis)
            
            return resampled.flux.value
            
        except Exception:
            # Fallback to numpy
            pass
    
    # Numpy interpolation fallback
    # Clean input data
    mask = np.isfinite(wavelengths) & np.isfinite(flux)
    if not np.any(mask):
        return np.zeros_like(target_wavelengths)
    
    clean_waves = np.asarray(wavelengths)[mask]
    clean_flux = np.asarray(flux)[mask]
    
    # Sort by wavelength
    sort_idx = np.argsort(clean_waves)
    clean_waves = clean_waves[sort_idx]
    clean_flux = clean_flux[sort_idx]
    
    try:
        interpolated_flux = np.interp(
            target_wavelengths, 
            clean_waves, 
            clean_flux,
            left=0.0, 
            right=0.0
        )
        return interpolated_flux
    except Exception:
        return np.zeros_like(target_wavelengths)


def interpolate_sdss_to_desi_grid(sdss_wavelengths: np.ndarray, 
                                 sdss_flux: np.ndarray) -> np.ndarray:
    """
    Interpolate SDSS spectrum to DESI wavelength grid.
    
    DESI typically covers 3600-9800 Å with ~7500 pixels.
    
    Args:
        sdss_wavelengths: SDSS wavelength array
        sdss_flux: SDSS flux array
        
    Returns:
        Flux interpolated to DESI grid
    """
    # DESI wavelength grid (log-spaced from 3600 to 9800 Å)
    desi_wavelengths = np.logspace(np.log10(3600), np.log10(9800), 7500)
    
    return interpolate_spectrum(
        sdss_wavelengths, 
        sdss_flux, 
        desi_wavelengths,
        method="specutils" if SPECUTILS_AVAILABLE else "linear"
    )


class ImagePreprocessor:
    """Preprocessor for astronomical images."""
    
    def __init__(self, 
                 target_size: int = 224,
                 percentile_lo: float = 5,
                 percentile_hi: float = 99):
        """
        Initialize image preprocessor.
        
        Args:
            target_size: Target image size (square)
            percentile_lo: Lower percentile for scaling
            percentile_hi: Upper percentile for scaling
        """
        self.target_size = target_size
        self.percentile_lo = percentile_lo
        self.percentile_hi = percentile_hi
    
    def __call__(self, flux_array: Union[np.ndarray, list]) -> Optional[Image.Image]:
        """
        Preprocess flux array to PIL Image.
        
        Args:
            flux_array: Input flux array
            
        Returns:
            Preprocessed PIL Image or None
        """
        return flux_to_pil(
            flux_array,
            p_lo=self.percentile_lo,
            p_hi=self.percentile_hi,
            target_size=self.target_size
        )


class SpectrumPreprocessor:
    """Preprocessor for astronomical spectra."""
    
    def __init__(self,
                 target_width: int = 224,
                 target_height: int = 224,
                 enhanced_rgb: bool = True):
        """
        Initialize spectrum preprocessor.
        
        Args:
            target_width: Target image width
            target_height: Target image height
            enhanced_rgb: Whether to use enhanced RGB representation
        """
        self.target_width = target_width
        self.target_height = target_height
        self.enhanced_rgb = enhanced_rgb
    
    def __call__(self, flux_array: Union[np.ndarray, list]) -> Optional[Image.Image]:
        """
        Preprocess spectrum to PIL Image.
        
        Args:
            flux_array: Input flux array
            
        Returns:
            Preprocessed PIL Image or None
        """
        return spectrum_to_pil(
            flux_array,
            width=self.target_width,
            height=self.target_height,
            enhanced=self.enhanced_rgb
        )
    
    def interpolate_to_grid(self, 
                           wavelengths: np.ndarray,
                           flux: np.ndarray,
                           target_grid: str = "desi") -> np.ndarray:
        """
        Interpolate spectrum to standard grid.
        
        Args:
            wavelengths: Input wavelength array
            flux: Input flux array
            target_grid: Target grid ('desi', 'sdss', or custom)
            
        Returns:
            Interpolated flux array
        """
        if target_grid == "desi":
            return interpolate_sdss_to_desi_grid(wavelengths, flux)
        elif target_grid == "sdss":
            # SDSS wavelength grid
            target_waves = np.linspace(3800, 9200, 3500)
            return interpolate_spectrum(wavelengths, flux, target_waves)
        else:
            raise ValueError(f"Unknown target grid: {target_grid}")


