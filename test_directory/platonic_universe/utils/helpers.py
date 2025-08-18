"""
General helper utilities for Platonic Universe.
"""

import os
import random
from typing import Any, Union, Optional, Dict
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def clear_gpu_memory() -> None:
    """Clear GPU memory cache if PyTorch is available."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    info = {"torch_available": TORCH_AVAILABLE}
    
    if TORCH_AVAILABLE:
        info.update({
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "current_device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        })
        
        if torch.cuda.is_available():
            info["cuda_devices"] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["cuda_devices"].append({
                    "device_id": i,
                    "name": props.name,
                    "total_memory_gb": props.total_memory / (1024**3),
                    "compute_capability": f"{props.major}.{props.minor}"
                })
    else:
        info.update({
            "cuda_available": False,
            "cuda_device_count": 0,
            "current_device": "cpu",
        })
    
    return info


def safe_divide(a: Union[float, np.ndarray], 
               b: Union[float, np.ndarray], 
               default: float = 0.0,
               epsilon: float = 1e-8) -> Union[float, np.ndarray]:
    """
    Safely divide two numbers/arrays, avoiding division by zero.
    
    Args:
        a: Numerator
        b: Denominator  
        default: Default value when denominator is zero
        epsilon: Small value added to denominator for numerical stability
        
    Returns:
        Result of safe division
    """
    if isinstance(b, np.ndarray):
        result = np.full_like(b, default, dtype=float)
        nonzero_mask = np.abs(b) > epsilon
        result[nonzero_mask] = a[nonzero_mask] / b[nonzero_mask] if isinstance(a, np.ndarray) else a / b[nonzero_mask]
        return result
    else:
        return a / (b + epsilon) if abs(b) > epsilon else default


def ensure_numpy(data: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """
    Convert data to numpy array.
    
    Args:
        data: Input data (list, tensor, etc.)
        dtype: Target numpy dtype
        
    Returns:
        Numpy array
    """
    if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
        # Convert PyTorch tensor
        if data.is_cuda:
            data = data.cpu()
        data = data.detach().numpy()
    
    # Convert to numpy array
    arr = np.asarray(data)
    
    # Set dtype if specified
    if dtype is not None:
        arr = arr.astype(dtype)
    
    return arr


def format_memory_size(size_bytes: int) -> str:
    """
    Format memory size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def get_memory_usage() -> Dict[str, Any]:
    """
    Get current memory usage information.
    
    Returns:
        Dictionary with memory usage info
    """
    import psutil
    
    # System memory
    memory = psutil.virtual_memory()
    info = {
        "system_memory": {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent,
            "total_formatted": format_memory_size(memory.total),
            "available_formatted": format_memory_size(memory.available),
            "used_formatted": format_memory_size(memory.used),
        }
    }
    
    # GPU memory
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            total = torch.cuda.get_device_properties(i).total_memory
            
            gpu_info.append({
                "device_id": i,
                "allocated": allocated,
                "reserved": reserved, 
                "total": total,
                "allocated_formatted": format_memory_size(allocated),
                "reserved_formatted": format_memory_size(reserved),
                "total_formatted": format_memory_size(total),
                "percent_allocated": (allocated / total) * 100 if total > 0 else 0,
            })
        
        info["gpu_memory"] = gpu_info
    
    return info


def check_requirements() -> Dict[str, Any]:
    """
    Check if required packages are available.
    
    Returns:
        Dictionary with availability status
    """
    requirements = {}
    
    # Core requirements
    try:
        import numpy
        requirements["numpy"] = {"available": True, "version": numpy.__version__}
    except ImportError:
        requirements["numpy"] = {"available": False, "version": None}
    
    try:
        import torch
        requirements["torch"] = {"available": True, "version": torch.__version__}
    except ImportError:
        requirements["torch"] = {"available": False, "version": None}
    
    try:
        import transformers
        requirements["transformers"] = {"available": True, "version": transformers.__version__}
    except ImportError:
        requirements["transformers"] = {"available": False, "version": None}
    
    try:
        import timm
        requirements["timm"] = {"available": True, "version": timm.__version__}
    except ImportError:
        requirements["timm"] = {"available": False, "version": None}
    
    try:
        from sklearn import __version__ as sklearn_version
        requirements["scikit-learn"] = {"available": True, "version": sklearn_version}
    except ImportError:
        requirements["scikit-learn"] = {"available": False, "version": None}
    
    try:
        import datasets
        requirements["datasets"] = {"available": True, "version": datasets.__version__}
    except ImportError:
        requirements["datasets"] = {"available": False, "version": None}
    
    # Optional requirements
    try:
        import scipy
        requirements["scipy"] = {"available": True, "version": scipy.__version__}
    except ImportError:
        requirements["scipy"] = {"available": False, "version": None}
    
    try:
        import astropy
        requirements["astropy"] = {"available": True, "version": astropy.__version__}
    except ImportError:
        requirements["astropy"] = {"available": False, "version": None}
    
    try:
        import specutils
        requirements["specutils"] = {"available": True, "version": specutils.__version__}
    except ImportError:
        requirements["specutils"] = {"available": False, "version": None}
    
    return requirements


def validate_embeddings(embeddings: np.ndarray, 
                       name: str = "embeddings") -> Dict[str, Any]:
    """
    Validate embedding array and return diagnostic information.
    
    Args:
        embeddings: Embedding array to validate
        name: Name for logging
        
    Returns:
        Dictionary with validation results
    """
    results = {"name": name, "valid": True, "issues": []}
    
    # Basic shape check
    if embeddings.ndim != 2:
        results["valid"] = False
        results["issues"].append(f"Expected 2D array, got {embeddings.ndim}D")
    
    # Check for NaN/inf values
    if np.any(np.isnan(embeddings)):
        results["valid"] = False
        results["issues"].append("Contains NaN values")
    
    if np.any(np.isinf(embeddings)):
        results["valid"] = False
        results["issues"].append("Contains infinite values")
    
    # Check for all-zero rows
    if embeddings.ndim == 2:
        zero_rows = np.all(embeddings == 0, axis=1)
        if np.any(zero_rows):
            results["issues"].append(f"{np.sum(zero_rows)} rows are all zeros")
    
    # Check for constant features
    if embeddings.ndim == 2 and embeddings.shape[1] > 1:
        constant_features = np.var(embeddings, axis=0) < 1e-10
        if np.any(constant_features):
            results["issues"].append(f"{np.sum(constant_features)} features are constant")
    
    # Basic statistics
    results["statistics"] = {
        "shape": embeddings.shape,
        "dtype": str(embeddings.dtype),
        "mean": float(np.mean(embeddings)),
        "std": float(np.std(embeddings)),
        "min": float(np.min(embeddings)),
        "max": float(np.max(embeddings)),
    }
    
    return results


def create_progress_callback(description: str = "Processing"):
    """
    Create a simple progress callback function.
    
    Args:
        description: Description for progress bar
        
    Returns:
        Progress callback function
    """
    def callback(current: int, total: int):
        if total > 0:
            percent = (current / total) * 100
            print(f"\r{description}: {current}/{total} ({percent:.1f}%)", end="", flush=True)
        else:
            print(f"\r{description}: {current}", end="", flush=True)
    
    return callback