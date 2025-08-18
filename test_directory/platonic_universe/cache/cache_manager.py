"""
Cache management for HuggingFace models and datasets.
"""

import os
import pathlib
import shutil
from typing import List, Dict, Any, Optional, Union
import warnings


class CacheManager:
    """Manages HuggingFace cache directories and disk space."""
    
    def __init__(self, cache_root: Optional[Union[str, pathlib.Path]] = None):
        """
        Initialize cache manager.
        
        Args:
            cache_root: Root directory for caches. If None, will auto-select.
        """
        self.cache_root = None
        self.is_setup = False
        
        if cache_root:
            self.setup_cache(cache_root)
    
    def setup_cache(self, cache_root: Union[str, pathlib.Path]) -> None:
        """
        Setup cache directory and environment variables.
        
        Args:
            cache_root: Root directory for all caches
        """
        self.cache_root = pathlib.Path(cache_root)
        
        # Create subdirectories
        for sub in ("hub", "datasets", "models"):
            (self.cache_root / sub).mkdir(parents=True, exist_ok=True)
        
        # Set environment variables
        os.environ.update({
            "HF_HOME": str(self.cache_root),
            "HUGGINGFACE_HUB_CACHE": str(self.cache_root / "hub"),
            "HF_DATASETS_CACHE": str(self.cache_root / "datasets"),
            "TRANSFORMERS_CACHE": str(self.cache_root / "models"),
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        })
        
        self.is_setup = True
    
    def pick_cache_root(self, candidates: Optional[List[str]] = None, 
                       need_gb: float = 3.0) -> pathlib.Path:
        """
        Automatically select a suitable cache directory.
        
        Args:
            candidates: List of candidate directories to try
            need_gb: Minimum free space required in GB
            
        Returns:
            Selected cache root path
            
        Raises:
            OSError: If no suitable location is found
        """
        if candidates is None:
            user = os.getenv("USER") or os.getenv("LOGNAME") or "user"
            candidates = [
                f"/tmp/hf_cache_{user}",
                "/dev/shm/hf_cache",
                "/run/shm/hf_cache", 
                "/scratch/hf_cache",
                "/local_scratch/hf_cache",
                str(pathlib.Path.cwd() / "hf_cache_local"),
            ]
        
        print(f"🔍 Searching for cache location with {need_gb:.1f}GB free space...")
        
        for root in candidates:
            try:
                p = pathlib.Path(root)
                p.mkdir(parents=True, exist_ok=True)
                
                # Check if writable
                if not os.access(root, os.W_OK):
                    print(f"   ❌ {root} - not writable")
                    continue
                
                # Check disk space
                stat = shutil.disk_usage(root)
                free_gb = stat.free / (1024**3)
                print(f"   📊 {root} - {free_gb:.2f}GB free", end="")
                
                if free_gb < need_gb:
                    print(" - insufficient space")
                    continue
                
                # Quick write test
                test_file = p / f"cache_test_{os.getpid()}.tmp"
                try:
                    with open(test_file, "wb") as f:
                        f.write(b"test")
                    test_file.unlink(missing_ok=True)
                except Exception:
                    print(" - write test failed")
                    continue
                
                print(" - ✅ SELECTED")
                return p
                
            except Exception as e:
                print(f"   ❌ {root} - error: {e}")
                continue
        
        # Last resort fallback
        user = os.getenv("USER") or os.getenv("LOGNAME") or "user"
        fallback = pathlib.Path(f"/tmp/hf_cache_{user}_lastresort")
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"⚠️ Using fallback: {fallback}")
        return fallback
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about current cache setup.
        
        Returns:
            Dictionary with cache information
        """
        if not self.is_setup or not self.cache_root:
            return {"setup": False, "cache_root": None}
        
        info = {
            "setup": True,
            "cache_root": str(self.cache_root),
            "subdirectories": {},
            "environment_vars": {},
            "disk_usage": {}
        }
        
        # Check subdirectories
        for sub in ("hub", "datasets", "models"):
            sub_path = self.cache_root / sub
            info["subdirectories"][sub] = {
                "path": str(sub_path),
                "exists": sub_path.exists(),
                "size_mb": self._get_directory_size(sub_path) if sub_path.exists() else 0
            }
        
        # Check environment variables
        env_vars = [
            "HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE", 
            "TRANSFORMERS_CACHE", "HF_HUB_DISABLE_SYMLINKS_WARNING"
        ]
        for var in env_vars:
            info["environment_vars"][var] = os.environ.get(var)
        
        # Disk usage
        try:
            stat = shutil.disk_usage(self.cache_root)
            info["disk_usage"] = {
                "total_gb": stat.total / (1024**3),
                "used_gb": (stat.total - stat.free) / (1024**3),
                "free_gb": stat.free / (1024**3),
                "percent_used": ((stat.total - stat.free) / stat.total) * 100
            }
        except Exception:
            info["disk_usage"] = {"error": "Could not get disk usage"}
        
        return info
    
    def clear_cache(self, subdirs: Optional[List[str]] = None, 
                   confirm: bool = True) -> Dict[str, bool]:
        """
        Clear cache directories.
        
        Args:
            subdirs: List of subdirectories to clear. If None, clears all.
            confirm: Whether to print confirmation messages
            
        Returns:
            Dictionary indicating success/failure for each subdirectory
        """
        if not self.is_setup or not self.cache_root:
            raise RuntimeError("Cache not setup. Call setup_cache() first.")
        
        if subdirs is None:
            subdirs = ["hub", "datasets", "models"]
        
        results = {}
        
        for subdir in subdirs:
            subdir_path = self.cache_root / subdir
            try:
                if subdir_path.exists():
                    size_before = self._get_directory_size(subdir_path)
                    shutil.rmtree(subdir_path, ignore_errors=True)
                    subdir_path.mkdir(parents=True, exist_ok=True)
                    
                    if confirm:
                        print(f"✅ Cleared {subdir} cache ({size_before:.1f} MB)")
                    results[subdir] = True
                else:
                    if confirm:
                        print(f"➖ {subdir} cache directory not found")
                    results[subdir] = True
                    
            except Exception as e:
                if confirm:
                    print(f"⚠️ Failed to clear {subdir} cache: {e}")
                results[subdir] = False
        
        return results
    
    def _get_directory_size(self, path: pathlib.Path) -> float:
        """Get directory size in MB."""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0


# Global cache manager instance
_global_cache_manager = CacheManager()


def setup_cache(cache_root: Union[str, pathlib.Path]) -> None:
    """
    Setup global cache directory.
    
    Args:
        cache_root: Root directory for all caches
    """
    _global_cache_manager.setup_cache(cache_root)


def pick_cache_root(candidates: Optional[List[str]] = None, 
                   need_gb: float = 3.0) -> pathlib.Path:
    """
    Automatically select and setup a suitable cache directory.
    
    Args:
        candidates: List of candidate directories to try
        need_gb: Minimum free space required in GB
        
    Returns:
        Selected cache root path
    """
    cache_root = _global_cache_manager.pick_cache_root(candidates, need_gb)
    _global_cache_manager.setup_cache(cache_root)
    return cache_root


def clear_cache(subdirs: Optional[List[str]] = None, 
               confirm: bool = True) -> Dict[str, bool]:
    """
    Clear global cache directories.
    
    Args:
        subdirs: List of subdirectories to clear. If None, clears all.
        confirm: Whether to print confirmation messages
        
    Returns:
        Dictionary indicating success/failure for each subdirectory
    """
    return _global_cache_manager.clear_cache(subdirs, confirm)


def get_cache_info() -> Dict[str, Any]:
    """
    Get information about current global cache setup.
    
    Returns:
        Dictionary with cache information
    """
    return _global_cache_manager.get_cache_info()


def force_clear_local_caches(cwd: Optional[pathlib.Path] = None) -> None:
    """
    Force clear any cache directories from current working directory.
    
    Args:
        cwd: Working directory to clean. If None, uses current directory.
    """
    if cwd is None:
        cwd = pathlib.Path.cwd()
    
    cache_dirs_to_remove = [
        ".cache",
        "cache", 
        "huggingface_cache",
        "transformers_cache",
        "datasets_cache",
        "__pycache__"
    ]
    
    # Look for process-specific cache directories
    try:
        for item in cwd.iterdir():
            if item.is_dir() and any(pattern in item.name.lower() for pattern in 
                                   ["cache", "hf_", "_cache_"]):
                cache_dirs_to_remove.append(item.name)
    except Exception:
        pass
    
    print("🧹 Force clearing local cache directories...")
    print(f"   Checking directory: {cwd}")
    
    for cache_dir in cache_dirs_to_remove:
        cache_path = cwd / cache_dir
        print(f"   Checking: {cache_path}")
        if cache_path.exists():
            try:
                shutil.rmtree(cache_path, ignore_errors=True)
                print(f"     ✅ Removed: {cache_path}")
            except Exception as e:
                print(f"     ⚠️ Could not remove {cache_path}: {e}")
        else:
            print(f"     ➖ Not found: {cache_path}")
    
    print("   Local cache cleanup complete.")


def check_disk_space(path: Union[str, pathlib.Path] = ".", 
                    min_gb: float = 1.0) -> tuple[bool, float]:
    """
    Check if sufficient disk space is available.
    
    Args:
        path: Path to check
        min_gb: Minimum required space in GB
        
    Returns:
        Tuple of (has_enough_space, free_gb)
    """
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024**3)
        return free_gb >= min_gb, free_gb
    except Exception:
        return True, float('inf')  # Assume OK if can't check