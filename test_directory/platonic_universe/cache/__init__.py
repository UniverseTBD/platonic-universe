"""
Cache management utilities for Platonic Universe.

This module provides functions for managing HuggingFace cache directories,
disk space monitoring, and cache cleanup operations.
"""

from .cache_manager import (
    setup_cache,
    pick_cache_root,
    clear_cache,
    get_cache_info,
    CacheManager,
)

__all__ = [
    "setup_cache",
    "pick_cache_root", 
    "clear_cache",
    "get_cache_info",
    "CacheManager",
]