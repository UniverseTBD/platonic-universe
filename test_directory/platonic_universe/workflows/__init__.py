"""
High-level workflow functions for Platonic Universe.

This module provides complete analysis pipelines that combine model loading,
data processing, embedding generation, and comparison metrics.
"""

from .workflow_runner import WorkflowRunner
from .embedding_comparison import (
    EmbeddingComparison,
    run_embedding_comparison,
    run_cross_modal_analysis,
)
from .model_comparison import (
    ModelComparison, 
    compare_vision_models,
    compare_model_scaling,
)

__all__ = [
    "WorkflowRunner",
    "EmbeddingComparison",
    "run_embedding_comparison", 
    "run_cross_modal_analysis",
    "ModelComparison",
    "compare_vision_models",
    "compare_model_scaling",
]