"""
Model comparison workflows and utilities.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass

from ..models import ModelLoader
from ..utils import compute_mknn_prh, validate_embeddings
from .embedding_comparison import EmbeddingComparison, EmbeddingComparisonConfig


@dataclass 
class ModelComparisonConfig:
    """Configuration for model comparison."""
    k_values: List[int] = None
    batch_size: int = 32
    feature_extraction_kwargs: Dict[str, Any] = None
    random_seed: int = 42
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [5, 10, 20, 50]
        if self.feature_extraction_kwargs is None:
            self.feature_extraction_kwargs = {}


class ModelComparison:
    """Class for comparing different models on the same data."""
    
    def __init__(self, config: Optional[ModelComparisonConfig] = None):
        """
        Initialize model comparison.
        
        Args:
            config: Configuration for comparison
        """
        self.config = config or ModelComparisonConfig()
        self.embedding_comparison = EmbeddingComparison(
            EmbeddingComparisonConfig(
                k_values=self.config.k_values,
                random_seed=self.config.random_seed
            )
        )
    
    def compare_vision_models(self,
                            model_specs: List[Dict[str, str]],
                            images: List,
                            reference_model: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare multiple vision models on the same images.
        
        Args:
            model_specs: List of model specifications [{"type": "vit", "id": "google/vit-base-patch16-224"}, ...]
            images: List of PIL Images to process
            reference_model: Name of reference model for comparison
            
        Returns:
            Model comparison results
        """
        print(f"🔬 Comparing {len(model_specs)} vision models on {len(images)} images")
        
        # Load models and extract embeddings
        model_embeddings = {}
        model_info = {}
        
        for i, spec in enumerate(model_specs):
            model_name = f"{spec['type']}_{spec['id'].replace('/', '_')}"
            print(f"   Processing model {i+1}/{len(model_specs)}: {model_name}")
            
            try:
                # Load model
                model = ModelLoader.load_model(
                    spec["type"], 
                    spec["id"],
                    **spec.get("kwargs", {})
                )
                
                # Extract embeddings
                embeddings = model.extract_features(
                    images,
                    batch_size=self.config.batch_size,
                    **self.config.feature_extraction_kwargs
                )
                
                # Validate embeddings
                validation = validate_embeddings(embeddings, model_name)
                if not validation["valid"]:
                    print(f"     ⚠️ Validation issues: {validation['issues']}")
                
                model_embeddings[model_name] = embeddings
                model_info[model_name] = {
                    "spec": spec,
                    "embedding_shape": embeddings.shape,
                    "validation": validation
                }
                
                print(f"     ✅ Embeddings: {embeddings.shape}")
                
            except Exception as e:
                print(f"     ❌ Failed: {e}")
                continue
        
        if len(model_embeddings) < 2:
            raise RuntimeError("Need at least 2 successful models for comparison")
        
        # Compare embeddings
        if reference_model and reference_model in model_embeddings:
            # Compare all against reference
            comparison_results = self.embedding_comparison.compare_multiple(
                model_embeddings, 
                reference_key=reference_model
            )
        else:
            # All pairwise comparisons
            comparison_results = self.embedding_comparison.compare_multiple(
                model_embeddings
            )
        
        # Analyze model performance
        performance_analysis = self._analyze_model_performance(
            model_embeddings, comparison_results
        )
        
        results = {
            "workflow": "vision_model_comparison",
            "model_info": model_info,
            "comparison_results": comparison_results,
            "performance_analysis": performance_analysis,
            "config": self.config
        }
        
        print(f"🏆 Best performing model: {performance_analysis['best_model']}")
        
        return results
    
    def compare_model_scaling(self,
                            model_type: str,
                            model_sizes: List[str],
                            images: List,
                            size_order: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare models of different sizes to analyze scaling effects.
        
        Args:
            model_type: Type of models (e.g., "vit", "dinov2")
            model_sizes: List of model identifiers in size order
            images: List of PIL Images to process
            size_order: Explicit ordering if not implicit in model_sizes
            
        Returns:
            Scaling comparison results
        """
        print(f"📈 Analyzing {model_type} scaling across {len(model_sizes)} sizes")
        
        # Load models and extract embeddings
        model_embeddings = {}
        model_info = {}
        
        for model_id in model_sizes:
            model_name = f"{model_type}_{model_id.replace('/', '_')}"
            print(f"   Processing: {model_name}")
            
            try:
                # Load model
                model = ModelLoader.load_model(model_type, model_id)
                
                # Extract embeddings
                embeddings = model.extract_features(
                    images,
                    batch_size=self.config.batch_size,
                    **self.config.feature_extraction_kwargs
                )
                
                model_embeddings[model_name] = embeddings
                model_info[model_name] = {
                    "model_id": model_id,
                    "embedding_shape": embeddings.shape
                }
                
                print(f"     ✅ Embeddings: {embeddings.shape}")
                
            except Exception as e:
                print(f"     ❌ Failed: {e}")
                continue
        
        # Analyze scaling
        if size_order is None:
            size_order = [f"{model_type}_{mid.replace('/', '_')}" for mid in model_sizes]
        
        scaling_analysis = self.embedding_comparison.analyze_scaling_effects(
            model_embeddings, size_order
        )
        
        results = {
            "workflow": "model_scaling_comparison",
            "model_type": model_type,
            "model_info": model_info,
            "scaling_analysis": scaling_analysis,
            "config": self.config
        }
        
        print(f"📊 Scaling trend: {'✅ Monotonic' if scaling_analysis['is_monotonic_improving'] else '⚠️ Non-monotonic'}")
        
        return results
    
    def compare_cross_modal_alignment(self,
                                    model_specs: List[Dict[str, str]],
                                    images_a: List,
                                    images_b: List,
                                    data_types: Tuple[str, str] = ("Images_A", "Images_B")) -> Dict[str, Any]:
        """
        Compare how well different models align two data modalities.
        
        Args:
            model_specs: List of model specifications
            images_a: First set of images
            images_b: Second set of images 
            data_types: Names for the two data types
            
        Returns:
            Cross-modal alignment comparison
        """
        print(f"🔗 Comparing cross-modal alignment across {len(model_specs)} models")
        
        alignment_scores = {}
        model_info = {}
        
        for spec in model_specs:
            model_name = f"{spec['type']}_{spec['id'].replace('/', '_')}"
            print(f"   Processing: {model_name}")
            
            try:
                # Load model
                model = ModelLoader.load_model(
                    spec["type"], 
                    spec["id"],
                    **spec.get("kwargs", {})
                )
                
                # Extract embeddings for both modalities
                embeddings_a = model.extract_features(
                    images_a,
                    batch_size=self.config.batch_size,
                    **self.config.feature_extraction_kwargs
                )
                
                embeddings_b = model.extract_features(
                    images_b,
                    batch_size=self.config.batch_size,
                    **self.config.feature_extraction_kwargs
                )
                
                # Compute cross-modal alignment
                comparison = self.embedding_comparison.compare_embeddings(
                    embeddings_a, embeddings_b, names=data_types
                )
                
                alignment_scores[model_name] = comparison["best_knn_score"]
                model_info[model_name] = {
                    "spec": spec,
                    "comparison": comparison
                }
                
                print(f"     ✅ Alignment score: {alignment_scores[model_name]:.4f}")
                
            except Exception as e:
                print(f"     ❌ Failed: {e}")
                continue
        
        # Rank models by alignment
        if alignment_scores:
            best_model = max(alignment_scores.keys(), key=lambda k: alignment_scores[k])
            
            results = {
                "workflow": "cross_modal_alignment_comparison",
                "data_types": data_types,
                "alignment_scores": alignment_scores,
                "model_info": model_info,
                "best_model": best_model,
                "best_score": alignment_scores[best_model],
                "ranking": sorted(alignment_scores.items(), key=lambda x: x[1], reverse=True),
                "config": self.config
            }
            
            print(f"🏆 Best alignment: {best_model} ({results['best_score']:.4f})")
            
        else:
            results = {"error": "No models completed successfully"}
        
        return results
    
    def _analyze_model_performance(self,
                                 model_embeddings: Dict[str, np.ndarray],
                                 comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall model performance from comparison results."""
        
        # Extract scores for each model
        model_scores = {}
        
        if "pairwise_comparisons" in comparison_results:
            for comparison_name, comparison in comparison_results["pairwise_comparisons"].items():
                # Parse comparison name to get model names
                if "_vs_" in comparison_name:
                    model_a, model_b = comparison_name.split("_vs_", 1)
                    
                    # Add scores for both models
                    score = comparison["best_knn_score"]
                    
                    if model_a not in model_scores:
                        model_scores[model_a] = []
                    if model_b not in model_scores:
                        model_scores[model_b] = []
                    
                    model_scores[model_a].append(score)
                    model_scores[model_b].append(score)
        
        # Compute average scores
        avg_scores = {}
        for model, scores in model_scores.items():
            avg_scores[model] = np.mean(scores) if scores else 0.0
        
        # Find best model
        best_model = max(avg_scores.keys(), key=lambda k: avg_scores[k]) if avg_scores else None
        
        # Compute embedding quality metrics
        from ..utils.metrics import compare_embedding_quality
        quality_comparison = compare_embedding_quality(model_embeddings)
        
        analysis = {
            "average_scores": avg_scores,
            "best_model": best_model,
            "best_average_score": avg_scores.get(best_model, 0.0) if best_model else 0.0,
            "score_statistics": {
                "mean": np.mean(list(avg_scores.values())) if avg_scores else 0.0,
                "std": np.std(list(avg_scores.values())) if avg_scores else 0.0,
                "min": np.min(list(avg_scores.values())) if avg_scores else 0.0,
                "max": np.max(list(avg_scores.values())) if avg_scores else 0.0
            },
            "embedding_quality": quality_comparison
        }
        
        return analysis


def compare_vision_models(model_specs: List[Dict[str, str]],
                         images: List,
                         k_values: List[int] = [5, 10, 20, 50],
                         **kwargs) -> Dict[str, Any]:
    """
    Convenience function for comparing vision models.
    
    Args:
        model_specs: List of model specifications
        images: List of PIL Images
        k_values: List of k values for comparison
        **kwargs: Additional configuration arguments
        
    Returns:
        Model comparison results
    """
    config = ModelComparisonConfig(k_values=k_values, **kwargs)
    comparison = ModelComparison(config)
    return comparison.compare_vision_models(model_specs, images)


def compare_model_scaling(model_type: str,
                         model_sizes: List[str],
                         images: List,
                         k_values: List[int] = [5, 10, 20, 50],
                         **kwargs) -> Dict[str, Any]:
    """
    Convenience function for analyzing model scaling.
    
    Args:
        model_type: Type of models to compare
        model_sizes: List of model identifiers in size order
        images: List of PIL Images
        k_values: List of k values for comparison
        **kwargs: Additional configuration arguments
        
    Returns:
        Scaling analysis results
    """
    config = ModelComparisonConfig(k_values=k_values, **kwargs)
    comparison = ModelComparison(config)
    return comparison.compare_model_scaling(model_type, model_sizes, images)