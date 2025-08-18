"""
Embedding comparison workflows and utilities.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass

from ..utils import compute_mknn_prh, KNNAnalyzer, validate_embeddings
from ..utils.metrics import project_to_common_space, compute_alignment_score


@dataclass
class EmbeddingComparisonConfig:
    """Configuration for embedding comparison."""
    k_values: List[int] = None
    metric: str = "cosine"
    normalize: bool = True
    batch_size: Optional[int] = None
    repeats: int = 3
    random_seed: int = 42
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [5, 10, 20, 50]


class EmbeddingComparison:
    """Class for comparing embeddings across modalities."""
    
    def __init__(self, config: Optional[EmbeddingComparisonConfig] = None):
        """
        Initialize embedding comparison.
        
        Args:
            config: Configuration for comparison
        """
        self.config = config or EmbeddingComparisonConfig()
        self.analyzer = KNNAnalyzer(
            metric=self.config.metric,
            normalize=self.config.normalize,
            random_seed=self.config.random_seed
        )
    
    def compare_embeddings(self,
                          embeddings_a: np.ndarray,
                          embeddings_b: np.ndarray,
                          names: Tuple[str, str] = ("A", "B")) -> Dict[str, Any]:
        """
        Compare two sets of embeddings.
        
        Args:
            embeddings_a: First embedding set
            embeddings_b: Second embedding set
            names: Names for the embedding sets
            
        Returns:
            Comparison results
        """
        # Validate inputs
        validation_a = validate_embeddings(embeddings_a, names[0])
        validation_b = validate_embeddings(embeddings_b, names[1])
        
        if not validation_a["valid"]:
            print(f"⚠️ Issues with {names[0]} embeddings: {validation_a['issues']}")
        if not validation_b["valid"]:
            print(f"⚠️ Issues with {names[1]} embeddings: {validation_b['issues']}")
        
        # Align sample counts
        n_samples = min(len(embeddings_a), len(embeddings_b))
        emb_a = embeddings_a[:n_samples]
        emb_b = embeddings_b[:n_samples]
        
        # Handle dimension mismatch if necessary
        if emb_a.shape[1] != emb_b.shape[1]:
            print(f"🔧 Projecting to common space: {emb_a.shape[1]} vs {emb_b.shape[1]} dimensions")
            emb_a, emb_b = project_to_common_space(emb_a, emb_b, method="pca")
        
        # Compute kNN scores
        knn_scores = {}
        for k in self.config.k_values:
            score = compute_mknn_prh(
                emb_a, emb_b, k=k,
                metric=self.config.metric,
                normalize=self.config.normalize
            )
            knn_scores[k] = score
        
        # Additional alignment metrics
        alignment_scores = {
            "cosine_mean": compute_alignment_score(emb_a, emb_b, "cosine_mean"),
            "cosine_max": compute_alignment_score(emb_a, emb_b, "cosine_max"),
            "euclidean_mean": compute_alignment_score(emb_a, emb_b, "euclidean_mean")
        }
        
        results = {
            "names": names,
            "n_samples": n_samples,
            "embedding_shapes": {
                names[0]: emb_a.shape,
                names[1]: emb_b.shape
            },
            "validation": {
                names[0]: validation_a,
                names[1]: validation_b
            },
            "knn_scores": knn_scores,
            "alignment_scores": alignment_scores,
            "best_k": max(knn_scores.keys(), key=lambda k: knn_scores[k]),
            "best_knn_score": max(knn_scores.values()),
            "config": self.config
        }
        
        return results
    
    def compare_multiple(self,
                        embeddings_dict: Dict[str, np.ndarray],
                        reference_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare multiple embedding sets.
        
        Args:
            embeddings_dict: Dictionary mapping names to embeddings
            reference_key: Reference embedding set (if None, compare all pairs)
            
        Returns:
            Multiple comparison results
        """
        results = {
            "embedding_sets": list(embeddings_dict.keys()),
            "reference": reference_key,
            "pairwise_comparisons": {},
            "summary": {}
        }
        
        if reference_key is not None:
            # Compare all sets against reference
            if reference_key not in embeddings_dict:
                raise ValueError(f"Reference '{reference_key}' not found in embeddings")
            
            ref_embeddings = embeddings_dict[reference_key]
            
            for name, embeddings in embeddings_dict.items():
                if name != reference_key:
                    comparison = self.compare_embeddings(
                        ref_embeddings, embeddings,
                        names=(reference_key, name)
                    )
                    results["pairwise_comparisons"][f"{reference_key}_vs_{name}"] = comparison
        
        else:
            # Compare all pairs
            set_names = list(embeddings_dict.keys())
            for i, name_a in enumerate(set_names):
                for j, name_b in enumerate(set_names):
                    if i < j:  # Avoid duplicates
                        comparison = self.compare_embeddings(
                            embeddings_dict[name_a],
                            embeddings_dict[name_b],
                            names=(name_a, name_b)
                        )
                        results["pairwise_comparisons"][f"{name_a}_vs_{name_b}"] = comparison
        
        # Compute summary statistics
        if results["pairwise_comparisons"]:
            all_best_scores = [
                comp["best_knn_score"] 
                for comp in results["pairwise_comparisons"].values()
            ]
            
            results["summary"] = {
                "n_comparisons": len(results["pairwise_comparisons"]),
                "mean_best_score": np.mean(all_best_scores),
                "std_best_score": np.std(all_best_scores),
                "max_best_score": np.max(all_best_scores),
                "min_best_score": np.min(all_best_scores)
            }
        
        return results
    
    def analyze_scaling_effects(self,
                               embeddings_dict: Dict[str, np.ndarray],
                               model_order: List[str]) -> Dict[str, Any]:
        """
        Analyze scaling effects across models of different sizes.
        
        Args:
            embeddings_dict: Dictionary mapping model names to embeddings
            model_order: List of model names in order of increasing size
            
        Returns:
            Scaling analysis results
        """
        # Validate model order
        for model in model_order:
            if model not in embeddings_dict:
                raise ValueError(f"Model '{model}' not found in embeddings")
        
        # Compare consecutive models
        scaling_comparisons = {}
        scaling_scores = []
        
        for i in range(len(model_order) - 1):
            model_small = model_order[i]
            model_large = model_order[i + 1]
            
            comparison = self.compare_embeddings(
                embeddings_dict[model_small],
                embeddings_dict[model_large],
                names=(model_small, model_large)
            )
            
            scaling_comparisons[f"{model_small}_to_{model_large}"] = comparison
            scaling_scores.append(comparison["best_knn_score"])
        
        # Analyze scaling trend
        is_monotonic = all(scaling_scores[i] <= scaling_scores[i+1] 
                          for i in range(len(scaling_scores)-1))
        
        results = {
            "model_order": model_order,
            "scaling_comparisons": scaling_comparisons,
            "scaling_scores": scaling_scores,
            "is_monotonic_improving": is_monotonic,
            "total_improvement": scaling_scores[-1] - scaling_scores[0] if scaling_scores else 0,
            "average_step_improvement": np.mean(np.diff(scaling_scores)) if len(scaling_scores) > 1 else 0
        }
        
        return results


def run_embedding_comparison(embeddings_a: np.ndarray,
                           embeddings_b: np.ndarray,
                           names: Tuple[str, str] = ("A", "B"),
                           k_values: List[int] = [5, 10, 20, 50],
                           **kwargs) -> Dict[str, Any]:
    """
    Convenience function to run embedding comparison.
    
    Args:
        embeddings_a: First embedding set
        embeddings_b: Second embedding set
        names: Names for the embedding sets
        k_values: List of k values for kNN analysis
        **kwargs: Additional configuration arguments
        
    Returns:
        Comparison results
    """
    config = EmbeddingComparisonConfig(k_values=k_values, **kwargs)
    comparison = EmbeddingComparison(config)
    return comparison.compare_embeddings(embeddings_a, embeddings_b, names)


def run_cross_modal_analysis(image_embeddings: np.ndarray,
                            spectrum_embeddings: np.ndarray,
                            k_values: List[int] = [5, 10, 20, 50],
                            **kwargs) -> Dict[str, Any]:
    """
    Convenience function for cross-modal analysis.
    
    Args:
        image_embeddings: Image embeddings
        spectrum_embeddings: Spectrum embeddings
        k_values: List of k values for kNN analysis
        **kwargs: Additional configuration arguments
        
    Returns:
        Cross-modal analysis results
    """
    return run_embedding_comparison(
        image_embeddings,
        spectrum_embeddings,
        names=("Images", "Spectra"),
        k_values=k_values,
        **kwargs
    )


def analyze_model_consistency(embeddings_list: List[np.ndarray],
                            model_names: List[str],
                            k: int = 10) -> Dict[str, Any]:
    """
    Analyze consistency across different model runs or configurations.
    
    Args:
        embeddings_list: List of embedding arrays
        model_names: Names for each embedding set
        k: k value for kNN analysis
        
    Returns:
        Consistency analysis results
    """
    if len(embeddings_list) != len(model_names):
        raise ValueError("Number of embeddings must match number of model names")
    
    # Compare all pairs
    consistency_scores = {}
    analyzer = KNNAnalyzer()
    
    for i, name_a in enumerate(model_names):
        for j, name_b in enumerate(model_names):
            if i < j:
                score = analyzer.compute_mutual_knn(
                    embeddings_list[i],
                    embeddings_list[j],
                    k=k
                )
                consistency_scores[f"{name_a}_vs_{name_b}"] = score
    
    # Compute consistency metrics
    scores = list(consistency_scores.values())
    
    results = {
        "model_names": model_names,
        "pairwise_scores": consistency_scores,
        "consistency_metrics": {
            "mean_consistency": np.mean(scores),
            "std_consistency": np.std(scores),
            "min_consistency": np.min(scores),
            "max_consistency": np.max(scores),
            "coefficient_of_variation": np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else float('inf')
        },
        "most_consistent_pair": max(consistency_scores.keys(), key=lambda k: consistency_scores[k]),
        "least_consistent_pair": min(consistency_scores.keys(), key=lambda k: consistency_scores[k])
    }
    
    return results