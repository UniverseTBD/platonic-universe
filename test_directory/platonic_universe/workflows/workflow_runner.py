"""
Main workflow runner for Platonic Universe.
"""

from typing import Dict, Any, Optional, List, Union
import time
import numpy as np
from pathlib import Path

from ..models import ModelLoader
from ..data import DatasetLoader
from ..cache import pick_cache_root, get_cache_info
from ..utils import compute_mknn_prh, KNNAnalyzer, set_seed, get_device_info, validate_embeddings


class WorkflowRunner:
    """Main workflow runner that orchestrates complete analysis pipelines."""
    
    def __init__(self, 
                 cache_dir: Optional[Union[str, Path]] = None,
                 random_seed: int = 42,
                 device: Optional[str] = None):
        """
        Initialize workflow runner.
        
        Args:
            cache_dir: Cache directory for models and datasets
            random_seed: Random seed for reproducibility
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        # Set random seed for reproducibility
        set_seed(random_seed)
        self.random_seed = random_seed
        
        # Setup cache
        if cache_dir is None:
            cache_dir = pick_cache_root()
        self.cache_dir = cache_dir
        
        # Initialize components
        self.data_loader = DatasetLoader(cache_dir)
        self.device = device
        
        # Track loaded models and data
        self.loaded_models = {}
        self.loaded_datasets = {}
        
        print(f"🚀 WorkflowRunner initialized")
        print(f"   Cache directory: {self.cache_dir}")
        print(f"   Random seed: {self.random_seed}")
        print(f"   Device: {self.device or 'auto'}")
    
    def load_model(self, model_type: str, model_id: str, **kwargs) -> str:
        """
        Load a model and return a reference key.
        
        Args:
            model_type: Type of model ('dinov2', 'vit', 'ijepa')
            model_id: Model identifier
            **kwargs: Additional model loading arguments
            
        Returns:
            Reference key for the loaded model
        """
        model_key = f"{model_type}_{model_id.replace('/', '_')}"
        
        if model_key not in self.loaded_models:
            print(f"🔧 Loading {model_type} model: {model_id}")
            start_time = time.time()
            
            model = ModelLoader.load_model(
                model_type, 
                model_id, 
                device=self.device,
                cache_dir=str(self.cache_dir / "models") if hasattr(self.cache_dir, '__truediv__') else None,
                **kwargs
            )
            
            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time:.1f}s")
            
            self.loaded_models[model_key] = {
                "model": model,
                "model_type": model_type,
                "model_id": model_id,
                "load_time": load_time,
                "kwargs": kwargs
            }
        
        return model_key
    
    def load_dataset(self, dataset_name: str, max_samples: int = 0, 
                    repo_id: Optional[str] = None, **kwargs) -> str:
        """
        Load a dataset and return a reference key.
        
        Args:
            dataset_name: Name of dataset ('hsc_sdss', 'hsc_jwst', 'desi_hsc') or 'custom'
            max_samples: Maximum samples to load (0 = all)
            repo_id: HuggingFace repository ID (required for 'custom' datasets)
            **kwargs: Additional dataset loading arguments
            
        Returns:
            Reference key for the loaded dataset
        """
        if dataset_name == "custom" and repo_id:
            dataset_key = f"custom_{repo_id.replace('/', '_')}_{max_samples}" if max_samples > 0 else f"custom_{repo_id.replace('/', '_')}"
        else:
            dataset_key = f"{dataset_name}_{max_samples}" if max_samples > 0 else dataset_name
        
        if dataset_key not in self.loaded_datasets:
            print(f"📥 Loading dataset: {dataset_name}")
            start_time = time.time()
            
            if dataset_name == "hsc_sdss":
                from ..data import load_hsc_sdss
                images_a, images_b = load_hsc_sdss(
                    max_samples=max_samples,
                    cache_dir=str(self.cache_dir / "datasets") if hasattr(self.cache_dir, '__truediv__') else None,
                    **kwargs
                )
                data_types = ("HSC_images", "SDSS_spectra")
                
            elif dataset_name == "hsc_jwst":
                from ..data import load_hsc_jwst
                images_a, images_b = load_hsc_jwst(
                    max_samples=max_samples,
                    cache_dir=str(self.cache_dir / "datasets") if hasattr(self.cache_dir, '__truediv__') else None,
                    **kwargs
                )
                data_types = ("HSC_images", "JWST_images")
                
            elif dataset_name == "desi_hsc":
                from ..data import load_desi_hsc
                images_a, images_b = load_desi_hsc(
                    max_samples=max_samples,
                    cache_dir=str(self.cache_dir / "datasets") if hasattr(self.cache_dir, '__truediv__') else None,
                    **kwargs
                )
                data_types = ("HSC_images", "DESI_images")
                
            elif dataset_name == "custom":
                if not repo_id:
                    raise ValueError("repo_id is required for custom datasets")
                from ..data import load_custom_dataset
                images_a, images_b = load_custom_dataset(
                    repo_id=repo_id,
                    max_samples=max_samples,
                    cache_dir=str(self.cache_dir / "datasets") if hasattr(self.cache_dir, '__truediv__') else None,
                    **kwargs
                )
                data_types = ("Custom_A", "Custom_B")
                
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}. Use 'hsc_sdss', 'hsc_jwst', 'desi_hsc', or 'custom' with repo_id")
            
            load_time = time.time() - start_time
            print(f"✅ Dataset loaded in {load_time:.1f}s: {len(images_a)} samples")
            
            self.loaded_datasets[dataset_key] = {
                "images_a": images_a,
                "images_b": images_b,
                "data_types": data_types,
                "dataset_name": dataset_name,
                "repo_id": repo_id,
                "n_samples": len(images_a),
                "load_time": load_time,
                "kwargs": kwargs
            }
        
        return dataset_key
    
    def extract_embeddings(self, model_key: str, dataset_key: str, 
                          data_type: str = "both", **kwargs) -> Dict[str, np.ndarray]:
        """
        Extract embeddings from loaded model and dataset.
        
        Args:
            model_key: Reference to loaded model
            dataset_key: Reference to loaded dataset
            data_type: Which data to process ('images_a', 'images_b', or 'both')
            **kwargs: Additional extraction arguments
            
        Returns:
            Dictionary with embeddings
        """
        if model_key not in self.loaded_models:
            raise ValueError(f"Model '{model_key}' not loaded")
        if dataset_key not in self.loaded_datasets:
            raise ValueError(f"Dataset '{dataset_key}' not loaded")
        
        model_info = self.loaded_models[model_key]
        dataset_info = self.loaded_datasets[dataset_key]
        model = model_info["model"]
        
        print(f"🔄 Extracting embeddings with {model_key}")
        
        embeddings = {}
        
        if data_type in ("images_a", "both"):
            print(f"   Processing {dataset_info['data_types'][0]}...")
            start_time = time.time()
            
            emb_a = model.extract_features(dataset_info["images_a"], **kwargs)
            validation = validate_embeddings(emb_a, dataset_info['data_types'][0])
            
            extract_time = time.time() - start_time
            print(f"   ✅ {dataset_info['data_types'][0]} embeddings: {emb_a.shape} in {extract_time:.1f}s")
            
            if not validation["valid"]:
                print(f"   ⚠️ Embedding validation issues: {validation['issues']}")
            
            embeddings["embeddings_a"] = emb_a
        
        if data_type in ("images_b", "both"):
            print(f"   Processing {dataset_info['data_types'][1]}...")
            start_time = time.time()
            
            emb_b = model.extract_features(dataset_info["images_b"], **kwargs)
            validation = validate_embeddings(emb_b, dataset_info['data_types'][1])
            
            extract_time = time.time() - start_time
            print(f"   ✅ {dataset_info['data_types'][1]} embeddings: {emb_b.shape} in {extract_time:.1f}s")
            
            if not validation["valid"]:
                print(f"   ⚠️ Embedding validation issues: {validation['issues']}")
            
            embeddings["embeddings_b"] = emb_b
        
        return embeddings
    
    def run_embedding_comparison(self, 
                               model_a: Union[str, Dict[str, str]],
                               model_b: Union[str, Dict[str, str]],
                               dataset: Union[str, Dict[str, Any]],
                               k_values: List[int] = [5, 10, 20, 50],
                               max_samples: int = 0,
                               **kwargs) -> Dict[str, Any]:
        """
        Run complete embedding comparison workflow.
        
        Args:
            model_a: Model specification for first modality
            model_b: Model specification for second modality  
            dataset: Dataset specification
            k_values: List of k values for kNN analysis
            max_samples: Maximum samples to use
            **kwargs: Additional arguments
            
        Returns:
            Complete analysis results
        """
        start_time = time.time()
        print("🎯 Starting embedding comparison workflow")
        
        # Load models
        if isinstance(model_a, str):
            model_a_key = model_a if model_a in self.loaded_models else None
        else:
            model_a_key = self.load_model(**model_a)
        
        if isinstance(model_b, str):
            model_b_key = model_b if model_b in self.loaded_models else None
        else:
            model_b_key = self.load_model(**model_b)
        
        # Load dataset
        if isinstance(dataset, str):
            dataset_key = dataset if dataset in self.loaded_datasets else None
        else:
            dataset_key = self.load_dataset(max_samples=max_samples, **dataset)
        
        # Extract embeddings
        embeddings_a = self.extract_embeddings(model_a_key, dataset_key, "images_a", **kwargs)
        embeddings_b = self.extract_embeddings(model_b_key, dataset_key, "images_b", **kwargs)
        
        # Align sample counts
        n_samples = min(len(embeddings_a["embeddings_a"]), len(embeddings_b["embeddings_b"]))
        emb_a = embeddings_a["embeddings_a"][:n_samples]
        emb_b = embeddings_b["embeddings_b"][:n_samples]
        
        print(f"📊 Running kNN analysis on {n_samples} aligned samples")
        
        # Run kNN analysis
        analyzer = KNNAnalyzer(random_seed=self.random_seed)
        knn_results = {}
        
        for k in k_values:
            score = analyzer.compute_mutual_knn(emb_a, emb_b, k=k)
            knn_results[k] = score
            print(f"   k={k}: {score:.4f}")
        
        total_time = time.time() - start_time
        
        # Compile results
        results = {
            "workflow": "embedding_comparison",
            "models": {
                "model_a": self.loaded_models[model_a_key],
                "model_b": self.loaded_models[model_b_key]
            },
            "dataset": self.loaded_datasets[dataset_key],
            "embeddings": {
                "embeddings_a_shape": emb_a.shape,
                "embeddings_b_shape": emb_b.shape,
                "n_aligned_samples": n_samples
            },
            "knn_scores": knn_results,
            "best_k": max(knn_results.keys(), key=lambda k: knn_results[k]),
            "best_score": max(knn_results.values()),
            "timing": {
                "total_time": total_time,
                "start_time": start_time
            },
            "config": {
                "k_values": k_values,
                "max_samples": max_samples,
                "random_seed": self.random_seed
            }
        }
        
        print(f"🎉 Comparison complete in {total_time:.1f}s")
        print(f"   Best score: {results['best_score']:.4f} at k={results['best_k']}")
        
        return results
    
    def run_cross_modal_analysis(self,
                                model_type: str,
                                model_ids: List[str], 
                                dataset_name: str,
                                k: int = 10,
                                max_samples: int = 0,
                                **kwargs) -> Dict[str, Any]:
        """
        Run cross-modal analysis across multiple models of the same type.
        
        Args:
            model_type: Type of models to compare
            model_ids: List of model identifiers
            dataset_name: Dataset to use
            k: k value for kNN analysis
            max_samples: Maximum samples to use
            **kwargs: Additional arguments
            
        Returns:
            Cross-modal analysis results
        """
        print(f"🔬 Starting cross-modal analysis")
        print(f"   Model type: {model_type}")
        print(f"   Models: {model_ids}")
        print(f"   Dataset: {dataset_name}")
        
        # Load dataset
        dataset_key = self.load_dataset(dataset_name, max_samples=max_samples)
        dataset_info = self.loaded_datasets[dataset_key]
        
        # Load all models and extract embeddings
        model_results = {}
        
        for model_id in model_ids:
            model_key = self.load_model(model_type, model_id, **kwargs)
            
            # Extract embeddings for both data types
            embeddings = self.extract_embeddings(model_key, dataset_key, "both", **kwargs)
            
            # Compute cross-modal score
            analyzer = KNNAnalyzer(random_seed=self.random_seed)
            score = analyzer.compute_mutual_knn(
                embeddings["embeddings_a"],
                embeddings["embeddings_b"],
                k=k
            )
            
            model_results[model_id] = {
                "model_key": model_key,
                "embeddings_a_shape": embeddings["embeddings_a"].shape,
                "embeddings_b_shape": embeddings["embeddings_b"].shape,
                "cross_modal_score": score
            }
            
            print(f"   {model_id}: {score:.4f}")
        
        # Find best performing model
        best_model = max(model_results.keys(), key=lambda m: model_results[m]["cross_modal_score"])
        
        results = {
            "workflow": "cross_modal_analysis",
            "model_type": model_type,
            "dataset": dataset_info,
            "models": model_results,
            "best_model": best_model,
            "best_score": model_results[best_model]["cross_modal_score"],
            "config": {
                "k": k,
                "max_samples": max_samples,
                "random_seed": self.random_seed
            }
        }
        
        print(f"🏆 Best model: {best_model} (score: {results['best_score']:.4f})")
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the workflow runner."""
        return {
            "cache_info": get_cache_info(),
            "device_info": get_device_info(),
            "loaded_models": {k: {
                "model_type": v["model_type"],
                "model_id": v["model_id"],
                "load_time": v["load_time"]
            } for k, v in self.loaded_models.items()},
            "loaded_datasets": {k: {
                "dataset_name": v["dataset_name"], 
                "n_samples": v["n_samples"],
                "load_time": v["load_time"]
            } for k, v in self.loaded_datasets.items()},
            "config": {
                "cache_dir": str(self.cache_dir),
                "random_seed": self.random_seed,
                "device": self.device
            }
        }
    
    def clear_models(self) -> None:
        """Clear all loaded models to free memory."""
        print("🧹 Clearing loaded models")
        self.loaded_models.clear()
        
        # Clear GPU memory if available
        try:
            from ..utils import clear_gpu_memory
            clear_gpu_memory()
        except Exception:
            pass
    
    def clear_datasets(self) -> None:
        """Clear all loaded datasets to free memory."""
        print("🧹 Clearing loaded datasets")
        self.loaded_datasets.clear()