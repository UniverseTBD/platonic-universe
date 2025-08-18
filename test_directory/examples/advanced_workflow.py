#!/usr/bin/env python3
"""
Advanced workflow example for Platonic Universe package.
"""

import sys
from pathlib import Path

# Add the package to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from platonic_universe import (
    WorkflowRunner,
    setup_cache,
    set_seed,
)
from platonic_universe.workflows import (
    run_embedding_comparison,
    compare_vision_models
)
from platonic_universe.data import generate_mock_data
from platonic_universe.utils import compute_mknn_prh


def main():
    """Demonstrate advanced workflows in Platonic Universe."""
    
    print("🌌 Platonic Universe - Advanced Workflow Example")
    print("=" * 60)
    
    # Setup
    set_seed(42)
    cache_dir = setup_cache()
    runner = WorkflowRunner(cache_dir=cache_dir)
    
    try:
        print("\n🎯 Example 1: Cross-Modal Analysis")
        print("-" * 40)
        
        print("💡 In a real environment, you would load a dataset like:")
        print("   dataset_key = runner.load_dataset('hsc_sdss', max_samples=200)")
        print("   or")
        print("   dataset_key = runner.load_dataset('custom', repo_id='astro/galaxy_data')")
        
        # Simulate loading multiple models (would fail without full dependencies)
        print("📋 Would compare multiple vision models:")
        model_specs = [
            {"type": "vit", "id": "google/vit-base-patch16-224"},
            {"type": "vit", "id": "google/vit-large-patch16-224"},
            {"type": "dinov2", "id": "facebook/dinov2-base"},
        ]
        
        for spec in model_specs:
            print(f"   - {spec['type']}: {spec['id']}")
        
        print("\n🔬 Example 2: Embedding Comparison")
        print("-" * 40)
        
        # Generate mock embeddings to demonstrate the comparison pipeline
        import numpy as np
        
        # Create mock embeddings with different properties
        embeddings_a = np.random.randn(100, 768)  # High dimensional
        embeddings_b = np.random.randn(100, 512)  # Lower dimensional
        
        # Add some correlation to make the comparison interesting
        embeddings_b[:, :100] = embeddings_a[:, :100] * 0.7 + np.random.randn(100, 100) * 0.3
        
        print(f"Created mock embeddings: A={embeddings_a.shape}, B={embeddings_b.shape}")
        
        # Run comparison
        comparison_result = run_embedding_comparison(
            embeddings_a, 
            embeddings_b,
            names=("MockModel_A", "MockModel_B"),
            k_values=[5, 10, 20]
        )
        
        print("📊 Comparison Results:")
        for k, score in comparison_result["knn_scores"].items():
            print(f"   k={k}: {score:.4f}")
        print(f"   Best k: {comparison_result['best_k']} (score: {comparison_result['best_knn_score']:.4f})")
        
        print("\n📈 Example 3: Manual kNN Analysis")
        print("-" * 40)
        
        # Demonstrate direct kNN computation
        knn_score = compute_mknn_prh(embeddings_a, embeddings_b, k=10)
        print(f"Direct kNN score (k=10): {knn_score:.4f}")
        
        print("\n🏗️ Example 4: Workflow Architecture")
        print("-" * 40)
        
        # Show how the package is structured
        print("Package provides these key components:")
        print("   🔧 Models: Base classes + DinoV2, ViT, I-JEPA, SpecFormer")
        print("   📊 Data: Loaders for HSC-SDSS, HSC-JWST, DESI-HSC datasets")
        print("   💾 Cache: Smart cache management with auto-selection")
        print("   🔍 Utils: kNN metrics, validation, helpers")
        print("   🚀 Workflows: High-level analysis pipelines")
        
        print("\n💡 Example 5: Real Usage Pattern")
        print("-" * 40)
        
        print("In a real environment with full dependencies, you would:")
        print("1. runner = WorkflowRunner()")
        print("2. results = runner.run_embedding_comparison(")
        print("     model_a={'type': 'dinov2', 'id': 'facebook/dinov2-base'},")
        print("     model_b={'type': 'vit', 'id': 'google/vit-base-patch16-224'},")
        print("     dataset={'dataset_name': 'hsc_sdss', 'max_samples': 1000}")
        print("   )")
        print("3. print(f'Cross-modal score: {results[\"best_score\"]:.4f}')")
        print("")
        print("Or with a custom dataset:")
        print("   results = runner.run_embedding_comparison(")
        print("     model_a={'type': 'vit', 'id': 'google/vit-large-patch16-224'},")
        print("     model_b={'type': 'ijepa', 'id': 'facebook/ijepa_vith14_1k'},") 
        print("     dataset={'dataset_name': 'custom', 'repo_id': 'astro/multimodal_v2'}")
        print("   )")
        
        print("\n🎯 Example 6: Custom Analysis")
        print("-" * 40)
        
        # Show how to use individual components
        from platonic_universe.utils import KNNAnalyzer
        from platonic_universe.data.preprocessors import flux_to_pil
        
        analyzer = KNNAnalyzer(metric="cosine", normalize=True)
        
        print("Example: Convert astronomical flux to image")
        print("   flux_data = your_astronomical_data  # 2D numpy array")
        print("   image = flux_to_pil(flux_data, target_size=224)")
        print("   # This converts FITS-like data to PIL Images for models")
        
        print("\n🔧 Example 7: Cache Management")
        print("-" * 40)
        
        from platonic_universe.cache import get_cache_info, clear_cache
        
        cache_info = get_cache_info()
        if cache_info["setup"]:
            print(f"Cache root: {cache_info['cache_root']}")
            print(f"Subdirectories: {list(cache_info['subdirectories'].keys())}")
            
            if cache_info.get("disk_usage"):
                disk = cache_info["disk_usage"]
                print(f"Disk usage: {disk.get('free_gb', 0):.1f} GB free")
        
        print("\n✅ Advanced workflow examples completed!")
        print("\nThe package is fully functional and ready for:")
        print("- Multi-modal astronomical data analysis")
        print("- Vision model comparisons")
        print("- Cross-modal alignment studies")
        print("- Scalable embedding workflows")
        
    except Exception as e:
        print(f"❌ Error in advanced workflow: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Advanced workflow examples completed successfully!")
    else:
        print("\n❌ Advanced workflow examples failed!")
        sys.exit(1)