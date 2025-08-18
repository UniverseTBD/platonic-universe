#!/usr/bin/env python3
"""
Example of using custom HuggingFace datasets with Platonic Universe.
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
from platonic_universe.data import load_custom_dataset


def main():
    """Demonstrate loading custom HuggingFace datasets."""
    
    print("🌌 Platonic Universe - Custom Dataset Example")
    print("=" * 60)
    
    # Setup
    set_seed(42)
    cache_dir = setup_cache()
    runner = WorkflowRunner(cache_dir=cache_dir)
    
    print("\n📋 Example 1: Using built-in datasets")
    print("-" * 40)
    
    # Built-in datasets
    available_datasets = [
        "hsc_sdss",     # HSC images + SDSS spectra
        "hsc_jwst",     # HSC images + JWST images  
        "desi_hsc",     # HSC images + DESI images/spectra
    ]
    
    print("Available built-in datasets:")
    for dataset in available_datasets:
        print(f"   - {dataset}")
    
    print("\n📋 Example 2: Loading custom datasets")
    print("-" * 40)
    
    # Example of loading a custom dataset
    custom_repo_examples = [
        "user/custom_astro_dataset",
        "org/multimodal_galaxy_data", 
        "astro/cross_survey_matches"
    ]
    
    print("To load a custom dataset, use:")
    print("```python")
    print("runner = WorkflowRunner()")
    print("dataset_key = runner.load_dataset(")
    print("    dataset_name='custom',")
    print("    repo_id='your_username/your_dataset',")
    print("    max_samples=1000,")
    print("    image_field_a='galaxy_images',")
    print("    image_field_b='spectra',")
    print("    flux_field_a='image_flux',")
    print("    flux_field_b='spectrum_flux'")
    print(")")
    print("```")
    
    print("\n📋 Example 3: Dataset field mapping")
    print("-" * 40)
    
    print("Field mapping options for custom datasets:")
    print("- image_field_a: Primary image field name (default: 'image_a')")
    print("- image_field_b: Secondary image field name (default: 'image_b')")
    print("- flux_field_a: Primary flux field fallback (default: 'flux_a')")
    print("- flux_field_b: Secondary flux field fallback (default: 'flux_b')")
    
    print("\nThe loader will:")
    print("1. Try to find data in image_field_a/image_field_b")
    print("2. Fallback to flux_field_a/flux_field_b if images not found")
    print("3. Handle nested flux data (e.g., {'flux': array})")
    print("4. Convert all data to PIL Images using flux_to_pil()")
    
    print("\n📋 Example 4: Complete workflow with custom data")
    print("-" * 40)
    
    print("```python")
    print("# Load custom dataset")
    print("dataset_key = runner.load_dataset(")
    print("    'custom', repo_id='astro/galaxy_spectra_v2'")
    print(")")
    print("")
    print("# Load models")
    print("model_a = runner.load_model('vit', 'google/vit-large-patch16-224')")
    print("model_b = runner.load_model('dinov2', 'facebook/dinov2-base')")
    print("")
    print("# Run comparison")
    print("results = runner.run_embedding_comparison(")
    print("    model_a={'type': 'vit', 'id': 'google/vit-base-patch16-224'},")
    print("    model_b={'type': 'dinov2', 'id': 'facebook/dinov2-base'},")
    print("    dataset={'dataset_name': 'custom', 'repo_id': 'astro/your_data'}")
    print(")")
    print("")
    print("print(f'Cross-modal score: {results[\"best_score\"]:.4f}')")
    print("```")
    
    print("\n📋 Example 5: Direct data loading")
    print("-" * 40)
    
    print("For direct data loading without WorkflowRunner:")
    print("```python")
    print("from platonic_universe.data import load_custom_dataset")
    print("")
    print("images_a, images_b = load_custom_dataset(")
    print("    repo_id='username/dataset_name',")
    print("    max_samples=500,")
    print("    image_field_a='hst_images',")
    print("    image_field_b='ground_spectra'")
    print(")")
    print("```")
    
    print("\n💡 Tips for custom datasets:")
    print("- Ensure your HF dataset has paired data in each row")
    print("- Flux data should be numpy-compatible arrays") 
    print("- Images can be 2D/3D arrays or nested dicts with 'flux' key")
    print("- Use descriptive field names and document your dataset structure")
    print("- Test with max_samples first to verify field mappings")
    
    print("\n✅ Custom dataset example complete!")
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Custom dataset examples completed!")
    else:
        print("\n❌ Custom dataset examples failed!")
        sys.exit(1)