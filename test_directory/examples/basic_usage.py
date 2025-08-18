#!/usr/bin/env python3
"""
Basic usage example for Platonic Universe package.
"""

import sys
from pathlib import Path

# Add the package to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from platonic_universe import (
    setup_cache,
    ModelLoader, 
    DatasetLoader,
    WorkflowRunner,
    set_seed,
    get_device_info
)


def main():
    """Demonstrate basic usage of Platonic Universe."""
    
    print("🌌 Platonic Universe - Basic Usage Example")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Show device information
    device_info = get_device_info()
    print(f"💻 Device info: {device_info['current_device']}")
    if device_info.get('cuda_available'):
        print(f"   🎮 CUDA devices: {device_info['cuda_device_count']}")
    
    # Setup cache (this will auto-select a suitable location)
    cache_dir = setup_cache()
    print(f"📁 Cache directory: {cache_dir}")
    
    try:
        # Initialize workflow runner
        runner = WorkflowRunner(cache_dir=cache_dir)
        
        # Show available datasets
        print("\n📥 Available datasets:")
        print("   - hsc_sdss: HSC images + SDSS spectra")
        print("   - hsc_jwst: HSC images + JWST images") 
        print("   - desi_hsc: HSC images + DESI images/spectra")
        print("   - custom: Any HuggingFace dataset (requires repo_id)")
        print("\n💡 For testing, you would load a real dataset like:")
        print("   dataset_key = runner.load_dataset('hsc_sdss', max_samples=100)")
        
        # Show available model types
        available_models = ModelLoader.available_models()
        print(f"\n🔧 Available model types: {available_models}")
        
        # Try to load a simple model (this may fail if dependencies are missing)
        try:
            print("\n🤖 Loading a simple model...")
            # Try loading a ViT model (most likely to work)
            model_key = runner.load_model(
                "vit", 
                "google/vit-base-patch16-224",
                use_safetensors=True
            )
            
            print("✅ Model loaded successfully!")
            
            print("💡 For actual embedding extraction, you would do:")
            print("   embeddings = runner.extract_embeddings(model_key, dataset_key)")
            print("   This requires a loaded dataset with real astronomical data.")
            
            # Show runner status
            status = runner.get_status()
            print(f"\n📊 Loaded models: {len(status['loaded_models'])}")
            print(f"📊 Loaded datasets: {len(status['loaded_datasets'])}")
            
        except Exception as e:
            print(f"⚠️ Model loading failed (this is expected if dependencies are missing): {e}")
            print("💡 This is normal in a minimal environment. The package structure is working correctly.")
        
        print("\n🎉 Basic functionality test complete!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Example completed successfully!")
    else:
        print("\n❌ Example failed!")
        sys.exit(1)