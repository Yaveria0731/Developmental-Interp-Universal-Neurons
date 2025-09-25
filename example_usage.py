#!/usr/bin/env python3
"""
Updated Example: Universal Neurons Analysis with Full Dataset Streaming
This version removes the artificial 1000-sample limit and uses the full dataset efficiently
"""

import os
import sys
from pathlib import Path
import psutil
import torch

# Import our improved pipeline modules
from universal_neurons_pipeline import NeuronStatsGenerator, UniversalNeuronAnalyzer
from dataset_utilities import (
    create_tokenized_dataset, 
    UniversalNeuronVisualizer,
    load_analysis_results,
    find_similar_neurons,
    compute_neuron_importance_scores
)
# Import our new streaming correlation computer
from improved_correlation_streaming import (
    StreamingNeuronCorrelationComputer,
    memory_efficient_correlation_analysis,
    run_with_memory_config
)

def detect_memory_configuration():
    """Automatically detect appropriate memory configuration"""
    
    # Get available RAM
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    # Check if CUDA is available and get VRAM
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Detected {ram_gb:.1f}GB RAM, {vram_gb:.1f}GB VRAM")
    else:
        vram_gb = 0
        print(f"Detected {ram_gb:.1f}GB RAM, no CUDA available")
    
    # Choose configuration based on available memory
    if ram_gb >= 32 and vram_gb >= 16:
        return "high_memory"
    elif ram_gb >= 16 and vram_gb >= 8:
        return "medium_memory"
    elif ram_gb >= 8:
        return "low_memory"
    else:
        return "very_low_memory"

def main():
    """Main execution pipeline with full dataset support"""
    
    print("=" * 60)
    print("UNIVERSAL NEURONS ANALYSIS - FULL DATASET VERSION")
    print("=" * 60)
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    # Stanford CRFM GPT2-Small models from HuggingFace
    MODELS = [
        "stanford-crfm/alias-gpt2-small-x21",
        "stanford-crfm/battlestar-gpt2-small-x49", 
        "stanford-crfm/caprica-gpt2-small-x81",
        "stanford-crfm/darkmatter-gpt2-small-x343",
        "stanford-crfm/expanse-gpt2-small-x777"
    ]
    
    # Automatically detect memory configuration
    memory_level = detect_memory_configuration()
    print(f"Auto-detected memory configuration: {memory_level}")
    
    # Base analysis parameters
    CONFIG = {
        'n_tokens': 5_000_000,        # 5M tokens for dataset creation
        'ctx_len': 512,               # Context length
        'correlation_threshold': 0.6,  # Correlation threshold for universality
        'min_models': 3,              # Minimum models a neuron must appear in
        'output_dir': 'universal_neurons_results_full',
        'dataset_dir': 'datasets'
    }
    
    print(f"Models to analyze: {len(MODELS)}")
    print(f"Correlation threshold: {CONFIG['correlation_threshold']}")
    print(f"Minimum models: {CONFIG['min_models']}")
    print(f"Using FULL dataset (no artificial limits)")
    
    # ========================================================================
    # STEP 1: CREATE TOKENIZED DATASET
    # ========================================================================
    
    print("\n" + "=" * 40)
    print("STEP 1: CREATING TOKENIZED DATASET")
    print("=" * 40)
    
    dataset_path = create_tokenized_dataset(
        model_name=MODELS[0],
        n_tokens=CONFIG['n_tokens'],
        ctx_len=CONFIG['ctx_len'],
        output_dir=CONFIG['dataset_dir']
    )
    
    print(f"✓ Dataset created: {dataset_path}")
    
    # ========================================================================
    # STEP 2: RUN ANALYSIS WITH STREAMING
    # ========================================================================
    
    print("\n" + "=" * 40)
    print("STEP 2: RUNNING FULL ANALYSIS WITH STREAMING")
    print("=" * 40)
    
    try:
        # Use the memory-appropriate configuration
        results = run_with_memory_config(
            models=MODELS,
            dataset_path=dataset_path,
            memory_level=memory_level
        )
        
        print("✓ Full analysis pipeline completed successfully!")
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        print("Trying with more conservative memory settings...")
        
        # Fallback to very low memory config
        try:
            results = run_with_memory_config(
                models=MODELS,
                dataset_path=dataset_path, 
                memory_level="very_low_memory"
            )
            print("✓ Analysis completed with conservative settings!")
            
        except Exception as e2:
            print(f"✗ Analysis failed even with conservative settings: {e2}")
            print("Consider:")
            print("1. Using fewer models")
            print("2. Creating a smaller dataset")
            print("3. Running on a machine with more memory")
            return False
    
    # ========================================================================
    # STEP 3: GENERATE VISUALIZATIONS
    # ========================================================================
    
    print("\n" + "=" * 40)
    print("STEP 3: GENERATING VISUALIZATIONS")
    print("=" * 40)
    
    plots_dir = Path(CONFIG['output_dir']) / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    visualizer = UniversalNeuronVisualizer(results)
    
    print("Creating correlation distribution plot...")
    visualizer.plot_correlation_distribution(
        save_path=plots_dir / 'correlation_distribution.png'
    )
    
    print("Creating universal vs regular properties comparison...")
    visualizer.plot_universal_properties_comparison(
        save_path=plots_dir / 'properties_comparison.png'
    )
    
    print("Creating interactive dashboard...")
    dashboard_path = plots_dir / 'dashboard.html'
    visualizer.create_analysis_dashboard(save_path=str(dashboard_path))
    
    print("✓ All visualizations generated!")
    
    # ========================================================================
    # STEP 4: DETAILED ANALYSIS AND INSIGHTS
    # ========================================================================
    
    print("\n" + "=" * 40)
    print("STEP 4: EXTRACTING INSIGHTS")
    print("=" * 40)
    
    n_universal = len(results['universal_neurons'])
    total_neurons_per_model = len(results['neuron_stats'][MODELS[0]])
    universality_rate = n_universal / total_neurons_per_model * 100
    
    print(f"Universal neurons found: {n_universal}")
    print(f"Total neurons per model: {total_neurons_per_model}")
    print(f"Universality rate: {universality_rate:.2f}%")
    
    if n_universal > 0:
        mean_correlation = results['universal_neurons']['mean_correlation'].mean()
        min_correlation = results['universal_neurons']['min_correlation'].mean()
        
        print(f"Average correlation strength: {mean_correlation:.3f}")
        print(f"Average minimum correlation: {min_correlation:.3f}")
        
        # Top universal neurons
        top_universal = results['universal_neurons'].nlargest(5, 'mean_correlation')
        print(f"\nTop 5 Universal Neurons:")
        for _, neuron in top_universal.iterrows():
            print(f"  L{neuron['reference_layer']}N{neuron['reference_neuron']}: "
                  f"r={neuron['mean_correlation']:.3f}, {neuron['n_models']} models")
        
        # Layer distribution
        layer_dist = results['universal_neurons']['reference_layer'].value_counts().sort_index()
        print(f"\nUniversal neurons by layer:")
        for layer, count in layer_dist.items():
            print(f"  Layer {layer}: {count} neurons")
            
    else:
        print("No universal neurons found with current thresholds.")
        print("This could mean:")
        print("1. Models are more diverse than expected")
        print("2. Threshold is too strict (try lowering to 0.4-0.5)")
        print("3. Need more data for stable correlations")
    
    # ========================================================================
    # STEP 5: SUMMARY AND DATA USAGE STATISTICS
    # ========================================================================
    
    print("\n" + "=" * 40)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 40)
    
    print(f"✓ Analyzed {len(MODELS)} models")
    print(f"✓ Used FULL dataset (no sample limits)")
    print(f"✓ Found {n_universal} universal neurons")
    print(f"✓ Memory configuration: {memory_level}")
    
    # Show dataset usage statistics
    import datasets
    dataset = datasets.load_from_disk(dataset_path)
    total_sequences = len(dataset)
    total_tokens = total_sequences * CONFIG['ctx_len']
    
    print(f"\nDataset Statistics:")
    print(f"  Total sequences: {total_sequences:,}")
    print(f"  Total tokens processed: {total_tokens:,}")
    print(f"  Context length: {CONFIG['ctx_len']}")
    print(f"  No artificial sample limits applied ✓")
    
    print(f"\nKey files generated:")
    print(f"  - {CONFIG['output_dir']}/universal_neurons.csv")
    print(f"  - {CONFIG['output_dir']}/universal_analysis.csv")
    print(f"  - {plots_dir}/dashboard.html")
    print(f"  - {CONFIG['output_dir']}/individual_correlations/ (correlation matrices)")
    
    return True

def quick_test_streaming():
    """Run a quick test to verify streaming works"""
    
    print("Running streaming test with small models...")
    
    test_models = ["gpt2", "distilgpt2"]  
    
    # Create small dataset
    dataset_path = create_tokenized_dataset(
        model_name="gpt2",
        n_tokens=200_000,  # 200k tokens
        ctx_len=256,
        output_dir="test_datasets"
    )
    
    # Test streaming correlation
    correlator = StreamingNeuronCorrelationComputer(test_models)
    correlations = correlator.compute_all_correlations_streaming(
        dataset_path,
        batch_size=8,
        max_samples=None,  # Use full dataset
        use_fp16=True
    )
    
    print(f"✓ Streaming test completed!")
    print(f"✓ Processed full dataset without sample limits")
    print(f"✓ Correlation shape: {list(correlations.values())[0].shape}")
    
    return correlations

def compare_sample_limits():
    """Compare results with and without sample limits"""
    
    print("Comparing limited vs full dataset results...")
    
    test_models = ["gpt2", "distilgpt2"]
    
    dataset_path = create_tokenized_dataset(
        model_name="gpt2",
        n_tokens=500_000,
        ctx_len=256,
        output_dir="comparison_datasets"
    )
    
    correlator = StreamingNeuronCorrelationComputer(test_models)
    
    # Run with limit (like original code)
    print("Running with 1000 sample limit...")
    corr_limited = correlator.compute_pairwise_correlation_streaming(
        "gpt2", "distilgpt2", dataset_path,
        max_samples=1000, batch_size=8
    )
    
    # Run with full dataset
    print("Running with full dataset...")
    corr_full = correlator.compute_pairwise_correlation_streaming(
        "gpt2", "distilgpt2", dataset_path, 
        max_samples=None, batch_size=8
    )
    
    # Compare statistics
    print("\nComparison Results:")
    print(f"Limited (1000 samples) - Mean |correlation|: {torch.abs(corr_limited).mean():.4f}")
    print(f"Full dataset - Mean |correlation|: {torch.abs(corr_full).mean():.4f}")
    print(f"Correlation between methods: {torch.corrcoef(torch.stack([corr_limited.flatten(), corr_full.flatten()]))[0,1]:.4f}")
    
    return corr_limited, corr_full

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            print("Running streaming test...")
            quick_test_streaming()
        elif sys.argv[1] == "--compare":
            print("Running comparison test...")
            compare_sample_limits()
        else:
            print("Unknown option. Use --test or --compare")
    else:
        print("Running full analysis with streaming (no sample limits)...")
        print("This will use your FULL dataset for better correlation estimates!")
        success = main()
        
        if success:
            print("\n🎉 Full dataset analysis completed successfully!")
            print("Your correlations are now computed on the entire dataset!")
        else:
            print("\n❌ Analysis failed. Check error messages above.")
            sys.exit(1)