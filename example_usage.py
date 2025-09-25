#!/usr/bin/env python3
"""
Complete Example: Universal Neurons Analysis
This script demonstrates the full pipeline for finding universal neurons
across the Stanford CRFM GPT2-small models.
"""

import os
import sys
from pathlib import Path

# Import our pipeline modules
from universal_neurons_pipeline import run_universal_neurons_analysis
from dataset_utilities import (
    create_tokenized_dataset, 
    UniversalNeuronVisualizer,
    load_analysis_results,
    find_similar_neurons,
    compute_neuron_importance_scores
)

def main():
    """Main execution pipeline"""
    
    print("=" * 60)
    print("UNIVERSAL NEURONS ANALYSIS - STANFORD CRFM MODELS")
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
    
    # Analysis parameters
    CONFIG = {
        'n_tokens': 5_000_000,        # 5M tokens for analysis
        'ctx_len': 512,               # Context length
        'correlation_threshold': 0.6,  # Correlation threshold for universality
        'min_models': 3,              # Minimum models a neuron must appear in
        'output_dir': 'universal_neurons_results',
        'dataset_dir': 'datasets'
    }
    
    print(f"Models to analyze: {len(MODELS)}")
    print(f"Correlation threshold: {CONFIG['correlation_threshold']}")
    print(f"Minimum models: {CONFIG['min_models']}")
    
    # ========================================================================
    # STEP 1: CREATE TOKENIZED DATASET
    # ========================================================================
    
    print("\n" + "=" * 40)
    print("STEP 1: CREATING TOKENIZED DATASET")
    print("=" * 40)
    
    # Use first model for tokenization (they should all have same tokenizer)
    dataset_path = create_tokenized_dataset(
        model_name=MODELS[0],
        n_tokens=CONFIG['n_tokens'],
        ctx_len=CONFIG['ctx_len'],
        output_dir=CONFIG['dataset_dir']
    )
    
    print(f"✓ Dataset created: {dataset_path}")
    
    # ========================================================================
    # STEP 2: RUN FULL ANALYSIS PIPELINE
    # ========================================================================
    
    print("\n" + "=" * 40)
    print("STEP 2: RUNNING ANALYSIS PIPELINE")
    print("=" * 40)
    
    try:
        results = run_universal_neurons_analysis(
            model_names=MODELS,
            dataset_path=dataset_path,
            output_dir=CONFIG['output_dir'],
            correlation_threshold=CONFIG['correlation_threshold'],
            min_models=CONFIG['min_models']
        )
        
        print("✓ Analysis pipeline completed successfully!")
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        print("You may need to adjust memory settings or use fewer models.")
        return False
    
    # ========================================================================
    # STEP 3: GENERATE VISUALIZATIONS
    # ========================================================================
    
    print("\n" + "=" * 40)
    print("STEP 3: GENERATING VISUALIZATIONS")
    print("=" * 40)
    
    # Create output directory for plots
    plots_dir = Path(CONFIG['output_dir']) / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Initialize visualizer
    visualizer = UniversalNeuronVisualizer(results)
    
    # Generate all visualizations
    print("Creating correlation distribution plot...")
    visualizer.plot_correlation_distribution(
        save_path=plots_dir / 'correlation_distribution.png'
    )
    
    print("Creating universal vs regular properties comparison...")
    visualizer.plot_universal_properties_comparison(
        save_path=plots_dir / 'properties_comparison.png'
    )
    
    print("Creating correlation matrix heatmaps...")
    # Create heatmaps for first few model pairs
    model_pairs = list(results['correlations'].keys())[:3]
    for i, pair in enumerate(model_pairs):
        visualizer.plot_correlation_matrix_heatmap(
            model_pair=pair,
            layer_focus=6,  # Focus on middle layer
            save_path=plots_dir / f'correlation_heatmap_{i}.png'
        )
    
    print("Creating interactive dashboard...")
    dashboard_path = plots_dir / 'dashboard.html'
    visualizer.create_analysis_dashboard(save_path=str(dashboard_path))
    
    print("Creating universal neuron network visualization...")
    visualizer.plot_universal_neuron_network(
        save_path=str(plots_dir / 'neuron_network.html')
    )
    
    print("✓ All visualizations generated!")
    
    # ========================================================================
    # STEP 4: DETAILED ANALYSIS AND INSIGHTS
    # ========================================================================
    
    print("\n" + "=" * 40)
    print("STEP 4: EXTRACTING INSIGHTS")
    print("=" * 40)
    
    # Basic statistics
    n_universal = len(results['universal_neurons'])
    total_neurons_per_model = len(results['neuron_stats'][MODELS[0]])
    universality_rate = n_universal / total_neurons_per_model * 100
    
    print(f"Universal neurons found: {n_universal}")
    print(f"Total neurons per model: {total_neurons_per_model}")
    print(f"Universality rate: {universality_rate:.2f}%")
    
    if n_universal > 0:
        # Analyze universal neuron properties
        mean_correlation = results['universal_neurons']['mean_correlation'].mean()
        min_correlation = results['universal_neurons']['min_correlation'].mean()
        
        print(f"Average correlation strength: {mean_correlation:.3f}")
        print(f"Average minimum correlation: {min_correlation:.3f}")
        
        # Find most universal neurons (highest correlations)
        top_universal = results['universal_neurons'].nlargest(5, 'mean_correlation')
        print(f"\nTop 5 Universal Neurons:")
        for _, neuron in top_universal.iterrows():
            print(f"  L{neuron['reference_layer']}N{neuron['reference_neuron']}: "
                  f"r={neuron['mean_correlation']:.3f}, {neuron['n_models']} models")
        
        # Analyze layer distribution
        layer_dist = results['universal_neurons']['reference_layer'].value_counts().sort_index()
        print(f"\nUniversal neurons by layer:")
        for layer, count in layer_dist.items():
            print(f"  Layer {layer}: {count} neurons")
        
        # Find interesting individual neurons
        print(f"\nAnalyzing individual neuron properties...")
        
        # Get stats for first model as reference
        ref_stats = results['neuron_stats'][MODELS[0]]
        
        # Compute importance scores
        importance_scores = compute_neuron_importance_scores(ref_stats)
        print(f"Most important neurons (by composite score):")
        for i, ((layer, neuron), row) in enumerate(importance_scores.head(5).iterrows()):
            print(f"  #{i+1}: L{layer}N{neuron} (score: {row['importance_score']:.3f})")
        
        # Find similar neurons to a top universal neuron
        if not top_universal.empty:
            top_ref = top_universal.iloc[0]
            ref_layer, ref_neuron = top_ref['reference_layer'], top_ref['reference_neuron']
            
            similar_neurons = find_similar_neurons(
                ref_stats, ref_layer, ref_neuron, top_k=5
            )
            
            print(f"\nNeurons similar to L{ref_layer}N{ref_neuron}:")
            for _, sim_neuron in similar_neurons.iterrows():
                print(f"  L{sim_neuron['layer']}N{sim_neuron['neuron']}: "
                      f"similarity={sim_neuron['similarity_score']:.3f}")
    
    else:
        print("No universal neurons found with current thresholds.")
        print("Consider lowering correlation_threshold or min_models parameters.")
    
    # ========================================================================
    # STEP 5: SUMMARY AND RECOMMENDATIONS
    # ========================================================================
    
    print("\n" + "=" * 40)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 40)
    
    print(f"✓ Analyzed {len(MODELS)} models")
    print(f"✓ Found {n_universal} universal neurons")
    print(f"✓ Generated visualizations in {plots_dir}")
    print(f"✓ Saved detailed results in {CONFIG['output_dir']}")
    
    print(f"\nKey files generated:")
    print(f"  - {CONFIG['output_dir']}/universal_neurons.csv")
    print(f"  - {CONFIG['output_dir']}/universal_analysis.csv")  
    print(f"  - {plots_dir}/dashboard.html (interactive dashboard)")
    print(f"  - {plots_dir}/*.png (static plots)")
    
    print(f"\nNext steps:")
    print(f"  1. Open dashboard.html in browser for interactive exploration")
    print(f"  2. Examine universal_neurons.csv for specific neuron mappings")
    print(f"  3. Use the analysis results for downstream experiments")
    
    if n_universal > 0:
        print(f"  4. Consider running intervention experiments on universal neurons")
        print(f"  5. Analyze what computational roles these neurons serve")
    
    return True

def quick_test():
    """Run a quick test with smaller models and less data"""
    
    print("Running quick test with smaller models...")
    
    # Use just GPT2 variants for quick testing
    test_models = ["gpt2", "distilgpt2"]
    
    # Create small dataset
    dataset_path = create_tokenized_dataset(
        model_name="gpt2",
        n_tokens=100_000,  # Much smaller
        ctx_len=256,
        output_dir="test_datasets"
    )
    
    # Run with relaxed parameters
    results = run_universal_neurons_analysis(
        model_names=test_models,
        dataset_path=dataset_path,
        output_dir="test_results",
        correlation_threshold=0.4,  # Lower threshold
        min_models=2
    )
    
    # Quick visualization
    visualizer = UniversalNeuronVisualizer(results)
    visualizer.create_analysis_dashboard("test_dashboard.html")
    
    print("Quick test completed! Check test_dashboard.html")
    return results

if __name__ == "__main__":
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running in test mode...")
        quick_test()
    else:
        print("Running full analysis...")
        print("(Use --test flag for quick test with smaller models)")
        success = main()
        
        if success:
            print("\n🎉 Analysis completed successfully!")
            print("Check the generated dashboard.html file for interactive results!")
        else:
            print("\n❌ Analysis failed. Check error messages above.")
            sys.exit(1)
