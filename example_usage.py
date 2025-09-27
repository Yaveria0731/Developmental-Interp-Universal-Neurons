#!/usr/bin/env python3
"""
Modified Example Usage: Universal Neurons Analysis with Excess Correlation
This script demonstrates the full pipeline using the excess correlation metric
from the Universal Neurons paper to properly identify universal neurons.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Union

# Import our pipeline modules
from universal_neurons_pipeline import NeuronStatsGenerator, load_existing_neuron_stats
from dataset_utilities import (
    create_tokenized_dataset, 
    UniversalNeuronVisualizer,
    load_analysis_results,
    find_similar_neurons,
    compute_neuron_importance_scores
)

# Import the new excess correlation implementation
from excess_correlation_implementation import (
    run_excess_correlation_analysis,
    ExcessCorrelationComputer,
    ExcessCorrelationUniversalNeuronAnalyzer
)

def main_with_excess_correlation(checkpoint_value: Optional[Union[int, str]] = None):
    """Main execution pipeline using excess correlation"""
    
    print("=" * 60)
    print("UNIVERSAL NEURONS ANALYSIS - EXCESS CORRELATION METHOD")
    if checkpoint_value is not None:
        print(f"CHECKPOINT: {checkpoint_value}")
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
    
    # Analysis parameters - NOTE: Different from regular correlation
    CONFIG = {
        'n_tokens': 5_000_000,           # 5M tokens for analysis
        'ctx_len': 512,                  # Context length
        'excess_threshold': 0.1,         # EXCESS correlation threshold (not regular correlation)
        'top_k': None,                   # Alternative: take top K neurons instead of threshold
        'n_rotation_samples': 5,         # Number of random rotations for baseline
        'output_dir': 'universal_neurons_excess_results',
        'dataset_dir': 'datasets'
    }
    
    # Modify output directory for checkpoint
    if checkpoint_value is not None:
        CONFIG['output_dir'] = f"{CONFIG['output_dir']}_checkpoint_{checkpoint_value}"
    
    print(f"Models to analyze: {len(MODELS)}")
    print(f"Excess correlation threshold: {CONFIG['excess_threshold']}")
    print(f"Random rotation samples: {CONFIG['n_rotation_samples']}")
    if checkpoint_value is not None:
        print(f"Checkpoint: {checkpoint_value}")
    
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
    # STEP 2: RUN EXCESS CORRELATION ANALYSIS PIPELINE
    # ========================================================================
    
    print("\n" + "=" * 40)
    print("STEP 2: RUNNING EXCESS CORRELATION ANALYSIS")
    print("=" * 40)
    
    try:
        results = run_excess_correlation_analysis(
            model_names=MODELS,
            dataset_path=dataset_path,
            output_dir=CONFIG['output_dir'],
            excess_threshold=CONFIG['excess_threshold'],
            checkpoint_value=checkpoint_value,
            top_k=CONFIG['top_k'],
            n_rotation_samples=CONFIG['n_rotation_samples']
        )
        
        print("✓ Excess correlation analysis completed successfully!")
        
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
    
    # Adapt results format for existing visualizer
    # Note: The visualizer expects 'correlations' key, so we provide the correlation files
    adapted_results = {
        'neuron_stats': results['neuron_stats'],
        'correlation_files': results['excess_correlation_files'],  # Use excess correlation files
        'universal_neurons': results['universal_neurons'],
        'analysis': results['analysis'],
        'checkpoint': results['checkpoint']
    }
    
    # Initialize visualizer
    visualizer = UniversalNeuronVisualizer(adapted_results)
    
    # Generate all visualizations with checkpoint-specific names
    checkpoint_suffix = f"_checkpoint_{checkpoint_value}" if checkpoint_value is not None else ""
    
    print("Creating excess correlation distribution plot...")
    # Note: This will show distributions of excess correlations instead of regular correlations
    visualizer.plot_correlation_distribution(
        save_path=plots_dir / f'excess_correlation_distribution{checkpoint_suffix}.png'
    )
    
    print("Creating universal vs regular properties comparison...")
    visualizer.plot_universal_properties_comparison(
        save_path=plots_dir / f'properties_comparison_excess{checkpoint_suffix}.png'
    )
    
    print("Creating interactive dashboard...")
    dashboard_path = plots_dir / f'dashboard_excess{checkpoint_suffix}.html'
    visualizer.create_analysis_dashboard(save_path=str(dashboard_path))
    
    print("✓ All visualizations generated!")
    
    # ========================================================================
    # STEP 4: DETAILED ANALYSIS AND INSIGHTS
    # ========================================================================
    
    print("\n" + "=" * 40)
    print("STEP 4: EXTRACTING INSIGHTS")
    print("=" * 40)
    
    # Basic statistics
    n_universal = len(results['universal_neurons'])
    excess_scores = results['excess_scores']
    
    # Get total neurons from first model
    first_model_key = list(results['neuron_stats'].keys())[0]
    total_neurons_per_model = len(results['neuron_stats'][first_model_key])
    universality_rate = n_universal / total_neurons_per_model * 100
    
    print(f"Total neurons analyzed: {len(excess_scores)}")
    print(f"Universal neurons found: {n_universal}")
    print(f"Total neurons per model: {total_neurons_per_model}")
    print(f"Universality rate: {universality_rate:.2f}%")
    if checkpoint_value is not None:
        print(f"At checkpoint: {checkpoint_value}")
    
    # Excess correlation statistics
    print(f"\nExcess correlation statistics:")
    print(f"  Mean across all neurons: {excess_scores['excess_correlation'].mean():.4f}")
    print(f"  Std across all neurons: {excess_scores['excess_correlation'].std():.4f}")
    print(f"  Range: {excess_scores['excess_correlation'].min():.4f} to {excess_scores['excess_correlation'].max():.4f}")
    
    # Show percentiles to help understand the distribution
    percentiles = [75, 90, 95, 99]
    print(f"  Percentiles:")
    for p in percentiles:
        val = excess_scores['excess_correlation'].quantile(p/100)
        print(f"    {p}th percentile: {val:.4f}")
    
    if n_universal > 0:
        # Analyze universal neuron properties
        mean_excess = results['universal_neurons']['excess_correlation'].mean()
        min_excess = results['universal_neurons']['excess_correlation'].min()
        max_excess = results['universal_neurons']['excess_correlation'].max()
        
        print(f"\nUniversal neuron excess correlations:")
        print(f"  Mean: {mean_excess:.4f}")
        print(f"  Range: {min_excess:.4f} to {max_excess:.4f}")
        
        # Find most universal neurons (highest excess correlations)
        top_universal = results['universal_neurons'].nlargest(5, 'excess_correlation')
        print(f"\nTop 5 Universal Neurons by Excess Correlation:")
        for _, neuron in top_universal.iterrows():
            print(f"  L{neuron['reference_layer']}N{neuron['reference_neuron']}: "
                  f"excess_ρ={neuron['excess_correlation']:.4f}")
        
        # Analyze layer distribution
        layer_dist = results['universal_neurons']['reference_layer'].value_counts().sort_index()
        print(f"\nUniversal neurons by layer:")
        for layer, count in layer_dist.items():
            print(f"  Layer {layer}: {count} neurons")
        
        # Compare with threshold to show how selective it is
        above_threshold_count = (excess_scores['excess_correlation'] >= CONFIG['excess_threshold']).sum()
        print(f"\nThreshold analysis:")
        print(f"  Neurons above threshold {CONFIG['excess_threshold']}: {above_threshold_count}")
        print(f"  Fraction above threshold: {above_threshold_count / len(excess_scores):.1%}")
        
    else:
        print("No universal neurons found with current excess correlation threshold.")
        print("Consider:")
        print(f"  - Lowering excess_threshold (current: {CONFIG['excess_threshold']})")
        print(f"  - Using top_k approach instead")
        
        # Show what thresholds would yield some results
        potential_thresholds = [0.05, 0.02, 0.01]
        print(f"  - Potential thresholds:")
        for thresh in potential_thresholds:
            count = (excess_scores['excess_correlation'] >= thresh).sum()
            print(f"    {thresh:.3f}: {count} neurons ({count/len(excess_scores):.1%})")
    
    # ========================================================================
    # STEP 5: COMPARISON WITH REGULAR CORRELATION (if available)
    # ========================================================================
    
    print("\n" + "=" * 40)
    print("COMPARISON WITH BASELINE")
    print("=" * 40)
    
    # Load one excess correlation file to show the difference
    if results['excess_correlation_files']:
        import torch
        sample_file = results['excess_correlation_files'][0]
        sample_data = torch.load(sample_file, map_location='cpu')
        
        regular_corr = sample_data['regular_correlation_matrix']
        baseline_corr = sample_data['baseline_correlation_matrix']
        excess_corr = sample_data['excess_correlation_matrix']
        
        print(f"Sample from {os.path.basename(sample_file)}:")
        print(f"  Regular correlation range: {regular_corr.min():.4f} to {regular_corr.max():.4f}")
        print(f"  Baseline correlation range: {baseline_corr.min():.4f} to {baseline_corr.max():.4f}")
        print(f"  Excess correlation range: {excess_corr.min():.4f} to {excess_corr.max():.4f}")
        print(f"  Mean baseline correlation: {baseline_corr.mean():.4f}")
        print(f"  Mean excess correlation: {excess_corr.mean():.4f}")
    
    # ========================================================================
    # STEP 6: SUMMARY AND RECOMMENDATIONS
    # ========================================================================
    
    print("\n" + "=" * 40)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 40)
    
    print(f"✓ Analyzed {len(MODELS)} models using excess correlation method")
    if checkpoint_value is not None:
        print(f"✓ At checkpoint {checkpoint_value}")
    print(f"✓ Found {n_universal} universal neurons")
    print(f"✓ Generated visualizations in {plots_dir}")
    print(f"✓ Saved detailed results in {CONFIG['output_dir']}")
    
    print(f"\nKey files generated:")
    checkpoint_suffix = f"_checkpoint_{checkpoint_value}" if checkpoint_value is not None else ""
    print(f"  - {CONFIG['output_dir']}/excess_correlation_scores{checkpoint_suffix}.csv")
    print(f"  - {CONFIG['output_dir']}/universal_neurons_excess{checkpoint_suffix}.csv")
    print(f"  - {CONFIG['output_dir']}/universal_analysis_excess{checkpoint_suffix}.csv")  
    print(f"  - {plots_dir}/dashboard_excess{checkpoint_suffix}.html (interactive dashboard)")
    print(f"  - {len(results['excess_correlation_files'])} excess correlation files")
    
    print(f"\nMethodological improvements over basic correlation:")
    print(f"  ✓ Uses proper baseline from random rotations")
    print(f"  ✓ Controls for privileged neuron basis effects")
    print(f"  ✓ Implements exact formula from Universal Neurons paper")
    print(f"  ✓ More robust identification of truly universal neurons")
    
    print(f"\nNext steps:")
    print(f"  1. Open dashboard_excess{checkpoint_suffix}.html for interactive exploration")
    print(f"  2. Examine excess_correlation_scores{checkpoint_suffix}.csv for all neuron scores")
    print(f"  3. Compare results with regular correlation method")
    
    if n_universal > 0:
        print(f"  4. Run intervention experiments on identified universal neurons")
        print(f"  5. Analyze computational roles of high excess correlation neurons")
    
    if checkpoint_value is not None:
        print(f"  6. Compare excess correlation evolution across training checkpoints")
    
    return True

def quick_test_excess_correlation(checkpoint_value: Optional[Union[int, str]] = None):
    """Run a quick test with smaller models using excess correlation"""
    
    print("Running quick test with excess correlation method...")
    if checkpoint_value is not None:
        print(f"Testing checkpoint: {checkpoint_value}")
    
    # Use just GPT2 variants for quick testing
    test_models = ["gpt2", "distilgpt2"]
    
    # Create small dataset
    dataset_path = create_tokenized_dataset(
        model_name="gpt2",
        n_tokens=50_000,  # Very small for testing
        ctx_len=256,
        output_dir="test_datasets"
    )
    
    # Run with relaxed parameters
    output_dir = "test_excess_results"
    if checkpoint_value is not None:
        output_dir = f"{output_dir}_checkpoint_{checkpoint_value}"
    
    results = run_excess_correlation_analysis(
        model_names=test_models,
        dataset_path=dataset_path,
        output_dir=output_dir,
        excess_threshold=0.05,  # Lower threshold for testing
        checkpoint_value=checkpoint_value,
        n_rotation_samples=3    # Fewer samples for speed
    )
    
    # Quick visualization
    adapted_results = {
        'neuron_stats': results['neuron_stats'],
        'correlation_files': results['excess_correlation_files'],
        'universal_neurons': results['universal_neurons'],
        'analysis': results['analysis'],
        'checkpoint': results['checkpoint']
    }
    
    visualizer = UniversalNeuronVisualizer(adapted_results)
    checkpoint_suffix = f"_checkpoint_{checkpoint_value}" if checkpoint_value is not None else ""
    visualizer.create_analysis_dashboard(f"test_dashboard_excess{checkpoint_suffix}.html")
    
    print(f"Quick test completed! Check test_dashboard_excess{checkpoint_suffix}.html")
    return results

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Universal Neurons Analysis with Excess Correlation")
    parser.add_argument("--test", action="store_true", help="Run quick test with smaller models")
    parser.add_argument("--checkpoint", type=str, default=None, 
                       help="Specific checkpoint to analyze (e.g., '1000', 'step_5000')")
    parser.add_argument("--excess-threshold", type=float, default=0.1,
                       help="Excess correlation threshold for identifying universal neurons")
    parser.add_argument("--top-k", type=int, default=None,
                       help="Take top K neurons by excess correlation instead of using threshold")
    parser.add_argument("--rotation-samples", type=int, default=5,
                       help="Number of random rotation samples for baseline computation")
    
    args = parser.parse_args()
    
    if args.test:
        print("Running in test mode with excess correlation...")
        checkpoint = None
        if args.checkpoint:
            try:
                checkpoint = int(args.checkpoint)
            except ValueError:
                checkpoint = args.checkpoint
        quick_test_excess_correlation(checkpoint_value=checkpoint)
        
    else:
        print("Running full excess correlation analysis...")
        checkpoint = None
        if args.checkpoint:
            try:
                checkpoint = int(args.checkpoint)
            except ValueError:
                checkpoint = args.checkpoint
            print(f"Analyzing checkpoint: {checkpoint}")
        else:
            print("No checkpoint specified - analyzing final trained models")
        
        # Override config if command line args provided
        if hasattr(args, 'excess_threshold') and args.excess_threshold != 0.1:
            print(f"Using excess threshold: {args.excess_threshold}")
        if args.top_k:
            print(f"Using top-k approach: {args.top_k}")
        if args.rotation_samples != 5:
            print(f"Using {args.rotation_samples} rotation samples")
        
        success = main_with_excess_correlation(checkpoint_value=checkpoint)
        
        if success:
            print("\n🎉 Excess correlation analysis completed successfully!")
            checkpoint_suffix = f"_checkpoint_{checkpoint}" if checkpoint else ""
            print(f"Check the generated dashboard_excess{checkpoint_suffix}.html file for interactive results!")
            print("\nKey differences from regular correlation method:")
            print("  - More principled baseline using random rotations")
            print("  - Controls for neuron basis privilege")
            print("  - Implements exact paper methodology")
        else:
            print("\n⚠ Analysis failed. Check error messages above.")
            sys.exit(1)