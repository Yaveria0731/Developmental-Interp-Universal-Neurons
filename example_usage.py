#!/usr/bin/env python3
"""
Complete Example: Universal Neurons Analysis with Checkpoint Support
This script demonstrates the full pipeline for finding universal neurons
across different training checkpoints of Stanford CRFM GPT2-small models.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Union

# Import our pipeline modules
from universal_neurons_pipeline import run_universal_neurons_analysis
from dataset_utilities import (
    create_tokenized_dataset, 
    UniversalNeuronVisualizer,
    load_analysis_results,
    find_similar_neurons,
    compute_neuron_importance_scores
)

def main(checkpoint_value: Optional[Union[int, str]] = None):
    """Main execution pipeline"""
    
    print("=" * 60)
    print("UNIVERSAL NEURONS ANALYSIS - STANFORD CRFM MODELS")
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
    
    # Analysis parameters
    CONFIG = {
        'n_tokens': 5_000_000,        # 5M tokens for analysis
        'ctx_len': 512,               # Context length
        'correlation_threshold': 0.6,  # Correlation threshold for universality
        'min_models': 3,              # Minimum models a neuron must appear in
        'output_dir': 'universal_neurons_results',
        'dataset_dir': 'datasets'
    }
    
    # Modify output directory for checkpoint
    if checkpoint_value is not None:
        CONFIG['output_dir'] = f"{CONFIG['output_dir']}_checkpoint_{checkpoint_value}"
    
    print(f"Models to analyze: {len(MODELS)}")
    print(f"Correlation threshold: {CONFIG['correlation_threshold']}")
    print(f"Minimum models: {CONFIG['min_models']}")
    if checkpoint_value is not None:
        print(f"Checkpoint: {checkpoint_value}")
    
    # ========================================================================
    # STEP 1: CREATE TOKENIZED DATASET
    # ========================================================================
    
    print("\n" + "=" * 40)
    print("STEP 1: CREATING TOKENIZED DATASET")
    print("=" * 40)
    
    # Use first model for tokenization (they should all have same tokenizer)
    # For checkpoint analysis, we'll use the base model for tokenization unless 
    # there are tokenizer differences at different checkpoints
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
            min_models=CONFIG['min_models'],
            checkpoint_value=checkpoint_value  # Pass checkpoint parameter
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
    
    # Generate all visualizations with checkpoint-specific names
    checkpoint_suffix = f"_checkpoint_{checkpoint_value}" if checkpoint_value is not None else ""
    
    print("Creating correlation distribution plot...")
    visualizer.plot_correlation_distribution(
        save_path=plots_dir / f'correlation_distribution{checkpoint_suffix}.png'
    )
    
    print("Creating universal vs regular properties comparison...")
    visualizer.plot_universal_properties_comparison(
        save_path=plots_dir / f'properties_comparison{checkpoint_suffix}.png'
    )
    
    print("Creating correlation matrix heatmaps...")
    # Create heatmaps for first few model pairs
    model_pairs = list(results['correlations'].keys())[:3]
    for i, pair in enumerate(model_pairs):
        visualizer.plot_correlation_matrix_heatmap(
            model_pair=pair,
            layer_focus=6,  # Focus on middle layer
            save_path=plots_dir / f'correlation_heatmap_{i}{checkpoint_suffix}.png'
        )
    
    print("Creating interactive dashboard...")
    dashboard_path = plots_dir / f'dashboard{checkpoint_suffix}.html'
    visualizer.create_analysis_dashboard(save_path=str(dashboard_path))
    
    print("Creating universal neuron network visualization...")
    visualizer.plot_universal_neuron_network(
        save_path=str(plots_dir / f'neuron_network{checkpoint_suffix}.html')
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
    
    # Get total neurons from first model (they should all be the same)
    first_model_key = list(results['neuron_stats'].keys())[0]
    total_neurons_per_model = len(results['neuron_stats'][first_model_key])
    universality_rate = n_universal / total_neurons_per_model * 100
    
    print(f"Universal neurons found: {n_universal}")
    print(f"Total neurons per model: {total_neurons_per_model}")
    print(f"Universality rate: {universality_rate:.2f}%")
    if checkpoint_value is not None:
        print(f"At checkpoint: {checkpoint_value}")
    
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
        ref_stats = results['neuron_stats'][first_model_key]
        
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
    if checkpoint_value is not None:
        print(f"✓ At checkpoint {checkpoint_value}")
    print(f"✓ Found {n_universal} universal neurons")
    print(f"✓ Generated visualizations in {plots_dir}")
    print(f"✓ Saved detailed results in {CONFIG['output_dir']}")
    
    print(f"\nKey files generated:")
    checkpoint_suffix = f"_checkpoint_{checkpoint_value}" if checkpoint_value is not None else ""
    print(f"  - {CONFIG['output_dir']}/universal_neurons{checkpoint_suffix}.csv")
    print(f"  - {CONFIG['output_dir']}/universal_analysis{checkpoint_suffix}.csv")  
    print(f"  - {plots_dir}/dashboard{checkpoint_suffix}.html (interactive dashboard)")
    print(f"  - {plots_dir}/*{checkpoint_suffix}.png (static plots)")
    
    print(f"\nNext steps:")
    print(f"  1. Open dashboard{checkpoint_suffix}.html in browser for interactive exploration")
    print(f"  2. Examine universal_neurons{checkpoint_suffix}.csv for specific neuron mappings")
    print(f"  3. Use the analysis results for downstream experiments")
    
    if n_universal > 0:
        print(f"  4. Consider running intervention experiments on universal neurons")
        print(f"  5. Analyze what computational roles these neurons serve")
    
    if checkpoint_value is not None:
        print(f"  6. Compare results across different checkpoints to study evolution")
    
    return True

def compare_checkpoints(checkpoint_list, base_config=None):
    """Run analysis across multiple checkpoints for comparison"""
    
    print("=" * 60)
    print("MULTI-CHECKPOINT COMPARISON ANALYSIS")
    print("=" * 60)
    
    results_by_checkpoint = {}
    
    for checkpoint in checkpoint_list:
        print(f"\n{'='*20} CHECKPOINT {checkpoint} {'='*20}")
        success = main(checkpoint_value=checkpoint)
        if success:
            # Load results for comparison
            results_dir = f"universal_neurons_results_checkpoint_{checkpoint}"
            try:
                results = load_analysis_results(results_dir)
                results_by_checkpoint[checkpoint] = results
                print(f"✓ Checkpoint {checkpoint} completed")
            except Exception as e:
                print(f"✗ Failed to load results for checkpoint {checkpoint}: {e}")
        else:
            print(f"✗ Analysis failed for checkpoint {checkpoint}")
    
    # Generate comparison report
    if len(results_by_checkpoint) > 1:
        print("\n" + "=" * 60)
        print("CHECKPOINT COMPARISON SUMMARY")
        print("=" * 60)
        
        comparison_data = []
        for checkpoint, results in results_by_checkpoint.items():
            if 'universal_neurons' in results:
                n_universal = len(results['universal_neurons'])
                if n_universal > 0:
                    mean_corr = results['universal_neurons']['mean_correlation'].mean()
                else:
                    mean_corr = 0.0
            else:
                n_universal = 0
                mean_corr = 0.0
            
            comparison_data.append({
                'checkpoint': checkpoint,
                'universal_neurons': n_universal,
                'mean_correlation': mean_corr
            })
        
        print("Checkpoint | Universal Neurons | Mean Correlation")
        print("-" * 50)
        for data in comparison_data:
            print(f"{data['checkpoint']:^10} | {data['universal_neurons']:^17} | {data['mean_correlation']:.3f}")
        
        # Save comparison
        import pandas as pd
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv("checkpoint_comparison.csv", index=False)
        print(f"\n✓ Comparison saved to checkpoint_comparison.csv")
    
    return results_by_checkpoint

def quick_test(checkpoint_value: Optional[Union[int, str]] = None):
    """Run a quick test with smaller models and less data"""
    
    print("Running quick test with smaller models...")
    if checkpoint_value is not None:
        print(f"Testing checkpoint: {checkpoint_value}")
    
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
    output_dir = "test_results"
    if checkpoint_value is not None:
        output_dir = f"{output_dir}_checkpoint_{checkpoint_value}"
    
    results = run_universal_neurons_analysis(
        model_names=test_models,
        dataset_path=dataset_path,
        output_dir=output_dir,
        correlation_threshold=0.4,  # Lower threshold
        min_models=2,
        checkpoint_value=checkpoint_value
    )
    
    # Quick visualization
    visualizer = UniversalNeuronVisualizer(results)
    checkpoint_suffix = f"_checkpoint_{checkpoint_value}" if checkpoint_value is not None else ""
    visualizer.create_analysis_dashboard(f"test_dashboard{checkpoint_suffix}.html")
    
    print(f"Quick test completed! Check test_dashboard{checkpoint_suffix}.html")
    return results

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Universal Neurons Analysis with Checkpoint Support")
    parser.add_argument("--test", action="store_true", help="Run quick test with smaller models")
    parser.add_argument("--checkpoint", type=str, default=None, 
                       help="Specific checkpoint to analyze (e.g., '1000', 'step_5000')")
    parser.add_argument("--compare-checkpoints", nargs="+", type=str, default=None,
                       help="List of checkpoints to compare (e.g., --compare-checkpoints 1000 2000 5000)")
    
    args = parser.parse_args()
    
    if args.compare_checkpoints:
        print("Running multi-checkpoint comparison...")
        checkpoint_list = args.compare_checkpoints
        # Convert to integers if they're numeric
        processed_checkpoints = []
        for cp in checkpoint_list:
            try:
                processed_checkpoints.append(int(cp))
            except ValueError:
                processed_checkpoints.append(cp)  # Keep as string if not numeric
        
        results_by_checkpoint = compare_checkpoints(processed_checkpoints)
        
    elif args.test:
        print("Running in test mode...")
        checkpoint = None
        if args.checkpoint:
            try:
                checkpoint = int(args.checkpoint)
            except ValueError:
                checkpoint = args.checkpoint
        quick_test(checkpoint_value=checkpoint)
        
    else:
        print("Running full analysis...")
        checkpoint = None
        if args.checkpoint:
            try:
                checkpoint = int(args.checkpoint)
            except ValueError:
                checkpoint = args.checkpoint
            print(f"Analyzing checkpoint: {checkpoint}")
        else:
            print("No checkpoint specified - analyzing final trained models")
        
        success = main(checkpoint_value=checkpoint)
        
        if success:
            print("\n🎉 Analysis completed successfully!")
            checkpoint_suffix = f"_checkpoint_{checkpoint}" if checkpoint else ""
            print(f"Check the generated dashboard{checkpoint_suffix}.html file for interactive results!")
        else:
            print("\n⚠ Analysis failed. Check error messages above.")
            sys.exit(1)