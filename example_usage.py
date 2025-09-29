#!/usr/bin/env python3
"""
Example Usage: Repository-Compatible Universal Neurons Analysis
Demonstrates the two-step pipeline matching the original repository structure.
"""

import argparse
from pathlib import Path
from universal_neurons import (
    create_tokenized_dataset,
    run_correlation_analysis,
    run_universal_neurons_analysis,
    run_full_pipeline,
    CorrelationComputer,
    ExcessCorrelationComputer,
    UniversalNeuronAnalyzer,
    NeuronStatsGenerator
)


def quick_test(checkpoint_value=None):
    """Run a quick test with smaller models and dataset"""
    print("Running quick test...")
    
    # Use smaller models for testing
    test_models = ["gpt2", "distilgpt2"]
    
    # Create small test dataset
    dataset_path = create_tokenized_dataset(
        model_name="gpt2",
        n_tokens=50000,  # Small for testing
        ctx_len=256,
        output_dir="test_datasets"
    )
    
    # Run full pipeline with test settings
    results = run_full_pipeline(
        model_names=test_models,
        dataset_path=dataset_path,
        output_dir="test_results",
        excess_threshold=0.02,  # Lower threshold for testing
        checkpoint_value=checkpoint_value,
        batch_size=8
    )
    
    print("Quick test completed! Check test_results/ directory.")
    return results


def full_analysis(checkpoint_value=None):
    """Run full analysis with Stanford CRFM models"""
    print("Running full analysis...")
    
    # Stanford CRFM GPT2-Small models
    models = [
        "stanford-crfm/alias-gpt2-small-x21",
        "stanford-crfm/battlestar-gpt2-small-x49",
        "stanford-crfm/caprica-gpt2-small-x81",
        "stanford-crfm/darkmatter-gpt2-small-x343",
        "stanford-crfm/expanse-gpt2-small-x777"
    ]
    
    # Create larger dataset for full analysis
    dataset_path = create_tokenized_dataset(
        model_name=models[0],
        n_tokens=1_000_000,  # 1M tokens for good statistics
        ctx_len=512,
        output_dir="datasets"
    )
    
    # Run full pipeline
    results = run_full_pipeline(
        model_names=models,
        dataset_path=dataset_path,
        output_dir="universal_neurons_results",
        excess_threshold=0.1,
        checkpoint_value=checkpoint_value,
        batch_size=8
    )
    
    print("Full analysis completed! Check universal_neurons_results/ directory.")
    return results


def correlation_only(checkpoint_value=None):
    """Run only correlation analysis step"""
    print("Running correlation analysis only...")
    
    models = [
        "stanford-crfm/alias-gpt2-small-x21",
        "stanford-crfm/battlestar-gpt2-small-x49",
        "stanford-crfm/caprica-gpt2-small-x81"
    ]
    
    # Create dataset
    dataset_path = create_tokenized_dataset(
        model_name=models[0],
        n_tokens=500_000,
        output_dir="datasets"
    )
    
    # Run only correlation analysis
    correlation_dir = run_correlation_analysis(
        model_names=models,
        dataset_path=dataset_path,
        checkpoint_value=checkpoint_value,
        batch_size=8
    )
    
    print(f"Correlation analysis completed! Results saved to: {correlation_dir}")
    return correlation_dir


def universal_neurons_only(correlation_results_dir="correlation_results", checkpoint_value=None):
    """Run only universal neurons analysis from existing correlations"""
    print("Running universal neurons analysis from existing correlations...")
    
    models = [
        "stanford-crfm/alias-gpt2-small-x21",
        "stanford-crfm/battlestar-gpt2-small-x49", 
        "stanford-crfm/caprica-gpt2-small-x81"
    ]
    
    # Create dataset path (should already exist from correlation step)
    dataset_path = f"datasets/stanford-crfm_alias-gpt2-small-x21/pile"
    
    # Run universal neurons analysis
    results = run_universal_neurons_analysis(
        model_names=models,
        dataset_path=dataset_path,
        output_dir="universal_neurons_results",
        excess_threshold=0.1,
        checkpoint_value=checkpoint_value,
        correlation_results_dir=correlation_results_dir
    )
    
    print("Universal neurons analysis completed!")
    return results


def analyze_checkpoint_progression(checkpoints):
    """Analyze how universality evolves across training checkpoints"""
    print(f"Analyzing checkpoint progression: {checkpoints}")
    
    models = [
        "stanford-crfm/alias-gpt2-small-x21",
        "stanford-crfm/battlestar-gpt2-small-x49",
        "stanford-crfm/caprica-gpt2-small-x81"
    ]
    
    # Create dataset once
    dataset_path = create_tokenized_dataset(
        model_name=models[0],
        n_tokens=500_000,
        output_dir="datasets"
    )
    
    checkpoint_results = {}
    
    for checkpoint in checkpoints:
        print(f"\nAnalyzing checkpoint {checkpoint}...")
        
        results = run_full_pipeline(
            model_names=models,
            dataset_path=dataset_path,
            output_dir=f"checkpoint_analysis_checkpoint_{checkpoint}",
            excess_threshold=0.05,
            checkpoint_value=checkpoint,
            batch_size=8
        )
        
        checkpoint_results[checkpoint] = results
    
    # Create comparison visualization
    import matplotlib.pyplot as plt
    import pandas as pd
    
    comparison_data = []
    for checkpoint, results in checkpoint_results.items():
        n_universal = len(results['universal_neurons'])
        if n_universal > 0:
            mean_excess = results['universal_neurons']['excess_correlation'].mean()
        else:
            mean_excess = 0.0
        
        comparison_data.append({
            'checkpoint': checkpoint,
            'n_universal': n_universal,
            'mean_excess_correlation': mean_excess
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(comparison_df['checkpoint'], comparison_df['n_universal'], 'o-')
    ax1.set_xlabel('Training Checkpoint')
    ax1.set_ylabel('Number of Universal Neurons')
    ax1.set_title('Universal Neuron Count vs Training Progress')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(comparison_df['checkpoint'], comparison_df['mean_excess_correlation'], 'o-', color='orange')
    ax2.set_xlabel('Training Checkpoint')
    ax2.set_ylabel('Mean Excess Correlation')
    ax2.set_title('Excess Correlation Strength vs Training Progress')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('checkpoint_analysis/checkpoint_progression.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save comparison data
    comparison_df.to_csv('checkpoint_analysis/checkpoint_comparison.csv', index=False)
    
    print("Checkpoint progression analysis completed!")
    print("Check checkpoint_analysis/ directory for results.")
    
    return checkpoint_results


def custom_analysis_example():
    """Example of using the classes directly for custom analysis"""
    print("Running custom analysis example...")
    
    models = ["gpt2", "distilgpt2"]
    
    # Create dataset
    dataset_path = create_tokenized_dataset(
        model_name="gpt2",
        n_tokens=50000,
        output_dir="custom_datasets"
    )
    
    # Step 1: Compute correlations manually
    print("Step 1: Computing correlations...")
    correlator = CorrelationComputer()
    
    # Compute regular correlation
    correlator.run_correlation_experiment(
        models[0], models[1], dataset_path,
        batch_size=4, baseline='none',
        output_dir="custom_correlations"
    )
    
    # Compute baseline correlation
    correlator.run_correlation_experiment(
        models[0], models[1], dataset_path,
        batch_size=4, baseline='rotation',
        output_dir="custom_correlations"
    )
    
    # Step 2: Compute excess correlations from saved matrices
    print("Step 2: Computing excess correlations...")
    excess_computer = ExcessCorrelationComputer("custom_correlations")
    
    dataset_name = "pile"  # basename of dataset_path
    excess_df = excess_computer.compute_excess_correlations_from_saved(models, dataset_name)
    
    # Step 3: Analyze results
    analyzer = UniversalNeuronAnalyzer(excess_df)
    
    # Get top 10 neurons by excess correlation
    top_neurons = analyzer.identify_universal_neurons(top_k=10)
    print("\nTop 10 universal neurons:")
    print(top_neurons[['layer', 'neuron', 'excess_correlation']])
    
    # Get neurons above threshold
    threshold_neurons = analyzer.identify_universal_neurons(excess_threshold=0.05)
    print(f"\nFound {len(threshold_neurons)} neurons above threshold 0.05")
    
    # Step 4: Compute neuron statistics
    print("Step 3: Computing neuron statistics...")
    stats_generator = NeuronStatsGenerator(models[0])
    neuron_stats = stats_generator.compute_neuron_stats()
    
    return {
        'excess_correlations': excess_df,
        'top_neurons': top_neurons,
        'threshold_neurons': threshold_neurons,
        'neuron_stats': neuron_stats
    }


def memory_efficient_test():
    """Test with minimal memory footprint"""
    print("Running memory efficient test...")
    
    models = ["gpt2", "distilgpt2"]
    
    dataset_path = create_tokenized_dataset(
        model_name="gpt2",
        n_tokens=50000,   # Very small dataset
        ctx_len=256,      # Short sequences
        output_dir="memory_test_datasets"
    )
    
    results = run_full_pipeline(
        model_names=models,
        dataset_path=dataset_path,
        output_dir="memory_test_results",
        excess_threshold=0.01,
        batch_size=4      # Small batch size
    )
    
    print("Memory efficient test completed!")
    return results


def two_step_example():
    """Example demonstrating the two-step pipeline explicitly"""
    print("Running two-step pipeline example...")
    
    models = ["gpt2", "distilgpt2"]
    
    # Create dataset
    dataset_path = create_tokenized_dataset(
        model_name="gpt2",
        n_tokens=100000,
        output_dir="two_step_datasets"
    )
    
    # STEP 1: Compute correlations
    print("\n" + "="*50)
    print("STEP 1: CORRELATION COMPUTATION")
    print("="*50)
    
    correlation_dir = run_correlation_analysis(
        model_names=models,
        dataset_path=dataset_path,
        batch_size=4,
        output_dir="two_step_correlations"
    )
    
    # STEP 2: Compute universal neurons from saved correlations
    print("\n" + "="*50)
    print("STEP 2: UNIVERSAL NEURONS ANALYSIS")
    print("="*50)
    
    results = run_universal_neurons_analysis(
        model_names=models,
        dataset_path=dataset_path,
        output_dir="two_step_results",
        excess_threshold=0.05,
        correlation_results_dir=correlation_dir
    )
    
    print("\nTwo-step pipeline completed!")
    print(f"Correlation results: {correlation_dir}")
    print(f"Universal neurons results: two_step_results")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Repository-Compatible Universal Neurons Analysis")
    parser.add_argument("--test", action="store_true", help="Run quick test with smaller models")
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint to analyze")
    parser.add_argument("--compare-checkpoints", nargs='+', type=int, 
                       help="Compare multiple checkpoints (e.g., --compare-checkpoints 1000 5000 10000)")
    parser.add_argument("--excess-threshold", type=float, default=0.1,
                       help="Excess correlation threshold for identifying universal neurons")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for processing")
    
    # Pipeline control arguments
    parser.add_argument("--correlation-only", action="store_true",
                       help="Run only correlation analysis step")
    parser.add_argument("--universal-only", action="store_true",
                       help="Run only universal neurons analysis from existing correlations")
    parser.add_argument("--correlation-dir", type=str, default="correlation_results",
                       help="Directory with existing correlation results")
    
    # Example arguments
    parser.add_argument("--memory-test", action="store_true", 
                       help="Run memory efficient test")
    parser.add_argument("--custom", action="store_true",
                       help="Run custom analysis example")
    parser.add_argument("--two-step", action="store_true",
                       help="Run two-step pipeline example")
    
    args = parser.parse_args()
    
    # Parse checkpoint
    checkpoint = None
    if args.checkpoint:
        try:
            checkpoint = int(args.checkpoint)
        except ValueError:
            checkpoint = args.checkpoint
    
    # Execute based on arguments
    if args.memory_test:
        print("Running memory efficient test...")
        memory_efficient_test()
        
    elif args.test:
        print("Running in test mode...")
        quick_test(checkpoint_value=checkpoint)
        
    elif args.correlation_only:
        print("Running correlation analysis only...")
        correlation_only(checkpoint_value=checkpoint)
        
    elif args.universal_only:
        print("Running universal neurons analysis only...")
        universal_neurons_only(
            correlation_results_dir=args.correlation_dir,
            checkpoint_value=checkpoint
        )
        
    elif args.compare_checkpoints:
        print("Running checkpoint comparison analysis...")
        analyze_checkpoint_progression(args.compare_checkpoints)
        
    elif args.custom:
        print("Running custom analysis example...")
        custom_analysis_example()
        
    elif args.two_step:
        print("Running two-step pipeline example...")
        two_step_example()
        
    else:
        print("Running full analysis...")
        full_analysis(checkpoint_value=checkpoint)


if __name__ == "__main__":
    main()