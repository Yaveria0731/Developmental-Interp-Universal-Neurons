#!/usr/bin/env python3
"""
Example Usage: Efficient Universal Neurons Analysis
Demonstrates how to use the efficient excess correlation method.
"""

import argparse
from pathlib import Path
from universal_neurons import (
    create_tokenized_dataset,
    run_universal_neurons_analysis,
    EfficientExcessCorrelationComputer,
    UniversalNeuronAnalyzer
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
    
    # Run analysis with relaxed parameters for speed
    results = run_universal_neurons_analysis(
        model_names=test_models,
        dataset_path=dataset_path,
        output_dir="test_results",
        excess_threshold=0.02,  # Lower threshold for testing
        checkpoint_value=checkpoint_value,
        n_rotation_samples=2,   # Fewer samples for speed
        batch_size=8            # Reasonable batch size for testing
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
    
    # Run full analysis
    results = run_universal_neurons_analysis(
        model_names=models,
        dataset_path=dataset_path,
        output_dir="universal_neurons_results",
        excess_threshold=0.1,
        checkpoint_value=checkpoint_value,
        n_rotation_samples=5,
        batch_size=8
    )
    
    print("Full analysis completed! Check universal_neurons_results/ directory.")
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
        n_tokens=500_000,  # 500K tokens for checkpoint analysis
        output_dir="datasets"
    )
    
    checkpoint_results = {}
    
    for checkpoint in checkpoints:
        print(f"\nAnalyzing checkpoint {checkpoint}...")
        
        results = run_universal_neurons_analysis(
            model_names=models,
            dataset_path=dataset_path,
            output_dir=f"checkpoint_analysis",
            excess_threshold=0.05,
            checkpoint_value=checkpoint,
            n_rotation_samples=3,
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


def memory_efficient_test():
    """Test with minimal memory footprint for very large models"""
    print("Running memory efficient test...")
    
    models = ["gpt2", "distilgpt2"]
    
    dataset_path = create_tokenized_dataset(
        model_name="gpt2",
        n_tokens=100000,  # Small dataset
        ctx_len=256,      # Short sequences
        output_dir="memory_test_datasets"
    )
    
    results = run_universal_neurons_analysis(
        model_names=models,
        dataset_path=dataset_path,
        output_dir="memory_test_results",
        excess_threshold=0.01,
        n_rotation_samples=2,  # Minimal rotations
        batch_size=4           # Small batch size
    )
    
    print("Memory efficient test completed!")
    return results


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
    
    # Initialize excess correlation computer
    correlator = EfficientExcessCorrelationComputer(
        model_names=models,
        n_rotation_samples=3
    )
    
    # Compute excess correlations
    print("Computing excess correlations...")
    excess_df = correlator.compute_excess_correlations(
        dataset_path=dataset_path,
        batch_size=4
    )
    
    # Analyze results
    analyzer = UniversalNeuronAnalyzer(excess_df)
    
    # Get top 10 neurons by excess correlation
    top_neurons = analyzer.identify_universal_neurons(top_k=10)
    print("\nTop 10 universal neurons:")
    print(top_neurons[['layer', 'neuron', 'excess_correlation']])
    
    # Get neurons above threshold
    threshold_neurons = analyzer.identify_universal_neurons(excess_threshold=0.05)
    print(f"\nFound {len(threshold_neurons)} neurons above threshold 0.05")
    
    return {
        'excess_correlations': excess_df,
        'top_neurons': top_neurons,
        'threshold_neurons': threshold_neurons
    }


def compare_models_example():
    """Example comparing different model families"""
    print("Running model comparison example...")
    
    # Compare different model architectures
    model_groups = [
        ["gpt2", "distilgpt2"],  # GPT-2 family
        # Add more model groups as needed
    ]
    
    results_by_group = {}
    
    for i, models in enumerate(model_groups):
        print(f"\nAnalyzing model group {i+1}: {models}")
        
        # Create dataset using first model in group
        dataset_path = create_tokenized_dataset(
            model_name=models[0],
            n_tokens=100000,
            output_dir=f"datasets_group_{i}"
        )
        
        # Run analysis
        results = run_universal_neurons_analysis(
            model_names=models,
            dataset_path=dataset_path,
            output_dir=f"results_group_{i}",
            excess_threshold=0.05,
            n_rotation_samples=3,
            batch_size=4
        )
        
        results_by_group[f"group_{i}"] = {
            'models': models,
            'results': results
        }
    
    # Compare results across groups
    print("\n" + "="*50)
    print("COMPARISON ACROSS MODEL GROUPS")
    print("="*50)
    
    for group_name, group_data in results_by_group.items():
        models = group_data['models']
        results = group_data['results']
        n_universal = len(results['universal_neurons'])
        
        print(f"{group_name}: {models}")
        print(f"  Universal neurons found: {n_universal}")
        if n_universal > 0:
            mean_excess = results['universal_neurons']['excess_correlation'].mean()
            print(f"  Mean excess correlation: {mean_excess:.4f}")
        print()
    
    return results_by_group


def main():
    parser = argparse.ArgumentParser(description="Efficient Universal Neurons Analysis")
    parser.add_argument("--test", action="store_true", help="Run quick test with smaller models")
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint to analyze")
    parser.add_argument("--compare-checkpoints", nargs='+', type=int, 
                       help="Compare multiple checkpoints (e.g., --compare-checkpoints 1000 5000 10000)")
    parser.add_argument("--excess-threshold", type=float, default=0.1,
                       help="Excess correlation threshold for identifying universal neurons")
    parser.add_argument("--memory-test", action="store_true", 
                       help="Run memory efficient test")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for processing")
    parser.add_argument("--n-rotations", type=int, default=5,
                       help="Number of rotation samples for baseline")
    parser.add_argument("--custom", action="store_true",
                       help="Run custom analysis example")
    parser.add_argument("--compare-models", action="store_true",
                       help="Run model comparison example")
    
    args = parser.parse_args()
    
    if args.memory_test:
        print("Running memory efficient test...")
        memory_efficient_test()
        
    elif args.test:
        print("Running in test mode...")
        checkpoint = None
        if args.checkpoint:
            try:
                checkpoint = int(args.checkpoint)
            except ValueError:
                checkpoint = args.checkpoint
        quick_test(checkpoint_value=checkpoint)
        
    elif args.compare_checkpoints:
        print("Running checkpoint comparison analysis...")
        analyze_checkpoint_progression(args.compare_checkpoints)
        
    elif args.custom:
        print("Running custom analysis example...")
        custom_analysis_example()
        
    elif args.compare_models:
        print("Running model comparison example...")
        compare_models_example()
        
    else:
        print("Running full analysis...")
        checkpoint = None
        if args.checkpoint:
            try:
                checkpoint = int(args.checkpoint)
            except ValueError:
                checkpoint = args.checkpoint
        full_analysis(checkpoint_value=checkpoint)


if __name__ == "__main__":
    main()