#!/usr/bin/env python3
"""
Example Usage: Clean Universal Neurons Analysis
Demonstrates how to use the streamlined excess correlation method.
"""

import argparse
from pathlib import Path
from universal_neurons_clean import (
    create_tokenized_dataset,
    run_universal_neurons_analysis,
    MemoryEfficientExcessCorrelationComputer,
    UniversalNeuronAnalyzer,
    UniversalNeuronVisualizer
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
    
    # Run analysis with relaxed parameters
    results = run_universal_neurons_analysis(
        model_names=test_models,
        dataset_path=dataset_path,
        output_dir="test_results",
        excess_threshold=0.02,  # Lower threshold for testing
        checkpoint_value=checkpoint_value,
        n_rotation_samples=2    # Fewer samples for speed
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
        n_tokens=2_000_000,  # 2M tokens
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
        n_rotation_samples=5
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
        n_tokens=1_000_000,
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
            n_rotation_samples=3
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


def main():
    parser = argparse.ArgumentParser(description="Universal Neurons Analysis with Excess Correlation")
    parser.add_argument("--test", action="store_true", help="Run quick test with smaller models")
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint to analyze")
    parser.add_argument("--compare-checkpoints", nargs='+', type=int, 
                       help="Compare multiple checkpoints (e.g., --compare-checkpoints 1000 5000 10000)")
    parser.add_argument("--excess-threshold", type=float, default=0.1,
                       help="Excess correlation threshold for identifying universal neurons")
    
    args = parser.parse_args()
    
    if args.test:
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
        