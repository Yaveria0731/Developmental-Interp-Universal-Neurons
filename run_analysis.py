#!/usr/bin/env python3
"""
Main script for running Universal Neurons analysis.

This script replicates the core functionality from "Universal Neurons in GPT Language Models"
with support for analyzing different training checkpoints.
"""

import argparse
import os
import sys
import torch
import pandas as pd
from pathlib import Path

# Import our modules
from dataset_utilities import setup_model_and_dataset, get_model_family
from neuron_stats import compute_comprehensive_neuron_stats, save_neuron_stats, load_neuron_stats
from universal_neurons import (
    run_universal_neuron_analysis, 
    save_correlation_results,
    compare_universal_neurons_across_checkpoints
)


def run_neuron_statistics(args):
    """Run neuron statistics computation."""
    
    print("="*60)
    print("RUNNING NEURON STATISTICS ANALYSIS")
    print("="*60)
    
    # Setup model and dataset with auto_download option
    model, dataset = setup_model_and_dataset(
        model_name=args.model_name,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        checkpoint=args.checkpoint,
        device=args.device,
        auto_download=args.auto_download  # Add this
    )
    
    # Compute comprehensive neuron statistics
    print(f"\nComputing neuron statistics for {args.model_name}...")
    if args.checkpoint:
        print(f"Using checkpoint: {args.checkpoint}")
    
    neuron_df = compute_comprehensive_neuron_stats(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Save results
    save_path = save_neuron_stats(
        neuron_df=neuron_df,
        save_path=args.output_dir,
        model_name=args.model_name,
        checkpoint=args.checkpoint
    )
    
    # Print summary statistics
    print("\n" + "="*60)
    print("NEURON STATISTICS SUMMARY")
    print("="*60)
    print(f"Total neurons analyzed: {len(neuron_df)}")
    print(f"Layers: {neuron_df.index.get_level_values('layer').max() + 1}")
    print(f"Neurons per layer: {neuron_df.index.get_level_values('neuron').max() + 1}")
    
    # Key statistics
    print(f"\nKey Statistics:")
    print(f"Mean activation: {neuron_df['act_mean'].mean():.4f} ± {neuron_df['act_mean'].std():.4f}")
    print(f"Mean sparsity: {neuron_df['act_sparsity'].mean():.4f} ± {neuron_df['act_sparsity'].std():.4f}")
    print(f"Mean vocab kurtosis: {neuron_df['vocab_kurt'].mean():.4f} ± {neuron_df['vocab_kurt'].std():.4f}")
    print(f"Mean L2 penalty: {neuron_df['weight_l2_penalty'].mean():.4f} ± {neuron_df['weight_l2_penalty'].std():.4f}")
    
    # High-level insights
    high_sparsity = (neuron_df['act_sparsity'] < 0.1).sum()
    high_vocab_kurt = (neuron_df['vocab_kurt'] > 10).sum()
    
    print(f"\nSpecialized Neurons:")
    print(f"High sparsity (< 0.1): {high_sparsity} ({100*high_sparsity/len(neuron_df):.1f}%)")
    print(f"High vocab kurtosis (> 10): {high_vocab_kurt} ({100*high_vocab_kurt/len(neuron_df):.1f}%)")
    
    return save_path


def run_universal_neuron_detection(args):
    """Run universal neuron detection between two models."""
    
    print("="*60)
    print("RUNNING UNIVERSAL NEURON DETECTION")
    print("="*60)
    
    # Setup models and dataset
    print("Setting up first model...")
    model_1, dataset = setup_model_and_dataset(
        model_name=args.model_1,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        checkpoint=args.checkpoint_1,
        device=args.device
    )
    
    print("Setting up second model...")
    model_2, _ = setup_model_and_dataset(
        model_name=args.model_2,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        checkpoint=args.checkpoint_2,
        device=args.device
    )
    
    # Run universal neuron analysis
    main_corr, baseline_corr, universal_neurons, analysis = run_universal_neuron_analysis(
        model_1=model_1,
        model_2=model_2,
        dataset=dataset,
        batch_size=args.batch_size,
        threshold=args.threshold,
        baseline=args.baseline,
        device=args.device,
        save_path=args.output_dir
    )
    
    # Save results
    universal_df, analysis_data = save_correlation_results(
        correlation_matrix=main_corr,
        baseline_matrix=baseline_corr,
        save_path=args.output_dir,
        model_1_name=args.model_1,
        model_2_name=args.model_2,
        dataset_name=args.dataset,
        checkpoint_1=args.checkpoint_1,
        checkpoint_2=args.checkpoint_2
    )
    
    # Print detailed results
    print("\n" + "="*60)
    print("UNIVERSAL NEURON DETECTION RESULTS")
    print("="*60)
    
    if len(universal_neurons) > 0:
        print(f"Universal neurons found: {len(universal_neurons)}")
        print(f"\nTop 10 universal neurons:")
        sorted_neurons = sorted(universal_neurons, key=lambda x: x[2], reverse=True)
        for i, (layer, neuron, excess_corr, max_corr) in enumerate(sorted_neurons[:10]):
            print(f"  {i+1}. Layer {layer}, Neuron {neuron}: excess={excess_corr:.4f}, max_corr={max_corr:.4f}")
        
        # Layer distribution
        layer_counts = {}
        for layer, neuron, _, _ in universal_neurons:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        
        print(f"\nDistribution by layer:")
        for layer in sorted(layer_counts.keys()):
            print(f"  Layer {layer}: {layer_counts[layer]} neurons")
    else:
        print("No universal neurons found above threshold.")
    
    return universal_df, analysis_data


def run_checkpoint_comparison(args):
    """Run comparison across multiple checkpoints."""
    
    print("="*60)
    print("RUNNING CHECKPOINT COMPARISON")
    print("="*60)
    
    checkpoints = [int(x) for x in args.checkpoints.split(',')]
    
    results = compare_universal_neurons_across_checkpoints(
        model_name=args.model_name,
        dataset=args.dataset,
        checkpoints=checkpoints,
        save_path=args.output_dir
    )
    
    # Analyze results across checkpoints
    print("\n" + "="*60)
    print("CHECKPOINT COMPARISON RESULTS")
    print("="*60)
    
    for comparison_name, data in results.items():
        universal_neurons = data['universal_neurons']
        analysis = data['analysis']
        
        print(f"\n{comparison_name}:")
        print(f"  Universal neurons: {len(universal_neurons)}")
        print(f"  Mean excess correlation: {analysis['excess_correlation_stats']['mean']:.4f}")
        print(f"  Universal percentage (>0.5): {analysis['universal_counts'][0.5]['percentage']:.2f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Universal Neurons Analysis - Replicate core functionality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode selection
    parser.add_argument(
        '--mode', 
        choices=['neuron_stats', 'universal_detection', 'checkpoint_comparison'],
        required=True,
        help='Analysis mode to run'
    )
    
    # Model arguments
    parser.add_argument('--model_name', type=str, help='Model name (for neuron_stats and checkpoint_comparison)')
    parser.add_argument('--model_1', type=str, help='First model name (for universal_detection)')
    parser.add_argument('--model_2', type=str, help='Second model name (for universal_detection)')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint', type=int, help='Training checkpoint for single model analysis')
    parser.add_argument('--checkpoint_1', type=int, help='Training checkpoint for first model')
    parser.add_argument('--checkpoint_2', type=int, help='Training checkpoint for second model')
    parser.add_argument('--checkpoints', type=str, help='Comma-separated checkpoints for comparison (e.g., "1000,5000,10000")')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='pile', help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    
    # Universal neuron detection arguments
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for universal neuron detection')
    parser.add_argument('--baseline', choices=['rotation', 'permutation', 'gaussian'], 
                       default='rotation', help='Baseline type for correlation comparison')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--auto_download', action='store_true', 
                   help='Automatically download and tokenize dataset if not found')
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == 'neuron_stats':
        if not args.model_name:
            parser.error("--model_name is required for neuron_stats mode")
    elif args.mode == 'universal_detection':
        if not args.model_1 or not args.model_2:
            parser.error("--model_1 and --model_2 are required for universal_detection mode")
    elif args.mode == 'checkpoint_comparison':
        if not args.model_name or not args.checkpoints:
            parser.error("--model_name and --checkpoints are required for checkpoint_comparison mode")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the appropriate analysis
    try:
        if args.mode == 'neuron_stats':
            result = run_neuron_statistics(args)
            print(f"\nNeuron statistics saved to: {result}")
            
        elif args.mode == 'universal_detection':
            universal_df, analysis = run_universal_neuron_detection(args)
            print(f"\nUniversal neuron results saved to: {args.output_dir}")
            
        elif args.mode == 'checkpoint_comparison':
            results = run_checkpoint_comparison(args)
            print(f"\nCheckpoint comparison results saved to: {args.output_dir}")
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()