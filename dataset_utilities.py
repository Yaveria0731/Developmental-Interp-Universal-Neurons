"""
Dataset Creation and Additional Analysis Utilities - Modified for Checkpoint Support
Supporting utilities for the universal neurons analysis across training checkpoints
"""

import os
import torch
import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union
from transformer_lens import HookedTransformer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# DATASET CREATION (UNCHANGED)
# ============================================================================

def create_tokenized_dataset(
    model_name: str,
    hf_dataset: str = "monology/pile-uncopyrighted", 
    hf_split: str = "train",
    n_tokens: int = 1000000,
    ctx_len: int = 512,
    output_dir: str = "tokenized_datasets"
) -> str:
    """
    Create a tokenized dataset for analysis
    
    Args:
        model_name: Name of the model to use for tokenization
        hf_dataset: HuggingFace dataset ID
        hf_split: Dataset split to use
        n_tokens: Total number of tokens to collect
        ctx_len: Context length for sequences
        output_dir: Directory to save the dataset
    
    Returns:
        Path to the saved dataset
    """
    
    print(f"Creating tokenized dataset using {model_name}")
    print(f"Target: {n_tokens:,} tokens, {ctx_len} context length")
    
    # Load model for tokenization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HookedTransformer.from_pretrained(model_name, device=device)
    tokenizer = model.tokenizer
    
    # Adjust context length to model's capacity
    ctx_len = min(ctx_len, model.cfg.n_ctx)
    print(f"Using context length: {ctx_len}")
    
    # Load dataset
    ds_stream = datasets.load_dataset(hf_dataset, split=hf_split, streaming=True)
    
    all_tokens = []
    total = 0
    
    print("Tokenizing texts...")
    for ex in ds_stream:
        if total >= n_tokens:
            break
            
        # Tokenize with truncation
        toks = tokenizer.encode(
            ex['text'],
            truncation=True,
            max_length=ctx_len,
            return_tensors='np'
        )[0].tolist()
        
        # Add tokens if we have space
        remaining = n_tokens - total
        add_toks = toks[:remaining]
        all_tokens.extend(add_toks)
        total += len(add_toks)
        
        if total % 100000 == 0:
            print(f"Progress: {total:,}/{n_tokens:,} tokens")
    
    print(f"Collected {total:,} tokens")
    
    # Chunk into sequences
    def chunkify(tok_list, chunk_size):
        return [tok_list[i:i+chunk_size] for i in range(0, len(tok_list), chunk_size)]
    
    sequences = chunkify(all_tokens, ctx_len)
    print(f"Created {len(sequences)} sequences")
    
    # Create HuggingFace dataset
    hf_ds = datasets.Dataset.from_dict({'tokens': sequences})
    
    # Save dataset
    model_clean = model_name.replace('/', '_')
    save_path = os.path.join(output_dir, model_clean, "pile")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    hf_ds.save_to_disk(save_path)
    
    print(f"Dataset saved to: {save_path}")
    return save_path

# ============================================================================
# VISUALIZATION UTILITIES (MODIFIED FOR CHECKPOINT SUPPORT)
# ============================================================================

class UniversalNeuronVisualizer:
    """Visualization utilities for universal neuron analysis with checkpoint support"""
    
    def __init__(self, results: Dict):
        self.neuron_stats = results['neuron_stats']
        self.correlations = results['correlations']
        self.universal_neurons = results['universal_neurons']
        self.analysis = results['analysis']
        self.checkpoint = results.get('checkpoint', None)
    
    def _get_title_suffix(self) -> str:
        """Get title suffix for checkpoint-specific plots"""
        return f" (Checkpoint {self.checkpoint})" if self.checkpoint is not None else ""
    
    def plot_correlation_distribution(self, save_path: Optional[str] = None):
        """Plot distribution of correlation values"""
        
        all_corrs = []
        for corr_matrix in self.correlations.values():
            # Flatten correlation matrix and remove diagonal
            corrs_flat = corr_matrix.flatten()
            # Remove perfect correlations (diagonal elements)
            corrs_clean = corrs_flat[torch.abs(corrs_flat) < 0.99]
            all_corrs.extend(corrs_clean.tolist())
        
        plt.figure(figsize=(10, 6))
        plt.hist(all_corrs, bins=100, alpha=0.7, density=True)
        plt.axvline(0.5, color='red', linestyle='--', label='Universal threshold')
        plt.xlabel('Pearson Correlation')
        plt.ylabel('Density')
        
        title = 'Distribution of Inter-Model Neuron Correlations'
        title += self._get_title_suffix()
        plt.title(title)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_universal_properties_comparison(self, save_path: Optional[str] = None):
        """Compare properties of universal vs regular neurons"""
        
        # Combine all model stats
        all_stats = []
        for model_name, stats_df in self.neuron_stats.items():
            stats_copy = stats_df.reset_index()
            stats_copy['model'] = model_name
            all_stats.append(stats_copy)
        
        combined_stats = pd.concat(all_stats, ignore_index=True)
        
        # Mark universal neurons
        universal_neurons_set = set()
        for _, row in self.universal_neurons.iterrows():
            for model, layer, neuron in row['matching_neurons']:
                universal_neurons_set.add((model, layer, neuron))
        
        combined_stats['is_universal'] = combined_stats.apply(
            lambda row: (row['model'], row['layer'], row['neuron']) in universal_neurons_set,
            axis=1
        )
        
        # Plot comparison
        properties = ['w_in_norm', 'w_out_norm', 'l2_penalty', 'vocab_var', 'vocab_kurt']
        available_props = [p for p in properties if p in combined_stats.columns]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, prop in enumerate(available_props[:6]):
            ax = axes[i]
            
            # Plot distributions
            universal_data = combined_stats[combined_stats['is_universal']][prop]
            regular_data = combined_stats[~combined_stats['is_universal']][prop]
            
            ax.hist(regular_data, bins=50, alpha=0.7, label='Regular', density=True)
            ax.hist(universal_data, bins=50, alpha=0.7, label='Universal', density=True)
            
            ax.set_xlabel(prop.replace('_', ' ').title())
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(available_props), len(axes)):
            fig.delaxes(axes[i])
        
        title = 'Universal vs Regular Neuron Properties'
        title += self._get_title_suffix()
        fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_matrix_heatmap(self, model_pair: tuple, 
                                      layer_focus: Optional[int] = None,
                                      save_path: Optional[str] = None):
        """Plot correlation heatmap between two models"""
        
        if model_pair not in self.correlations:
            # Try reverse order
            model_pair = (model_pair[1], model_pair[0])
        
        corr_matrix = self.correlations[model_pair]
        
        if layer_focus is not None:
            # Focus on specific layer
            corr_data = corr_matrix[layer_focus, :, layer_focus, :].numpy()
            title = f'Neuron Correlations: {model_pair[0]} vs {model_pair[1]} (Layer {layer_focus})'
        else:
            # Average across all layers (diagonal layers)
            n_layers = corr_matrix.shape[0]
            corr_data = torch.stack([corr_matrix[i, :, i, :] for i in range(n_layers)]).mean(0).numpy()
            title = f'Average Neuron Correlations: {model_pair[0]} vs {model_pair[1]}'
        
        title += self._get_title_suffix()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_data, cmap='RdBu_r', center=0, 
                   xticklabels=False, yticklabels=False)
        plt.title(title)
        plt.xlabel(f'{model_pair[1]} Neurons')
        plt.ylabel(f'{model_pair[0]} Neurons')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_universal_neuron_network(self, save_path: Optional[str] = None):
        """Plot network of universal neuron connections"""
        
        fig = go.Figure()
        
        # Extract unique models
        all_models = set()
        for _, row in self.universal_neurons.iterrows():
            for model, layer, neuron in row['matching_neurons']:
                all_models.add(model)
        all_models = list(all_models)
        
        # Create network layout
        model_positions = {
            model: (i * 2, 0) for i, model in enumerate(all_models)
        }
        
        # Add nodes for each universal neuron group
        for idx, row in self.universal_neurons.iterrows():
            neurons = row['matching_neurons']
            correlations = row['correlations']
            
            # Node positions
            x_coords = []
            y_coords = []
            hover_texts = []
            
            for i, (model, layer, neuron) in enumerate(neurons):
                x, base_y = model_positions[model]
                y = base_y + (idx % 10) * 0.1  # Spread vertically
                
                x_coords.append(x)
                y_coords.append(y)
                hover_texts.append(f'{model}<br>L{layer}N{neuron}')
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                mode='markers+lines',
                name=f'Universal Group {idx}',
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>',
                line=dict(width=2, color=f'hsl({idx*30}, 70%, 50%)'),
                marker=dict(size=8)
            ))
        
        title = 'Universal Neuron Connection Network'
        title += self._get_title_suffix()
        
        fig.update_layout(
            title=title,
            xaxis_title='Models',
            showlegend=False,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        fig.show()
    
    def create_analysis_dashboard(self, save_path: str = "universal_neurons_dashboard.html"):
        """Create interactive dashboard with all visualizations"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Correlation Distribution',
                'Universal vs Regular Properties',
                'Model Correlation Matrix',
                'Universal Neuron Statistics'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Correlation distribution
        all_corrs = []
        for corr_matrix in self.correlations.values():
            corrs_flat = corr_matrix.flatten()
            corrs_clean = corrs_flat[torch.abs(corrs_flat) < 0.99]
            all_corrs.extend(corrs_clean.tolist())
        
        fig.add_trace(
            go.Histogram(x=all_corrs, name="Correlations", nbinsx=50),
            row=1, col=1
        )
        
        # 2. Universal neuron count by model
        universal_counts = self.analysis.groupby('model')['n_universal'].first()
        fig.add_trace(
            go.Bar(x=universal_counts.index, y=universal_counts.values, 
                  name="Universal Neurons"),
            row=1, col=2
        )
        
        # 3. Property comparison
        property_comparison = self.analysis.pivot(
            index='statistic', columns='model', values='difference'
        ).fillna(0)
        
        fig.add_trace(
            go.Heatmap(z=property_comparison.values,
                      x=property_comparison.columns,
                      y=property_comparison.index,
                      colorscale='RdBu', zmid=0),
            row=2, col=1
        )
        
        # 4. Correlation strength distribution
        corr_strengths = self.universal_neurons['mean_correlation']
        fig.add_trace(
            go.Histogram(x=corr_strengths, name="Mean Correlations", nbinsx=20),
            row=2, col=2
        )
        
        title = "Universal Neurons Analysis Dashboard"
        title += self._get_title_suffix()
        
        fig.update_layout(
            height=800,
            title_text=title,
            showlegend=False
        )
        
        fig.write_html(save_path)
        print(f"Dashboard saved to {save_path}")
        return fig

# ============================================================================
# CHECKPOINT COMPARISON UTILITIES (NEW)
# ============================================================================

class CheckpointComparisonVisualizer:
    """Visualize comparisons across different training checkpoints"""
    
    def __init__(self, results_by_checkpoint: Dict[Union[int, str], Dict]):
        self.results_by_checkpoint = results_by_checkpoint
        self.checkpoints = sorted(results_by_checkpoint.keys())
    
    def plot_universality_evolution(self, save_path: Optional[str] = None):
        """Plot how universality evolves across checkpoints"""
        
        checkpoint_data = []
        for checkpoint in self.checkpoints:
            results = self.results_by_checkpoint[checkpoint]
            n_universal = len(results.get('universal_neurons', []))
            
            if n_universal > 0:
                mean_correlation = results['universal_neurons']['mean_correlation'].mean()
            else:
                mean_correlation = 0.0
            
            checkpoint_data.append({
                'checkpoint': checkpoint,
                'n_universal': n_universal,
                'mean_correlation': mean_correlation
            })
        
        df = pd.DataFrame(checkpoint_data)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot number of universal neurons
        ax1.plot(df['checkpoint'], df['n_universal'], 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Training Checkpoint')
        ax1.set_ylabel('Number of Universal Neurons')
        ax1.set_title('Universal Neuron Count vs Training Progress')
        ax1.grid(True, alpha=0.3)
        
        # Plot mean correlation strength
        ax2.plot(df['checkpoint'], df['mean_correlation'], 'o-', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel('Training Checkpoint')
        ax2.set_ylabel('Mean Correlation Strength')
        ax2.set_title('Universal Correlation Strength vs Training Progress')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return df
    
    def plot_layer_distribution_evolution(self, save_path: Optional[str] = None):
        """Plot how universal neuron distribution across layers evolves"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, checkpoint in enumerate(self.checkpoints[:6]):  # Show first 6 checkpoints
            if i >= len(axes):
                break
                
            results = self.results_by_checkpoint[checkpoint]
            if 'universal_neurons' in results and len(results['universal_neurons']) > 0:
                layer_dist = results['universal_neurons']['reference_layer'].value_counts().sort_index()
                
                ax = axes[i]
                ax.bar(layer_dist.index, layer_dist.values)
                ax.set_title(f'Checkpoint {checkpoint}')
                ax.set_xlabel('Layer')
                ax.set_ylabel('Universal Neurons')
                ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(self.checkpoints), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Universal Neuron Layer Distribution Evolution', fontsize=16)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_checkpoint_comparison_dashboard(self, save_path: str = "checkpoint_comparison_dashboard.html"):
        """Create interactive dashboard comparing across checkpoints"""
        
        # Prepare data
        evolution_data = []
        for checkpoint in self.checkpoints:
            results = self.results_by_checkpoint[checkpoint]
            n_universal = len(results.get('universal_neurons', []))
            
            if n_universal > 0:
                mean_correlation = results['universal_neurons']['mean_correlation'].mean()
                # Get layer distribution
                layer_dist = results['universal_neurons']['reference_layer'].value_counts().sort_index()
            else:
                mean_correlation = 0.0
                layer_dist = pd.Series()
            
            evolution_data.append({
                'checkpoint': checkpoint,
                'n_universal': n_universal,
                'mean_correlation': mean_correlation,
                'layer_distribution': layer_dist.to_dict() if len(layer_dist) > 0 else {}
            })
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Universal Neuron Count Evolution',
                'Correlation Strength Evolution', 
                'Layer Distribution (Latest Checkpoint)',
                'Checkpoint Comparison Summary'
            ]
        )
        
        # 1. Universal neuron count evolution
        checkpoints = [d['checkpoint'] for d in evolution_data]
        n_universal_counts = [d['n_universal'] for d in evolution_data]
        
        fig.add_trace(
            go.Scatter(x=checkpoints, y=n_universal_counts, 
                      mode='lines+markers', name='Universal Count'),
            row=1, col=1
        )
        
        # 2. Correlation strength evolution
        mean_correlations = [d['mean_correlation'] for d in evolution_data]
        
        fig.add_trace(
            go.Scatter(x=checkpoints, y=mean_correlations,
                      mode='lines+markers', name='Mean Correlation'),
            row=1, col=2
        )
        
        # 3. Layer distribution for latest checkpoint
        if evolution_data and evolution_data[-1]['layer_distribution']:
            latest_layers = evolution_data[-1]['layer_distribution']
            fig.add_trace(
                go.Bar(x=list(latest_layers.keys()), y=list(latest_layers.values()),
                      name='Layer Distribution'),
                row=2, col=1
            )
        
        # 4. Summary heatmap
        summary_matrix = []
        checkpoint_labels = []
        for data in evolution_data:
            summary_matrix.append([data['n_universal'], data['mean_correlation']])
            checkpoint_labels.append(str(data['checkpoint']))
        
        if summary_matrix:
            fig.add_trace(
                go.Heatmap(z=np.array(summary_matrix).T,
                          x=checkpoint_labels,
                          y=['Universal Count', 'Mean Correlation'],
                          colorscale='Viridis'),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Checkpoint Comparison Dashboard",
            showlegend=False
        )
        
        fig.write_html(save_path)
        print(f"Checkpoint comparison dashboard saved to {save_path}")
        return fig

# ============================================================================
# ANALYSIS UTILITIES (MODIFIED TO HANDLE CHECKPOINT FILENAMES)
# ============================================================================

def load_analysis_results(results_dir: str) -> Dict:
    """Load saved analysis results"""
    
    results = {}
    
    # Load neuron stats
    neuron_stats = {}
    for file in os.listdir(results_dir):
        if file.endswith('_neuron_stats.csv'):
            # Extract model name, handling checkpoint suffixes
            model_name = file.replace('_neuron_stats.csv', '')
            if '_checkpoint_' in model_name:
                # Keep the full identifier including checkpoint
                pass
            else:
                model_name = model_name.replace('_', '/')
            
            stats_df = pd.read_csv(os.path.join(results_dir, file), index_col=[0, 1])
            neuron_stats[model_name] = stats_df
    
    results['neuron_stats'] = neuron_stats
    
    # Load correlations
    for file in os.listdir(results_dir):
        if file.startswith('correlations') and file.endswith('.pt'):
            corr_file = os.path.join(results_dir, file)
            results['correlations'] = torch.load(corr_file)
            break
    
    # Load universal neurons
    for file in os.listdir(results_dir):
        if file.startswith('universal_neurons') and file.endswith('.csv'):
            universal_file = os.path.join(results_dir, file)
            results['universal_neurons'] = pd.read_csv(universal_file)
            break
    
    # Load analysis
    for file in os.listdir(results_dir):
        if file.startswith('universal_analysis') and file.endswith('.csv'):
            analysis_file = os.path.join(results_dir, file)
            results['analysis'] = pd.read_csv(analysis_file)
            break
    
    # Try to extract checkpoint info from directory name
    if '_checkpoint_' in results_dir:
        checkpoint_part = results_dir.split('_checkpoint_')[-1]
        try:
            results['checkpoint'] = int(checkpoint_part)
        except ValueError:
            results['checkpoint'] = checkpoint_part
    
    return results

def find_similar_neurons(
    neuron_stats: pd.DataFrame,
    reference_layer: int,
    reference_neuron: int,
    similarity_metrics: List[str] = ['vocab_kurt', 'vocab_skew', 'l2_penalty'],
    top_k: int = 10
) -> pd.DataFrame:
    """Find neurons with similar properties to a reference neuron"""
    
    if (reference_layer, reference_neuron) not in neuron_stats.index:
        raise ValueError(f"Reference neuron L{reference_layer}N{reference_neuron} not found")
    
    # Get reference values
    ref_values = neuron_stats.loc[(reference_layer, reference_neuron)]
    
    # Compute similarity scores
    similarities = []
    for (layer, neuron), row in neuron_stats.iterrows():
        if layer == reference_layer and neuron == reference_neuron:
            continue  # Skip self
        
        # Compute Euclidean distance in property space
        distances = []
        for metric in similarity_metrics:
            if metric in ref_values.index and metric in row.index:
                # Normalize by standard deviation
                metric_std = neuron_stats[metric].std()
                if metric_std > 0:
                    dist = abs(ref_values[metric] - row[metric]) / metric_std
                    distances.append(dist)
        
        if distances:
            total_distance = np.mean(distances)
            similarities.append({
                'layer': layer,
                'neuron': neuron,
                'similarity_score': 1.0 / (1.0 + total_distance),  # Convert to similarity
                'distance': total_distance
            })
    
    # Sort by similarity and return top k
    similarities_df = pd.DataFrame(similarities)
    similarities_df = similarities_df.sort_values('similarity_score', ascending=False)
    
    return similarities_df.head(top_k)

def compute_neuron_importance_scores(neuron_stats: pd.DataFrame) -> pd.DataFrame:
    """Compute importance scores for neurons based on multiple factors"""
    
    importance_scores = neuron_stats.copy()
    
    # Normalize features to 0-1 scale
    for col in ['w_out_norm', 'vocab_var', 'vocab_kurt']:
        if col in importance_scores.columns:
            col_min = importance_scores[col].min()
            col_max = importance_scores[col].max()
            if col_max > col_min:
                importance_scores[f'{col}_normalized'] = (
                    importance_scores[col] - col_min
                ) / (col_max - col_min)
    
    # Compute composite importance score
    importance_factors = []
    weights = {'w_out_norm_normalized': 0.3, 'vocab_var_normalized': 0.4, 'vocab_kurt_normalized': 0.3}
    
    for factor, weight in weights.items():
        if factor in importance_scores.columns:
            importance_factors.append(weight * importance_scores[factor])
    
    if importance_factors:
        importance_scores['importance_score'] = sum(importance_factors)
    else:
        importance_scores['importance_score'] = 0.0
    
    return importance_scores.sort_values('importance_score', ascending=False)

# ============================================================================
# CHECKPOINT ANALYSIS UTILITIES (NEW)
# ============================================================================

def analyze_checkpoint_progression(results_by_checkpoint: Dict[Union[int, str], Dict]) -> pd.DataFrame:
    """Analyze how universal neurons change across checkpoints"""
    
    progression_data = []
    
    for checkpoint, results in results_by_checkpoint.items():
        universal_neurons = results.get('universal_neurons', pd.DataFrame())
        
        if len(universal_neurons) > 0:
            # Basic statistics
            n_universal = len(universal_neurons)
            mean_correlation = universal_neurons['mean_correlation'].mean()
            max_correlation = universal_neurons['mean_correlation'].max()
            
            # Layer distribution entropy (measure of how spread across layers)
            layer_counts = universal_neurons['reference_layer'].value_counts()
            layer_probs = layer_counts / layer_counts.sum()
            layer_entropy = -np.sum(layer_probs * np.log2(layer_probs + 1e-8))
            
            # Stability (how many neurons remain from previous checkpoint)
            stability = 0.0
            if len(progression_data) > 0:
                prev_neurons = set()
                for _, row in results_by_checkpoint[progression_data[-1]['checkpoint']]['universal_neurons'].iterrows():
                    prev_neurons.add((row['reference_layer'], row['reference_neuron']))
                
                curr_neurons = set()
                for _, row in universal_neurons.iterrows():
                    curr_neurons.add((row['reference_layer'], row['reference_neuron']))
                
                if len(prev_neurons) > 0:
                    stability = len(prev_neurons & curr_neurons) / len(prev_neurons)
            
        else:
            n_universal = 0
            mean_correlation = 0.0
            max_correlation = 0.0
            layer_entropy = 0.0
            stability = 0.0
        
        progression_data.append({
            'checkpoint': checkpoint,
            'n_universal': n_universal,
            'mean_correlation': mean_correlation,
            'max_correlation': max_correlation,
            'layer_entropy': layer_entropy,
            'stability': stability
        })
    
    return pd.DataFrame(progression_data).sort_values('checkpoint')

def find_persistent_universal_neurons(results_by_checkpoint: Dict[Union[int, str], Dict], 
                                    min_checkpoints: int = 2) -> pd.DataFrame:
    """Find neurons that remain universal across multiple checkpoints"""
    
    neuron_appearances = {}
    
    for checkpoint, results in results_by_checkpoint.items():
        universal_neurons = results.get('universal_neurons', pd.DataFrame())
        
        for _, row in universal_neurons.iterrows():
            neuron_id = (row['reference_layer'], row['reference_neuron'])
            
            if neuron_id not in neuron_appearances:
                neuron_appearances[neuron_id] = []
            
            neuron_appearances[neuron_id].append({
                'checkpoint': checkpoint,
                'mean_correlation': row['mean_correlation'],
                'n_models': row['n_models']
            })
    
    # Filter for persistent neurons
    persistent_neurons = []
    for neuron_id, appearances in neuron_appearances.items():
        if len(appearances) >= min_checkpoints:
            layer, neuron = neuron_id
            checkpoints_present = [app['checkpoint'] for app in appearances]
            mean_correlations = [app['mean_correlation'] for app in appearances]
            
            persistent_neurons.append({
                'layer': layer,
                'neuron': neuron,
                'checkpoints_present': checkpoints_present,
                'n_checkpoints': len(appearances),
                'mean_correlation_across_checkpoints': np.mean(mean_correlations),
                'correlation_stability': np.std(mean_correlations)
            })
    
    return pd.DataFrame(persistent_neurons).sort_values('n_checkpoints', ascending=False)

# ============================================================================
# EXAMPLE USAGE AND TESTING (MODIFIED)
# ============================================================================

def run_quick_test(checkpoint_value: Optional[Union[int, str]] = None):
    """Run a quick test with a subset of models and data"""
    
    print("Running quick test of universal neuron analysis...")
    if checkpoint_value is not None:
        print(f"Testing checkpoint: {checkpoint_value}")
    
    # Use smaller models for testing
    test_models = [
        "gpt2",
        "distilgpt2"
    ]
    
    # Create small test dataset
    dataset_path = create_tokenized_dataset(
        model_name="gpt2",
        n_tokens=50000,  # Small for testing
        ctx_len=256,
        output_dir="test_datasets"
    )
    
    # Run analysis
    from universal_neurons_pipeline import run_universal_neurons_analysis
    
    output_dir = "test_results"
    if checkpoint_value is not None:
        output_dir = f"{output_dir}_checkpoint_{checkpoint_value}"
    
    results = run_universal_neurons_analysis(
        model_names=test_models,
        dataset_path=dataset_path,
        output_dir=output_dir,
        correlation_threshold=0.3,  # Lower threshold for testing
        min_models=2,
        checkpoint_value=checkpoint_value
    )
    
    # Create visualizations
    visualizer = UniversalNeuronVisualizer(results)
    visualizer.plot_correlation_distribution()
    
    checkpoint_suffix = f"_checkpoint_{checkpoint_value}" if checkpoint_value is not None else ""
    visualizer.create_analysis_dashboard(f"test_dashboard{checkpoint_suffix}.html")
    
    print(f"Test completed! Check test_dashboard{checkpoint_suffix}.html")
    return results

if __name__ == "__main__":
    # Example: Create dataset for Stanford models
    print("Creating tokenized dataset...")
    
    dataset_path = create_tokenized_dataset(
        model_name="stanford-crfm/alias-gpt2-small-x21",
        n_tokens=2000000,  # 2M tokens
        ctx_len=512,
        output_dir="datasets"
    )
    
    print(f"Dataset created at: {dataset_path}")
    print("You can now run the main analysis pipeline with checkpoint support!")