"""
Universal Neurons Analysis - Clean Implementation with Memory-Efficient Excess Correlation
Based on the universal-neurons-new methodology for identifying universal neurons.
"""

import os
import torch
import datasets
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from transformer_lens import HookedTransformer
from torch.utils.data import DataLoader
import einops
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class MemoryEfficientExcessCorrelationComputer:
    """
    Memory-efficient excess correlation computer that processes each neuron individually.
    Implements the exact formula from the Universal Neurons paper:
    ϱi = (1/|M|) * Σ_m [max_j ρ^{a,m}_{i,j} - max_j ρ̄^{a,m}_{i,j}]
    """
    
    def __init__(self, model_names: List[str], device: str = "cuda", 
                 checkpoint_value: Optional[Union[int, str]] = None,
                 n_rotation_samples: int = 5):
        self.model_names = model_names
        self.device = device if torch.cuda.is_available() else "cpu"
        self.checkpoint_value = checkpoint_value
        self.n_rotation_samples = n_rotation_samples
        self.models = {}  # Lazy loading
    
    def _get_model_identifier(self, model_name: str) -> str:
        """Get model identifier including checkpoint info"""
        if self.checkpoint_value is not None:
            return f"{model_name}_checkpoint_{self.checkpoint_value}"
        return model_name
    
    def _load_model(self, model_name: str):
        """Load a single model with checkpoint support"""
        model_id = self._get_model_identifier(model_name)
        if model_id not in self.models:
            print(f"Loading {model_name}...")
            if self.checkpoint_value is not None:
                model = HookedTransformer.from_pretrained(
                    model_name, 
                    device=self.device, 
                    checkpoint_value=self.checkpoint_value
                )
            else:
                model = HookedTransformer.from_pretrained(model_name, device=self.device)
            
            model.eval()
            self.models[model_id] = model
            torch.set_grad_enabled(False)
    
    def generate_random_rotation_matrix(self, d_mlp: int) -> torch.Tensor:
        """Generate a random orthogonal rotation matrix using QR decomposition"""
        random_matrix = torch.randn(d_mlp, d_mlp, dtype=torch.float32)
        Q, R = torch.linalg.qr(random_matrix)
        if torch.det(Q) < 0:
            Q[:, 0] *= -1
        return Q.to(self.device)
    
    def get_activations(self, model, inputs, target_layer: Optional[int] = None, 
                       target_neuron: Optional[int] = None):
        """Get MLP activations - optionally for specific neuron"""
        hooks = []
        
        def save_activation_hook(tensor, hook):
            hook.ctx['activation'] = tensor.detach()
        
        if target_layer is not None:
            hooks = [(f'blocks.{target_layer}.mlp.hook_post', save_activation_hook)]
        else:
            hooks = [(f'blocks.{layer}.mlp.hook_post', save_activation_hook) 
                    for layer in range(model.cfg.n_layers)]
        
        with torch.no_grad():
            model.run_with_hooks(inputs, fwd_hooks=hooks)
        
        if target_layer is not None:
            activation = model.hook_dict[f'blocks.{target_layer}.mlp.hook_post'].ctx['activation']
            if target_neuron is not None:
                activation = activation[:, :, target_neuron]
            model.reset_hooks()
            return activation
        else:
            activations = torch.stack([
                model.hook_dict[f'blocks.{layer}.mlp.hook_post'].ctx['activation'] 
                for layer in range(model.cfg.n_layers)
            ])
            model.reset_hooks()
            return activations
    
    def compute_pearson_correlation(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute Pearson correlation between two 1D tensors"""
        if len(x) < 2 or len(y) < 2:
            return 0.0
        
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]
        
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        
        numerator = (x_centered * y_centered).sum()
        denominator = torch.sqrt((x_centered**2).sum() * (y_centered**2).sum())
        
        if denominator == 0:
            return 0.0
        
        corr = numerator / denominator
        return corr.item() if not torch.isnan(corr) else 0.0
    
    def compute_excess_correlations_for_all_neurons(self, dataset_path: str, 
                                                   batch_size: int = 8) -> pd.DataFrame:
        """
        Memory-efficient computation of excess correlations for all neurons.
        Returns a DataFrame with excess correlation scores for each neuron.
        """
        print("Computing excess correlations using memory-efficient method...")
        
        # Load all models
        for model_name in self.model_names:
            self._load_model(model_name)
        
        reference_model_name = self.model_names[0]
        reference_model_id = self._get_model_identifier(reference_model_name)
        reference_model = self.models[reference_model_id]
        
        # Load dataset
        tokenized_dataset = datasets.load_from_disk(dataset_path)
        if hasattr(tokenized_dataset, 'column_names') and 'tokens' in tokenized_dataset.column_names:
            dataset_to_use = tokenized_dataset['tokens']
        else:
            dataset_to_use = tokenized_dataset
        
        def collate_fn(batch):
            if isinstance(batch[0], list):
                sequences = batch
            elif isinstance(batch[0], dict) and 'tokens' in batch[0]:
                sequences = [item['tokens'] for item in batch]
            else:
                sequences = batch
            
            max_len = max(len(seq) for seq in sequences)
            padded = [seq + [0] * (max_len - len(seq)) for seq in sequences]
            return torch.tensor(padded, dtype=torch.long)
        
        dataloader = DataLoader(
            dataset_to_use, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
        
        n_layers = reference_model.cfg.n_layers
        d_mlp = reference_model.cfg.d_mlp
        
        # Pre-generate rotation matrices for all other models
        print("Pre-generating rotation matrices...")
        rotation_matrices = {}
        for model_name in self.model_names[1:]:
            model_id = self._get_model_identifier(model_name)
            model = self.models[model_id]
            rotation_matrices[model_id] = {}
            for layer in range(model.cfg.n_layers):
                rotation_matrices[model_id][layer] = []
                for _ in range(self.n_rotation_samples):
                    rot_matrix = self.generate_random_rotation_matrix(model.cfg.d_mlp)
                    rotation_matrices[model_id][layer].append(rot_matrix)
        
        excess_correlation_scores = []
        
        # Process each neuron in reference model
        print(f"Processing {n_layers * d_mlp} neurons...")
        for layer in tqdm(range(n_layers), desc="Processing layers"):
            for neuron in tqdm(range(d_mlp), desc=f"Layer {layer} neurons", leave=False):
                
                # Collect activations for this specific neuron across all batches
                reference_activations = []
                other_model_activations = {model_id: [] for model_id in self.models.keys() 
                                         if model_id != reference_model_id}
                
                for batch in dataloader:
                    batch = batch.to(self.device)
                    
                    # Get reference neuron activation
                    ref_act = self.get_activations(reference_model, batch, 
                                                 target_layer=layer, target_neuron=neuron)
                    ref_act = ref_act.flatten()
                    
                    # Get all activations from other models
                    for model_name in self.model_names[1:]:
                        model_id = self._get_model_identifier(model_name)
                        model = self.models[model_id]
                        other_acts = self.get_activations(model, batch)
                        other_acts = einops.rearrange(other_acts, 'l b s d -> l d (b s)')
                        other_model_activations[model_id].append(other_acts.cpu())
                    
                    # Apply mask and store
                    valid_mask = (batch.flatten() != 0)
                    if valid_mask.sum() > 0:
                        reference_activations.append(ref_act[valid_mask].cpu())
                
                if not reference_activations:
                    continue
                
                # Concatenate all reference activations
                ref_all = torch.cat(reference_activations, dim=0)
                
                # Compute excess correlation for this neuron
                excess_correlations_per_model = []
                
                for model_name in self.model_names[1:]:
                    model_id = self._get_model_identifier(model_name)
                    model = self.models[model_id]
                    
                    # Concatenate other model activations
                    other_all = torch.cat(other_model_activations[model_id], dim=-1)
                    
                    # Compute regular correlations (find max)
                    max_regular_corr = -1.0
                    for other_layer in range(model.cfg.n_layers):
                        for other_neuron in range(model.cfg.d_mlp):
                            other_neuron_acts = other_all[other_layer, other_neuron]
                            corr = self.compute_pearson_correlation(ref_all, other_neuron_acts)
                            max_regular_corr = max(max_regular_corr, corr)
                    
                    # Compute baseline correlations with rotations (find max)
                    max_baseline_corrs = []
                    for rotation_idx in range(self.n_rotation_samples):
                        max_baseline_corr = -1.0
                        for other_layer in range(model.cfg.n_layers):
                            # Apply rotation
                            rotation_matrix = rotation_matrices[model_id][other_layer][rotation_idx]
                            rotated_acts = rotation_matrix @ other_all[other_layer]
                            
                            for other_neuron in range(model.cfg.d_mlp):
                                rotated_neuron_acts = rotated_acts[other_neuron]
                                corr = self.compute_pearson_correlation(ref_all, rotated_neuron_acts)
                                max_baseline_corr = max(max_baseline_corr, corr)
                        
                        max_baseline_corrs.append(max_baseline_corr)
                    
                    # Average baseline across rotations
                    mean_max_baseline = np.mean(max_baseline_corrs)
                    
                    # Excess correlation for this model pair
                    excess_corr = max_regular_corr - mean_max_baseline
                    excess_correlations_per_model.append(excess_corr)
                
                # Average excess correlation across all other models (paper formula)
                mean_excess_correlation = np.mean(excess_correlations_per_model)
                
                excess_correlation_scores.append({
                    'layer': layer,
                    'neuron': neuron,
                    'excess_correlation': mean_excess_correlation,
                    'n_models_compared': len(excess_correlations_per_model),
                    'excess_correlations_per_model': excess_correlations_per_model
                })
                
                # Periodic cleanup
                if (neuron + 1) % 50 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Convert to DataFrame
        excess_df = pd.DataFrame(excess_correlation_scores)
        excess_df.set_index(['layer', 'neuron'], inplace=True)
        
        print(f"Computed excess correlations for {len(excess_df)} neurons")
        print(f"Excess correlation range: {excess_df['excess_correlation'].min():.4f} to {excess_df['excess_correlation'].max():.4f}")
        
        return excess_df


class UniversalNeuronAnalyzer:
    """Identify and analyze universal neurons using excess correlation"""
    
    def __init__(self, excess_correlation_df: pd.DataFrame):
        self.excess_correlation_df = excess_correlation_df
    
    def identify_universal_neurons(self, excess_threshold: float = 0.1, 
                                 top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Identify universal neurons using excess correlation threshold or top-k selection.
        """
        if top_k is not None:
            # Return top k neurons by excess correlation
            universal_neurons = self.excess_correlation_df.nlargest(top_k, 'excess_correlation').reset_index()
            print(f"Selected top {top_k} neurons by excess correlation")
        else:
            # Filter by threshold
            universal_mask = self.excess_correlation_df['excess_correlation'] >= excess_threshold
            universal_neurons = self.excess_correlation_df[universal_mask].reset_index()
            print(f"Found {len(universal_neurons)} neurons with excess correlation >= {excess_threshold}")
        
        if len(universal_neurons) > 0:
            print(f"Universal neurons statistics:")
            print(f"  Mean excess correlation: {universal_neurons['excess_correlation'].mean():.4f}")
            print(f"  Range: {universal_neurons['excess_correlation'].min():.4f} to {universal_neurons['excess_correlation'].max():.4f}")
        
        return universal_neurons


class NeuronStatsGenerator:
    """Generate neuron statistics for analysis"""
    
    def __init__(self, model_name: str, device: str = "cuda", 
                 checkpoint_value: Optional[Union[int, str]] = None):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.checkpoint_value = checkpoint_value
        self.model = None
    
    def _load_model(self):
        """Load model lazily"""
        if self.model is None:
            if self.checkpoint_value is not None:
                self.model = HookedTransformer.from_pretrained(
                    self.model_name, device=self.device, checkpoint_value=self.checkpoint_value
                )
            else:
                self.model = HookedTransformer.from_pretrained(self.model_name, device=self.device)
            self.model.eval()
            torch.set_grad_enabled(False)
    
    def compute_neuron_stats(self) -> pd.DataFrame:
        """Compute basic neuron statistics"""
        self._load_model()
        
        # Weight statistics
        W_in = einops.rearrange(self.model.W_in, 'l d n -> l n d')
        W_out = self.model.W_out
        
        W_in_norms = torch.norm(W_in, dim=-1)
        W_out_norms = torch.norm(W_out, dim=-1)
        l2_penalty = W_in_norms**2 + W_out_norms**2
        
        # Vocab composition statistics
        W_U = self.model.W_U / self.model.W_U.norm(dim=0, keepdim=True)
        
        stats_list = []
        for layer in range(self.model.cfg.n_layers):
            w_out = self.model.W_out[layer]
            w_out_norm = w_out / w_out.norm(dim=1)[:, None]
            vocab_cosines = w_out_norm @ W_U
            
            for neuron in range(self.model.cfg.d_mlp):
                stats_list.append({
                    'layer': layer,
                    'neuron': neuron,
                    'w_in_norm': W_in_norms[layer, neuron].item(),
                    'w_out_norm': W_out_norms[layer, neuron].item(),
                    'l2_penalty': l2_penalty[layer, neuron].item(),
                    'vocab_var': vocab_cosines[neuron].var().item(),
                    'vocab_kurt': ((vocab_cosines[neuron] - vocab_cosines[neuron].mean()) ** 4).mean().item() / vocab_cosines[neuron].var().item() ** 2
                })
        
        stats_df = pd.DataFrame(stats_list)
        stats_df.set_index(['layer', 'neuron'], inplace=True)
        return stats_df


class UniversalNeuronVisualizer:
    """Create visualizations for universal neuron analysis"""
    
    def __init__(self, excess_correlation_df: pd.DataFrame, universal_neurons_df: pd.DataFrame,
                 neuron_stats: Optional[Dict[str, pd.DataFrame]] = None):
        self.excess_correlation_df = excess_correlation_df
        self.universal_neurons_df = universal_neurons_df
        self.neuron_stats = neuron_stats or {}
    
    def plot_excess_correlation_distribution(self, save_path: Optional[str] = None):
        """Plot distribution of excess correlation values"""
        plt.figure(figsize=(10, 6))
        
        # Plot histogram
        excess_values = self.excess_correlation_df['excess_correlation'].values
        plt.hist(excess_values, bins=50, alpha=0.7, density=True, label='All neurons')
        
        # Mark universal neurons if any
        if len(self.universal_neurons_df) > 0:
            universal_values = self.universal_neurons_df['excess_correlation'].values
            plt.hist(universal_values, bins=20, alpha=0.8, density=True, 
                    label='Universal neurons', color='red')
        
        plt.axvline(0, color='black', linestyle='--', alpha=0.5, label='Zero excess')
        plt.xlabel('Excess Correlation')
        plt.ylabel('Density')
        plt.title('Distribution of Excess Correlation Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_universal_neurons_by_layer(self, save_path: Optional[str] = None):
        """Plot distribution of universal neurons across layers"""
        if len(self.universal_neurons_df) == 0:
            print("No universal neurons to plot")
            return
        
        plt.figure(figsize=(12, 6))
        
        layer_counts = self.universal_neurons_df['layer'].value_counts().sort_index()
        plt.bar(layer_counts.index, layer_counts.values, alpha=0.7)
        plt.xlabel('Layer')
        plt.ylabel('Number of Universal Neurons')
        plt.title('Universal Neurons Distribution Across Layers')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_excess_correlation_vs_properties(self, model_stats: pd.DataFrame, 
                                            save_path: Optional[str] = None):
        """Plot excess correlation vs neuron properties"""
        # Merge excess correlation with neuron stats
        merged_df = self.excess_correlation_df.reset_index().merge(
            model_stats.reset_index(), on=['layer', 'neuron'], how='left'
        )
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        properties = ['w_out_norm', 'l2_penalty', 'vocab_var', 'vocab_kurt']
        for i, prop in enumerate(properties):
            if prop in merged_df.columns:
                ax = axes[i // 2, i % 2]
                ax.scatter(merged_df[prop], merged_df['excess_correlation'], 
                          alpha=0.3, s=1)
                ax.set_xlabel(prop.replace('_', ' ').title())
                ax.set_ylabel('Excess Correlation')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_tokenized_dataset(model_name: str, hf_dataset: str = "monology/pile-uncopyrighted", 
                           n_tokens: int = 1000000, ctx_len: int = 512, 
                           output_dir: str = "datasets") -> str:
    """Create a tokenized dataset for analysis"""
    print(f"Creating tokenized dataset using {model_name}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HookedTransformer.from_pretrained(model_name, device=device)
    tokenizer = model.tokenizer
    
    ctx_len = min(ctx_len, model.cfg.n_ctx)
    print(f"Using context length: {ctx_len}")
    
    ds_stream = datasets.load_dataset(hf_dataset, split="train", streaming=True)
    
    all_tokens = []
    total = 0
    
    for ex in ds_stream:
        if total >= n_tokens:
            break
        
        toks = tokenizer.encode(ex['text'], truncation=True, max_length=ctx_len, 
                               return_tensors='np')[0].tolist()
        
        remaining = n_tokens - total
        add_toks = toks[:remaining]
        all_tokens.extend(add_toks)
        total += len(add_toks)
        
        if total % 100000 == 0:
            print(f"Progress: {total:,}/{n_tokens:,} tokens")
    
    # Chunk into sequences
    sequences = [all_tokens[i:i+ctx_len] for i in range(0, len(all_tokens), ctx_len)]
    hf_ds = datasets.Dataset.from_dict({'tokens': sequences})
    
    # Save dataset
    model_clean = model_name.replace('/', '_')
    save_path = os.path.join(output_dir, model_clean, "pile")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    hf_ds.save_to_disk(save_path)
    
    print(f"Dataset saved to: {save_path}")
    return save_path


def run_universal_neurons_analysis(model_names: List[str], dataset_path: str,
                                 output_dir: str = "results", 
                                 excess_threshold: float = 0.1,
                                 checkpoint_value: Optional[Union[int, str]] = None,
                                 top_k: Optional[int] = None,
                                 n_rotation_samples: int = 5) -> Dict:
    """
    Run complete universal neurons analysis using excess correlation method.
    """
    
    if checkpoint_value is not None:
        output_dir = f"{output_dir}_checkpoint_{checkpoint_value}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("UNIVERSAL NEURONS ANALYSIS - EXCESS CORRELATION METHOD")
    if checkpoint_value is not None:
        print(f"CHECKPOINT: {checkpoint_value}")
    print("=" * 60)
    
    # Step 1: Compute excess correlations
    print("\nStep 1: Computing excess correlations...")
    correlator = MemoryEfficientExcessCorrelationComputer(
        model_names, checkpoint_value=checkpoint_value, 
        n_rotation_samples=n_rotation_samples
    )
    
    excess_correlation_df = correlator.compute_excess_correlations_for_all_neurons(dataset_path)
    
    # Save excess correlation scores
    excess_file = os.path.join(output_dir, "excess_correlation_scores.csv")
    excess_correlation_df.to_csv(excess_file)
    print(f"Saved excess correlation scores to {excess_file}")
    
    # Step 2: Identify universal neurons
    print("\nStep 2: Identifying universal neurons...")
    analyzer = UniversalNeuronAnalyzer(excess_correlation_df)
    universal_neurons_df = analyzer.identify_universal_neurons(
        excess_threshold=excess_threshold, top_k=top_k
    )
    
    # Save universal neurons
    universal_file = os.path.join(output_dir, "universal_neurons.csv")
    universal_neurons_df.to_csv(universal_file, index=False)
    print(f"Saved universal neurons to {universal_file}")
    
    # Step 3: Generate neuron statistics for the first model
    print("\nStep 3: Computing neuron statistics...")
    stats_generator = NeuronStatsGenerator(model_names[0], checkpoint_value=checkpoint_value)
    neuron_stats_df = stats_generator.compute_neuron_stats()
    
    stats_file = os.path.join(output_dir, "neuron_stats.csv")
    neuron_stats_df.to_csv(stats_file)
    print(f"Saved neuron statistics to {stats_file}")
    
    # Step 4: Create visualizations
    print("\nStep 4: Creating visualizations...")
    visualizer = UniversalNeuronVisualizer(
        excess_correlation_df, universal_neurons_df, {model_names[0]: neuron_stats_df}
    )
    
    plots_dir = Path(output_dir) / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    visualizer.plot_excess_correlation_distribution(
        save_path=plots_dir / 'excess_correlation_distribution.png'
    )
    visualizer.plot_universal_neurons_by_layer(
        save_path=plots_dir / 'universal_neurons_by_layer.png'
    )
    visualizer.plot_excess_correlation_vs_properties(
        neuron_stats_df, save_path=plots_dir / 'excess_correlation_vs_properties.png'
    )
    
    # Summary
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"Models analyzed: {len(model_names)}")
    print(f"Total neurons: {len(excess_correlation_df)}")
    print(f"Universal neurons found: {len(universal_neurons_df)}")
    print(f"Excess threshold used: {excess_threshold}")
    if top_k:
        print(f"Top-k selection: {top_k}")
    print(f"Results saved to: {output_dir}")
    
    return {
        'excess_correlation_scores': excess_correlation_df,
        'universal_neurons': universal_neurons_df,
        'neuron_stats': neuron_stats_df,
        'checkpoint': checkpoint_value
    }


if __name__ == "__main__":
    # Example usage
    models = [
        "stanford-crfm/alias-gpt2-small-x21",
        "stanford-crfm/battlestar-gpt2-small-x49",
        "stanford-crfm/caprica-gpt2-small-x81"
    ]
    
    # Create dataset
    dataset_path = create_tokenized_dataset(
        model_name=models[0],
        n_tokens=500000,  # Reduced for testing
        output_dir="datasets"
    )
    
    # Run analysis
    results = run_universal_neurons_analysis(
        model_names=models,
        dataset_path=dataset_path,
        excess_threshold=0.05,  # Lower threshold for testing
        n_rotation_samples=3    # Fewer samples for speed
    )
    
    print("Analysis complete! Check the results directory for outputs.")
