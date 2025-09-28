"""
Universal Neurons Analysis - Memory-Efficient Implementation with Streaming Correlation
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


class StreamingCorrelationComputer:
    """Compute Pearson correlation incrementally without storing all data"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.reset()
    
    def reset(self):
        self.n = 0
        self.sum_x = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.sum_y = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.sum_xx = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.sum_yy = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.sum_xy = torch.tensor(0.0, dtype=torch.float64, device=self.device)
    
    def update(self, x: torch.Tensor, y: torch.Tensor):
        """Update correlation statistics with new data batch"""
        x = x.to(torch.float64).to(self.device)
        y = y.to(torch.float64).to(self.device)
        
        batch_size = x.numel()
        self.n += batch_size
        
        self.sum_x += x.sum()
        self.sum_y += y.sum()
        self.sum_xx += (x ** 2).sum()
        self.sum_yy += (y ** 2).sum()
        self.sum_xy += (x * y).sum()
    
    def correlation(self) -> float:
        """Compute final correlation coefficient"""
        if self.n < 2:
            return 0.0
        
        mean_x = self.sum_x / self.n
        mean_y = self.sum_y / self.n
        
        numerator = self.sum_xy - self.n * mean_x * mean_y
        denom_x = self.sum_xx - self.n * mean_x ** 2
        denom_y = self.sum_yy - self.n * mean_y ** 2
        
        denominator = torch.sqrt(denom_x * denom_y)
        
        if denominator == 0:
            return 0.0
        
        corr = numerator / denominator
        return corr.item() if not torch.isnan(corr) else 0.0


class MemoryEfficientExcessCorrelationComputer:
    """
    Memory-efficient excess correlation computer using streaming computation.
    Implements the exact formula from the Universal Neurons paper:
    Ï±i = (1/|M|) * Î£_m [max_j Ï^{a,m}_{i,j} - max_j ÏÌ„^{a,m}_{i,j}]
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
    
    def generate_random_rotation_matrix(self, d_mlp: int, seed: int = None) -> torch.Tensor:
        """Generate a random orthogonal rotation matrix using QR decomposition"""
        if seed is not None:
            torch.manual_seed(seed)
        
        random_matrix = torch.randn(d_mlp, d_mlp, dtype=torch.float32)
        Q, R = torch.linalg.qr(random_matrix)
        if torch.det(Q) < 0:
            Q[:, 0] *= -1
        return Q.to(self.device)
    
    def get_activations_batch(self, model, inputs, layer: Optional[int] = None):
        """Get MLP activations for a batch of inputs"""
        hooks = []
        
        def save_activation_hook(tensor, hook):
            hook.ctx['activation'] = tensor.detach()
        
        if layer is not None:
            hooks = [(f'blocks.{layer}.mlp.hook_post', save_activation_hook)]
        else:
            hooks = [(f'blocks.{layer}.mlp.hook_post', save_activation_hook) 
                    for layer in range(model.cfg.n_layers)]
        
        with torch.no_grad():
            model.run_with_hooks(inputs, fwd_hooks=hooks)
        
        if layer is not None:
            activation = model.hook_dict[f'blocks.{layer}.mlp.hook_post'].ctx['activation']
            model.reset_hooks()
            return activation  # Shape: [batch, seq, d_mlp]
        else:
            activations = torch.stack([
                model.hook_dict[f'blocks.{layer}.mlp.hook_post'].ctx['activation'] 
                for layer in range(model.cfg.n_layers)
            ])  # Shape: [n_layers, batch, seq, d_mlp]
            model.reset_hooks()
            return activations
    
    def compute_max_correlation_for_batch(self, ref_neuron_acts: torch.Tensor, 
                                        other_model_acts: torch.Tensor) -> float:
        """Compute maximum correlation between reference neuron and all neurons in other model"""
        # ref_neuron_acts: [batch * seq] flattened
        # other_model_acts: [n_layers, batch, seq, d_mlp]
        
        max_corr = -1.0
        ref_acts_flat = ref_neuron_acts.flatten().cpu()
        
        for layer in range(other_model_acts.shape[0]):
            for neuron in range(other_model_acts.shape[-1]):
                other_neuron_acts = other_model_acts[layer, :, :, neuron].flatten().cpu()
                
                # Compute correlation using streaming approach (simplified for batch)
                if len(ref_acts_flat) != len(other_neuron_acts):
                    min_len = min(len(ref_acts_flat), len(other_neuron_acts))
                    ref_acts_flat = ref_acts_flat[:min_len]
                    other_neuron_acts = other_neuron_acts[:min_len]
                
                if len(ref_acts_flat) < 2:
                    continue
                
                corr = torch.corrcoef(torch.stack([ref_acts_flat, other_neuron_acts]))[0, 1]
                if not torch.isnan(corr):
                    max_corr = max(max_corr, corr.item())
        
        return max_corr
    
    def compute_excess_correlations_streaming(self, dataset_path: str, 
                                            batch_size: int = 8) -> pd.DataFrame:
        """
        Memory-efficient computation of excess correlations using streaming approach.
        """
        print("Computing excess correlations using streaming method...")
        
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
        
        # Initialize correlation computers for each neuron
        correlation_computers = {}
        for layer in range(n_layers):
            for neuron in range(d_mlp):
                correlation_computers[(layer, neuron)] = {
                    'regular_correlations': [],  # Store max correlations per model per batch
                    'baseline_correlations': [[] for _ in range(self.n_rotation_samples)]  # Per rotation
                }
        
        print(f"Processing {len(dataloader)} batches...")
        
        # Process each batch
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            batch = batch.to(self.device)
            
            # Create mask for valid tokens (non-padding)
            valid_mask = (batch != 0)
            
            # Get reference model activations for all layers
            ref_activations = self.get_activations_batch(reference_model, batch)
            # Shape: [n_layers, batch, seq, d_mlp]
            
            # Process each other model
            for model_name in self.model_names[1:]:
                model_id = self._get_model_identifier(model_name)
                model = self.models[model_id]
                
                # Get other model activations
                other_activations = self.get_activations_batch(model, batch)
                
                # For each reference neuron, compute correlations
                for layer in range(n_layers):
                    for neuron in range(d_mlp):
                        # Get reference neuron activations for this batch
                        ref_neuron_acts = ref_activations[layer, :, :, neuron]
                        # Apply mask and flatten
                        ref_neuron_masked = ref_neuron_acts[valid_mask]
                        
                        if len(ref_neuron_masked) < 2:
                            continue
                        
                        # Compute regular max correlation
                        max_regular_corr = self.compute_max_correlation_for_batch(
                            ref_neuron_masked, other_activations
                        )
                        correlation_computers[(layer, neuron)]['regular_correlations'].append(max_regular_corr)
                        
                        # Compute baseline correlations with rotations
                        for rot_idx in range(self.n_rotation_samples):
                            # Generate rotation matrix deterministically
                            seed = layer * 10000 + neuron * 100 + rot_idx + batch_idx
                            rotation_matrix = self.generate_random_rotation_matrix(
                                model.cfg.d_mlp, seed=seed
                            )
                            
                            # Apply rotation to other model activations
                            rotated_activations = torch.zeros_like(other_activations)
                            for l in range(other_activations.shape[0]):
                                # other_activations[l]: [batch, seq, d_mlp]
                                acts_2d = other_activations[l].view(-1, model.cfg.d_mlp)  # [batch*seq, d_mlp]
                                rotated_2d = acts_2d @ rotation_matrix.T  # [batch*seq, d_mlp]
                                rotated_activations[l] = rotated_2d.view(other_activations[l].shape)
                            
                            # Compute max correlation with rotated activations
                            max_baseline_corr = self.compute_max_correlation_for_batch(
                                ref_neuron_masked, rotated_activations
                            )
                            correlation_computers[(layer, neuron)]['baseline_correlations'][rot_idx].append(max_baseline_corr)
            
            # Clear memory periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Compute final excess correlation scores
        print("Computing final excess correlation scores...")
        excess_correlation_scores = []
        
        for layer in range(n_layers):
            for neuron in range(d_mlp):
                correlations_data = correlation_computers[(layer, neuron)]
                
                # Average regular correlations across models and batches
                regular_corrs = correlations_data['regular_correlations']
                if not regular_corrs:
                    mean_regular_corr = 0.0
                else:
                    mean_regular_corr = np.mean(regular_corrs)
                
                # Average baseline correlations across rotations and batches
                baseline_means = []
                for rot_idx in range(self.n_rotation_samples):
                    baseline_corrs = correlations_data['baseline_correlations'][rot_idx]
                    if baseline_corrs:
                        baseline_means.append(np.mean(baseline_corrs))
                
                if baseline_means:
                    mean_baseline_corr = np.mean(baseline_means)
                else:
                    mean_baseline_corr = 0.0
                
                # Excess correlation
                excess_correlation = mean_regular_corr - mean_baseline_corr
                
                excess_correlation_scores.append({
                    'layer': layer,
                    'neuron': neuron,
                    'excess_correlation': excess_correlation,
                    'regular_correlation': mean_regular_corr,
                    'baseline_correlation': mean_baseline_corr,
                    'n_models_compared': len(self.model_names) - 1,
                })
        
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
                                 n_rotation_samples: int = 5,
                                 batch_size: int = 4) -> Dict:  # Reduced default batch size
    """
    Run complete universal neurons analysis using memory-efficient excess correlation method.
    """
    
    if checkpoint_value is not None:
        output_dir = f"{output_dir}_checkpoint_{checkpoint_value}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("UNIVERSAL NEURONS ANALYSIS - MEMORY-EFFICIENT EXCESS CORRELATION")
    if checkpoint_value is not None:
        print(f"CHECKPOINT: {checkpoint_value}")
    print("=" * 60)
    
    # Step 1: Compute excess correlations
    print("\nStep 1: Computing excess correlations...")
    correlator = MemoryEfficientExcessCorrelationComputer(
        model_names, checkpoint_value=checkpoint_value, 
        n_rotation_samples=n_rotation_samples
    )
    
    excess_correlation_df = correlator.compute_excess_correlations_streaming(
        dataset_path, batch_size=batch_size
    )
    
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
        n_tokens=500000,  # Reduced for memory efficiency
        output_dir="datasets"
    )
    
    # Run analysis with memory-efficient settings
    results = run_universal_neurons_analysis(
        model_names=models,
        dataset_path=dataset_path,
        excess_threshold=0.05,  # Lower threshold for testing
        n_rotation_samples=3,   # Fewer samples for speed
        batch_size=4            # Smaller batch size for memory
    )
    
    print("Analysis complete! Check the results directory for outputs.")