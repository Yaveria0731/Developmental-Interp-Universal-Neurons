"""
Universal Neurons Analysis - Repository-Compatible Implementation
Matches the exact structure and output format of the original repository.
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


class StreamingPearsonComputer:
    """
    Fast streaming correlation computer with proper device handling.
    """
    def __init__(self, model_1, model_2, device='cpu'):
        m1_layers = model_1.cfg.n_layers
        m2_layers = model_2.cfg.n_layers
        m1_dmlp = model_1.cfg.d_mlp
        m2_dmlp = model_2.cfg.d_mlp
        self.device = device

        # Initialize all tensors on the correct device
        self.m1_sum = torch.zeros(
            (m1_layers, m1_dmlp), dtype=torch.float64, device=device)
        self.m1_sum_sq = torch.zeros(
            (m1_layers, m1_dmlp), dtype=torch.float64, device=device)

        self.m2_sum = torch.zeros(
            (m2_layers, m2_dmlp), dtype=torch.float64, device=device)
        self.m2_sum_sq = torch.zeros(
            (m2_layers, m2_dmlp), dtype=torch.float64, device=device)

        self.m1_m2_sum = torch.zeros(
            (m1_layers, m1_dmlp, m2_layers, m2_dmlp),
            dtype=torch.float64, device=device
        )
        self.n = 0

    def update_correlation_data(self, batch_1_acts, batch_2_acts):
        """Update correlation statistics with new batch - fixed device handling"""
        # Ensure all tensors are on the same device
        batch_1_acts = batch_1_acts.to(self.device, dtype=torch.float64)
        batch_2_acts = batch_2_acts.to(self.device, dtype=torch.float64)
        
        for l1 in range(batch_1_acts.shape[0]):
            for l2 in range(batch_2_acts.shape[0]):
                layerwise_result = einops.einsum(
                    batch_1_acts[l1], batch_2_acts[l2], 
                    'n1 t, n2 t -> n1 n2'
                )
                self.m1_m2_sum[l1, :, l2, :] += layerwise_result

        self.m1_sum += batch_1_acts.sum(dim=-1)
        self.m1_sum_sq += (batch_1_acts**2).sum(dim=-1)
        self.m2_sum += batch_2_acts.sum(dim=-1)
        self.m2_sum_sq += (batch_2_acts**2).sum(dim=-1)

        self.n += batch_1_acts.shape[-1]

    def compute_correlation(self):
        """Compute final correlation matrix"""
        layer_correlations = []
        
        for l1 in range(self.m1_sum.shape[0]):
            numerator = self.m1_m2_sum[l1, :, :, :] - (1 / self.n) * einops.einsum(
                self.m1_sum[l1, :], self.m2_sum, 'n1, l2 n2 -> n1 l2 n2')

            m1_norm = (self.m1_sum_sq[l1, :] - (1 / self.n) * self.m1_sum[l1, :]**2)**0.5
            m2_norm = (self.m2_sum_sq - (1 / self.n) * self.m2_sum**2)**0.5

            l_correlation = numerator / einops.einsum(
                m1_norm, m2_norm, 'n1, l2 n2 -> n1 l2 n2'
            )
            layer_correlations.append(l_correlation.to(torch.float16))

        correlation = torch.stack(layer_correlations, dim=0)
        return correlation


def save_activation_hook(tensor, hook, device='cpu'):
    """Hook to save activations"""
    hook.ctx['activation'] = tensor.detach().to(torch.float16).to(device)


def get_activations(model, inputs, device='cpu', filter_padding=True):
    """
    Get MLP activations for a batch of inputs.
    Returns shape: [n_layers, n_neurons, n_tokens]
    """
    hooks = [
        (f'blocks.{layer_ix}.mlp.hook_post',
         partial(save_activation_hook, device=device))
        for layer_ix in range(model.cfg.n_layers)
    ]

    with torch.no_grad():
        model.run_with_hooks(
            inputs,
            fwd_hooks=hooks,
            stop_at_layer=model.cfg.n_layers + 1
        )
    
    activations = torch.stack([
        model.hook_dict[hook_pt[0]].ctx['activation'] for hook_pt in hooks
    ], dim=0)
    model.reset_hooks()

    # Reshape: [layers, batch, seq, neurons] -> [layers, neurons, batch*seq]
    activations = einops.rearrange(
        activations, 'l b s n -> l n (b s)')

    if filter_padding:
        # Filter out padding tokens
        pad_token_id = getattr(model.tokenizer, 'pad_token_id', 0)
        if pad_token_id is None:
            pad_token_id = 0
        
        valid_tokens = (inputs != pad_token_id).flatten()
        activations = activations[:, :, valid_tokens]

    return activations


def flatten_layers(correlation_data):
    """Flatten correlation matrix from [l1, n1, l2, n2] to [(l1*n1), (l2*n2)]"""
    return einops.rearrange(correlation_data, 'l1 n1 l2 n2 -> (l1 n1) (l2 n2)')


def summarize_correlation_matrix(correlation_matrix):
    """Compute correlation matrix summary statistics - matches repository exactly"""
    # compute distribution summary
    bin_edges = torch.linspace(-1, 1, 100)
    
    # Compute histogram using vectorized operations
    bin_counts = torch.zeros(correlation_matrix.shape[0], len(bin_edges) - 1, dtype=torch.int32)
    for i in range(correlation_matrix.shape[0]):
        bin_counts[i] = torch.histc(correlation_matrix[i], bins=len(bin_edges)-1, min=-1, max=1).to(torch.int32)

    # compute left and right tails
    max_tail_v, max_tail_ix = torch.topk(
        correlation_matrix, 50, dim=1, largest=True)
    min_tail_v, min_tail_ix = torch.topk(
        correlation_matrix, 50, dim=1, largest=False)

    max_v, max_ix = torch.max(correlation_matrix, dim=1)
    min_v, min_ix = torch.min(correlation_matrix, dim=1)

    # compute corr distribution moments
    corr_mean = correlation_matrix.mean(dim=1)
    corr_diffs = correlation_matrix - corr_mean[:, None]
    corr_var = torch.mean(torch.pow(corr_diffs, 2.0), dim=1)
    corr_std = torch.pow(corr_var, 0.5)
    corr_zscore = corr_diffs / corr_std[:, None]
    corr_skew = torch.mean(torch.pow(corr_zscore, 3.0), dim=1)
    corr_kurt = torch.mean(torch.pow(corr_zscore, 4.0), dim=1)

    correlation_summary = {
        'diag_corr': correlation_matrix.diagonal().to(torch.float16),
        'obo_corr': torch.diag(correlation_matrix, diagonal=1).to(torch.float16),
        'bin_counts': bin_counts.to(torch.int32),
        'max_corr': max_v.to(torch.float16),
        'max_corr_ix': max_ix.to(torch.int32),
        'min_corr': min_v.to(torch.float16),
        'min_corr_ix': min_ix.to(torch.int32),
        'max_tail_corr': max_tail_v.to(torch.float16),
        'max_tail_corr_ix': max_tail_ix.to(torch.int32),
        'min_tail_corr': min_tail_v.to(torch.float16),
        'min_tail_corr_ix': min_tail_ix.to(torch.int32),
        'corr_mean': corr_mean.to(torch.float16),
        'corr_var': corr_var.to(torch.float16),
        'corr_skew': corr_skew.to(torch.float16),
        'corr_kurt': corr_kurt.to(torch.float16)
    }
    return correlation_summary


class CorrelationComputer:
    """
    Step 1: Compute and save correlation matrices in repository format
    """
    
    def __init__(self, device: str = "cuda", checkpoint_value: Optional[Union[int, str]] = None):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.checkpoint_value = checkpoint_value
        self.models = {}
    
    def _load_model(self, model_name: str):
        """Load a model"""
        if model_name not in self.models:
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
            self.models[model_name] = model
            torch.set_grad_enabled(False)
    
    def generate_random_rotation_matrix(self, d_mlp: int, seed: int = None) -> torch.Tensor:
        """Generate random orthogonal rotation matrix"""
        if seed is not None:
            torch.manual_seed(seed)
        
        random_matrix = torch.randn(d_mlp, d_mlp, dtype=torch.float32, device=self.device)
        Q, R = torch.linalg.qr(random_matrix)
        
        # Ensure proper rotation (det = 1)
        if torch.det(Q) < 0:
            Q[:, 0] *= -1
        
        return Q
    
    def compute_correlation_matrix(self, model_a_name: str, model_b_name: str, 
                                 dataset_path: str, batch_size: int = 8,
                                 baseline: str = 'none') -> torch.Tensor:
        """Compute correlation matrix between two models"""
        
        # Load models
        self._load_model(model_a_name)
        self._load_model(model_b_name)
        
        model_a = self.models[model_a_name]
        model_b = self.models[model_b_name]
        
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
        
        # Set up correlation computer
        corr_computer = StreamingPearsonComputer(model_a, model_b, device=self.device)
        
        # Generate rotation matrix if needed
        rotation_matrix = None
        if baseline == 'rotation':
            rotation_matrix = self.generate_random_rotation_matrix(model_b.cfg.d_mlp)
        
        for batch in tqdm(dataloader, desc="Processing batches"):
            batch = batch.to(self.device)
            
            # Get activations
            acts_a = get_activations(model_a, batch, device=self.device)
            acts_b = get_activations(model_b, batch, device=self.device)
            
            # Apply rotation if needed
            if baseline == 'rotation' and rotation_matrix is not None:
                rotated_acts = []
                for l in range(acts_b.shape[0]):
                    rotated = torch.matmul(rotation_matrix, acts_b[l])
                    rotated_acts.append(rotated)
                acts_b = torch.stack(rotated_acts, dim=0)
            
            # Update correlation statistics
            corr_computer.update_correlation_data(acts_a, acts_b)
        
        # Compute final correlation matrix
        correlation_matrix = corr_computer.compute_correlation()
        
        # Clear GPU memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return correlation_matrix

    def run_correlation_experiment(self, model_1_name: str, model_2_name: str, 
                                 token_dataset: str, batch_size: int = 8,
                                 baseline: str = 'none', save_full_matrix: bool = True,
                                 output_dir: str = 'correlation_results'):
        """
        Run correlation experiment and save in repository format
        """
        print(f"Computing correlations: {model_1_name} vs {model_2_name} (baseline: {baseline})")
        
        # Compute correlation matrix
        correlation = self.compute_correlation_matrix(
            model_1_name, model_2_name, token_dataset, batch_size, baseline
        )
        
        # Create output directory in repository format
        save_path = os.path.join(
            output_dir,
            f'{model_1_name}+{model_2_name}',
            os.path.basename(token_dataset),
            f'pearson.{baseline}'
        )
        os.makedirs(save_path, exist_ok=True)
        
        # Save full correlation matrix if requested
        if save_full_matrix:
            torch.save(
                correlation.cpu().to(torch.float16),
                os.path.join(save_path, 'correlation.pt')
            )
        
        # Compute and save correlation summaries
        correlation_flat = flatten_layers(correlation.cpu()).to(torch.float32)
        corr_summary = summarize_correlation_matrix(correlation_flat)
        corr_summary_T = summarize_correlation_matrix(correlation_flat.T)
        
        torch.save(corr_summary, os.path.join(save_path, 'correlation_summary.pt'))
        torch.save(corr_summary_T, os.path.join(save_path, 'correlation_summary_T.pt'))
        
        print(f"Saved correlation results to {save_path}")
        return save_path


class NeuronStatsGenerator:
    """Generate neuron statistics for analysis - restored from original code"""
    
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
        """Compute basic neuron statistics - matches original implementation"""
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


class ExcessCorrelationComputer:
    """
    Step 2: Compute excess correlations from saved correlation matrices
    """
    
    def __init__(self, correlation_results_dir: str = 'correlation_results'):
        self.correlation_results_dir = correlation_results_dir
    
    def load_correlation_results(self, model_1_name: str, model_2_name: str, 
                               dataset: str, baseline: str = 'none') -> torch.Tensor:
        """Load correlation results from saved files"""
        file_path = os.path.join(
            self.correlation_results_dir,
            f'{model_1_name}+{model_2_name}',
            dataset,
            f'pearson.{baseline}',
            'correlation.pt'
        )
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Correlation file not found: {file_path}")
        
        correlation_data = torch.load(file_path, map_location='cpu')
        return correlation_data
    
    def make_correlation_result_df(self, model_a: str, model_b: str, dataset: str) -> pd.DataFrame:
        """
        Create correlation result dataframe - matches repository format exactly
        """
        # Load regular and baseline correlations
        corr_data = self.load_correlation_results(model_a, model_b, dataset, 'none')
        baseline_corr_data = self.load_correlation_results(model_a, model_b, dataset, 'rotation')
        
        n_layers_m1, n_neurons_m1, n_layers_m2, n_neurons_m2 = corr_data.shape
        
        # Flatten correlation matrices
        corr_data_flat = flatten_layers(corr_data).numpy()
        baseline_corr_data_flat = flatten_layers(baseline_corr_data).numpy()
        
        # Handle NaNs
        if np.isnan(corr_data_flat).any():
            print(f'Warning: setting {np.isnan(corr_data_flat).sum()} nans to zero')
            corr_data_flat = np.nan_to_num(corr_data_flat, nan=0.0)
        
        if np.isnan(baseline_corr_data_flat).any():
            print(f'Warning: setting {np.isnan(baseline_corr_data_flat).sum()} baseline nans to zero')
            baseline_corr_data_flat = np.nan_to_num(baseline_corr_data_flat, nan=0.0)
        
        # Compute max correlations
        max_corr = corr_data_flat.max(axis=1)
        max_corr_ix = corr_data_flat.argmax(axis=1)
        corr_data_diag = np.diag(corr_data_flat)
        
        baseline_max_corr = baseline_corr_data_flat.max(axis=1)
        baseline_max_corr_ix = baseline_corr_data_flat.argmax(axis=1)
        
        # Convert indices back to layer/neuron coordinates
        max_sim = np.unravel_index(max_corr_ix, (n_layers_m2, n_neurons_m2))
        baseline_max_sim = np.unravel_index(baseline_max_corr_ix, (n_layers_m2, n_neurons_m2))
        
        # Create dataframe with exact repository format
        corr_df = pd.DataFrame({
            'max_corr': max_corr,
            'max_sim_layer': max_sim[0],
            'max_sim_neuron': max_sim[1],
            'diag_corr': corr_data_diag,
            'baseline': baseline_max_corr,
            'baseline_layer': baseline_max_sim[0]
        }, index=pd.MultiIndex.from_product([range(n_layers_m1), range(n_neurons_m1)]))
        corr_df.index.names = ['layer', 'neuron']
        
        return corr_df

    def compute_excess_correlations_from_saved(self, model_names: List[str], 
                                             dataset: str) -> pd.DataFrame:
        """
        Compute excess correlations from saved correlation matrices
        """
        if len(model_names) < 2:
            raise ValueError("Need at least 2 models for comparison")
        
        reference_model = model_names[0]
        comparison_models = model_names[1:]
        
        print(f"Computing excess correlations from saved matrices...")
        print(f"Reference model: {reference_model}")
        print(f"Comparison models: {comparison_models}")
        
        # Collect all correlation dataframes
        all_corr_dfs = []
        for comp_model in comparison_models:
            print(f"Loading correlations: {reference_model} vs {comp_model}")
            corr_df = self.make_correlation_result_df(reference_model, comp_model, dataset)
            all_corr_dfs.append(corr_df)
        
        # Combine correlation results
        combined_corr_df = pd.concat(all_corr_dfs, keys=comparison_models)
        
        # Compute excess correlations per neuron
        excess_scores = []
        for (layer, neuron), group in combined_corr_df.groupby(level=[1, 2]):
            regular_corrs = group['max_corr'].values
            baseline_corrs = group['baseline'].values
            
            mean_regular = np.mean(regular_corrs)
            mean_baseline = np.mean(baseline_corrs)
            excess = mean_regular - mean_baseline
            
            excess_scores.append({
                'layer': layer,
                'neuron': neuron,
                'excess_correlation': excess,
                'regular_correlation': mean_regular,
                'baseline_correlation': mean_baseline,
                'n_models_compared': len(comparison_models)
            })
        
        excess_df = pd.DataFrame(excess_scores)
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
        """Identify universal neurons using excess correlation threshold or top-k selection"""
        if top_k is not None:
            universal_neurons = self.excess_correlation_df.nlargest(top_k, 'excess_correlation').reset_index()
            print(f"Selected top {top_k} neurons by excess correlation")
        else:
            universal_mask = self.excess_correlation_df['excess_correlation'] >= excess_threshold
            universal_neurons = self.excess_correlation_df[universal_mask].reset_index()
            print(f"Found {len(universal_neurons)} neurons with excess correlation >= {excess_threshold}")
        
        if len(universal_neurons) > 0:
            print(f"Universal neurons statistics:")
            print(f"  Mean excess correlation: {universal_neurons['excess_correlation'].mean():.4f}")
            print(f"  Range: {universal_neurons['excess_correlation'].min():.4f} to {universal_neurons['excess_correlation'].max():.4f}")
        
        return universal_neurons


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
    print(