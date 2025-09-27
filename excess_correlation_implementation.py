"""
Enhanced Correlation Computation with Excess Correlation Implementation
This module implements the excess correlation metric from the Universal Neurons paper
to properly identify universal neurons by comparing against a rotated baseline.
"""

import os
import torch
import datasets
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from transformer_lens import HookedTransformer
from torch.utils.data import DataLoader
import einops
from functools import partial
from tqdm import tqdm
import glob

class ExcessCorrelationComputer:
    """
    Compute excess correlations between neurons across different models.
    
    Excess correlation = mean_max_correlation - mean_max_baseline_correlation
    where baseline is computed using random rotations of neuron activations.
    """
    
    def __init__(self, model_names: List[str], device: str = "cuda", 
                 checkpoint_value: Optional[Union[int, str]] = None,
                 n_rotation_samples: int = 5):
        self.model_names = model_names
        self.requested_device = device
        self.checkpoint_value = checkpoint_value
        self.n_rotation_samples = n_rotation_samples  # Number of random rotations for baseline
        self.models = {}  # Lazy loading
    
    def _get_model_identifier(self, model_name: str) -> str:
        """Get model identifier including checkpoint info"""
        if self.checkpoint_value is not None:
            return f"{model_name}_checkpoint_{self.checkpoint_value}"
        else:
            return model_name
    
    def _load_model(self, model_name: str):
        """Load a single model"""
        model_id = self._get_model_identifier(model_name)
        if model_id not in self.models:
            print(f"Loading {model_name}...")
            if self.checkpoint_value is not None:
                model = HookedTransformer.from_pretrained(
                    model_name, 
                    device=self.requested_device, 
                    checkpoint_value=self.checkpoint_value
                )
            else:
                model = HookedTransformer.from_pretrained(model_name, device=self.requested_device)
            
            model.eval()
            self.models[model_id] = model
            torch.set_grad_enabled(False)
    
    def get_expected_excess_correlation_filename(self, model1_name: str, model2_name: str, output_dir: str) -> str:
        """Get expected filename for excess correlation between two models"""
        model1_id = self._get_model_identifier(model1_name)
        model2_id = self._get_model_identifier(model2_name)
        
        safe_model1 = model1_id.replace('/', '_').replace('-', '_')
        safe_model2 = model2_id.replace('/', '_').replace('-', '_')
        
        if self.checkpoint_value is not None:
            filename = f"excess_correlation_{safe_model1}_vs_{safe_model2}_checkpoint_{self.checkpoint_value}.pt"
        else:
            filename = f"excess_correlation_{safe_model1}_vs_{safe_model2}.pt"
        
        return os.path.join(output_dir, filename)
    
    def excess_correlation_file_exists(self, model1_name: str, model2_name: str, output_dir: str) -> bool:
        """Check if excess correlation file already exists"""
        return os.path.exists(self.get_expected_excess_correlation_filename(model1_name, model2_name, output_dir))
    
    def get_model_device(self, model):
        """Get the actual device of model parameters"""
        return next(model.parameters()).device
    
    def get_activations(self, model, inputs) -> torch.Tensor:
        """Get MLP activations for all layers"""
        hooks = []
        
        def save_activation_hook(tensor, hook):
            hook.ctx['activation'] = tensor.detach()
        
        # Set up hooks for all MLP layers
        for layer in range(model.cfg.n_layers):
            hooks.append((f'blocks.{layer}.mlp.hook_post', save_activation_hook))
        
        with torch.no_grad():
            model.run_with_hooks(inputs, fwd_hooks=hooks)
        
        # Stack activations: [n_layers, batch, seq, d_mlp]
        activations = torch.stack([
            model.hook_dict[f'blocks.{layer}.mlp.hook_post'].ctx['activation'] 
            for layer in range(model.cfg.n_layers)
        ])
        
        model.reset_hooks()
        
        # Reshape to [n_layers, d_mlp, batch*seq]
        activations = einops.rearrange(activations, 'l b s d -> l d (b s)')
        
        return activations
    
    def generate_random_rotation_matrix(self, d_mlp: int, device: torch.device) -> torch.Tensor:
        """
        Generate a random orthogonal rotation matrix using QR decomposition.
        This creates a proper rotation matrix (orthogonal with determinant 1).
        """
        # Generate random matrix
        random_matrix = torch.randn(d_mlp, d_mlp, device=device, dtype=torch.float32)
        
        # QR decomposition to get orthogonal matrix
        Q, R = torch.linalg.qr(random_matrix)
        
        # Ensure proper rotation (determinant = 1)
        # If determinant is -1, flip one column to make it +1
        if torch.det(Q) < 0:
            Q[:, 0] *= -1
        
        return Q
    
    def compute_streaming_correlation_with_baseline(self, model1_name: str, model2_name: str,
                                                  dataset_path: str, output_dir: str, 
                                                  batch_size: int = 32) -> str:
        """
        Compute both regular correlations and baseline correlations (with random rotations).
        
        The baseline is computed by rotating the activations of model2 with random orthogonal matrices
        and computing correlations with model1. This establishes what correlation we'd expect
        if there's no privileged neuron basis.
        """
        
        # Check if file already exists
        expected_file = self.get_expected_excess_correlation_filename(model1_name, model2_name, output_dir)
        if os.path.exists(expected_file):
            print(f"Excess correlation file already exists: {os.path.basename(expected_file)} - skipping computation")
            return expected_file
        
        # Load models
        self._load_model(model1_name)
        self._load_model(model2_name)
        
        model1_id = self._get_model_identifier(model1_name)
        model2_id = self._get_model_identifier(model2_name)
        
        model1 = self.models[model1_id]
        model2 = self.models[model2_id]
        
        # Get actual devices
        device1 = self.get_model_device(model1)
        device2 = self.get_model_device(model2)
        compute_device = device1 if torch.cuda.is_available() else 'cpu'
        
        print(f"Computing excess correlation between {model1_name} and {model2_name} on {compute_device}")
        
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
        
        n_layers1 = model1.cfg.n_layers
        n_layers2 = model2.cfg.n_layers
        d_mlp1 = model1.cfg.d_mlp
        d_mlp2 = model2.cfg.d_mlp
        
        # Initialize correlation computation for regular correlations
        m1_sum = torch.zeros(n_layers1, d_mlp1, dtype=torch.float64, device=compute_device)
        m1_sum_sq = torch.zeros(n_layers1, d_mlp1, dtype=torch.float64, device=compute_device)
        m2_sum = torch.zeros(n_layers2, d_mlp2, dtype=torch.float64, device=compute_device) 
        m2_sum_sq = torch.zeros(n_layers2, d_mlp2, dtype=torch.float64, device=compute_device)
        
        # Cross-correlation for regular correlations
        cross_sum = torch.zeros(n_layers1, d_mlp1, n_layers2, d_mlp2, 
                               dtype=torch.float64, device=compute_device)
        
        # Initialize for baseline correlations with multiple rotation samples
        baseline_cross_sums = []
        baseline_m2_sums = []
        baseline_m2_sum_sqs = []
        
        # Generate rotation matrices for each layer and sample
        rotation_matrices = {}
        for layer in range(n_layers2):
            rotation_matrices[layer] = []
            for sample in range(self.n_rotation_samples):
                rot_matrix = self.generate_random_rotation_matrix(d_mlp2, compute_device)
                rotation_matrices[layer].append(rot_matrix)
        
        # Initialize baseline statistics for each rotation sample
        for sample in range(self.n_rotation_samples):
            baseline_cross_sums.append(torch.zeros(n_layers1, d_mlp1, n_layers2, d_mlp2, 
                                                  dtype=torch.float64, device=compute_device))
            baseline_m2_sums.append(torch.zeros(n_layers2, d_mlp2, dtype=torch.float64, device=compute_device))
            baseline_m2_sum_sqs.append(torch.zeros(n_layers2, d_mlp2, dtype=torch.float64, device=compute_device))
        
        n_samples = 0
        
        print(f"Allocated correlation tensors on {compute_device}")
        print(f"Computing {self.n_rotation_samples} baseline rotation samples")
        
        for batch in tqdm(dataloader, desc="Computing correlations and baselines"):
            # Keep original batch for consistent masking
            original_batch = batch.clone()
            
            # Send batch to model devices
            batch1 = batch.to(device1)
            batch2 = batch.to(device2) if device1 != device2 else batch1
            
            # Get activations
            acts1 = self.get_activations(model1, batch1)  # [l1, d1, t]
            acts2 = self.get_activations(model2, batch2)  # [l2, d2, t]
            
            # Create consistent valid mask
            valid_mask = (original_batch.flatten() != 0)
            
            # Move activations to compute device
            acts1_compute = acts1.to(compute_device)
            acts2_compute = acts2.to(compute_device) 
            valid_mask_compute = valid_mask.to(compute_device)
            
            # Apply mask
            acts1_filtered = acts1_compute[:, :, valid_mask_compute]  # [l1, d1, t_valid]
            acts2_filtered = acts2_compute[:, :, valid_mask_compute]  # [l2, d2, t_valid]
            
            n_tokens = acts1_filtered.shape[-1]
            if n_tokens == 0:
                continue
                
            n_samples += n_tokens
            
            # Update statistics for regular correlations
            m1_sum += acts1_filtered.sum(dim=-1)
            m1_sum_sq += (acts1_filtered**2).sum(dim=-1)
            m2_sum += acts2_filtered.sum(dim=-1)
            m2_sum_sq += (acts2_filtered**2).sum(dim=-1)
            
            # Regular cross products
            for l1 in range(n_layers1):
                for l2 in range(n_layers2):
                    cross_sum[l1, :, l2, :] += torch.mm(
                        acts1_filtered[l1], acts2_filtered[l2].T
                    )
            
            # Compute baseline correlations with rotated activations
            for sample in range(self.n_rotation_samples):
                # Rotate activations for model2 at each layer
                acts2_rotated = torch.zeros_like(acts2_filtered)
                for l2 in range(n_layers2):
                    # Apply rotation: R @ acts2[l2]  (R is d_mlp x d_mlp, acts2[l2] is d_mlp x t_valid)
                    acts2_rotated[l2] = rotation_matrices[l2][sample] @ acts2_filtered[l2]
                
                # Update baseline statistics
                baseline_m2_sums[sample] += acts2_rotated.sum(dim=-1)
                baseline_m2_sum_sqs[sample] += (acts2_rotated**2).sum(dim=-1)
                
                # Baseline cross products
                for l1 in range(n_layers1):
                    for l2 in range(n_layers2):
                        baseline_cross_sums[sample][l1, :, l2, :] += torch.mm(
                            acts1_filtered[l1], acts2_rotated[l2].T
                        )
        
        if n_samples == 0:
            raise ValueError("No valid samples found in dataset")
        
        print("Computing final correlations...")
        
        # Compute regular correlations
        regular_correlations = torch.zeros_like(cross_sum, device=compute_device)
        
        for l1 in range(n_layers1):
            for l2 in range(n_layers2):
                # Numerator: E[XY] - E[X]E[Y]
                numerator = cross_sum[l1, :, l2, :] / n_samples - torch.outer(
                    m1_sum[l1] / n_samples, m2_sum[l2] / n_samples
                )
                
                # Denominator: sqrt(Var[X] * Var[Y])
                var1 = m1_sum_sq[l1] / n_samples - (m1_sum[l1] / n_samples)**2
                var2 = m2_sum_sq[l2] / n_samples - (m2_sum[l2] / n_samples)**2
                denominator = torch.outer(torch.sqrt(var1 + 1e-8), torch.sqrt(var2 + 1e-8))
                
                regular_correlations[l1, :, l2, :] = numerator / denominator
        
        # Compute baseline correlations for each rotation sample
        baseline_correlations = []
        
        for sample in range(self.n_rotation_samples):
            baseline_corr = torch.zeros_like(cross_sum, device=compute_device)
            
            for l1 in range(n_layers1):
                for l2 in range(n_layers2):
                    # Numerator for baseline
                    numerator = baseline_cross_sums[sample][l1, :, l2, :] / n_samples - torch.outer(
                        m1_sum[l1] / n_samples, baseline_m2_sums[sample][l2] / n_samples
                    )
                    
                    # Denominator for baseline
                    var1 = m1_sum_sq[l1] / n_samples - (m1_sum[l1] / n_samples)**2
                    var2_baseline = (baseline_m2_sum_sqs[sample][l2] / n_samples - 
                                   (baseline_m2_sums[sample][l2] / n_samples)**2)
                    denominator = torch.outer(torch.sqrt(var1 + 1e-8), torch.sqrt(var2_baseline + 1e-8))
                    
                    baseline_corr[l1, :, l2, :] = numerator / denominator
            
            baseline_correlations.append(baseline_corr.cpu())
        
        # Average baseline correlations across rotation samples
        mean_baseline_correlation = torch.stack(baseline_correlations).mean(dim=0)
        
        # Compute excess correlation = regular - baseline
        excess_correlation = regular_correlations.cpu() - mean_baseline_correlation
        
        # Save results
        print(f"Saving excess correlation to {expected_file}...")
        torch.save({
            'excess_correlation_matrix': excess_correlation,
            'regular_correlation_matrix': regular_correlations.cpu(),
            'baseline_correlation_matrix': mean_baseline_correlation,
            'baseline_samples': baseline_correlations,  # All individual samples
            'model1': model1_id,
            'model2': model2_id,
            'n_samples': n_samples,
            'n_rotation_samples': self.n_rotation_samples,
            'checkpoint': self.checkpoint_value
        }, expected_file)
        
        print(f"Excess correlation computation complete for {model1_name} vs {model2_name}")
        return expected_file
    
    def compute_all_excess_correlations(self, dataset_path: str, output_dir: str) -> List[str]:
        """Compute excess correlations between all model pairs"""
        correlation_files = []
        
        for i, model1 in enumerate(self.model_names):
            for j, model2 in enumerate(self.model_names[i:], i):
                if i == j:
                    continue  # Skip self-correlation
                
                print(f"Computing excess correlation for pair: ({model1}, {model2})")
                
                corr_file = self.compute_streaming_correlation_with_baseline(
                    model1, model2, dataset_path, output_dir
                )
                correlation_files.append(corr_file)
                
                # Force garbage collection to free memory
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return correlation_files


class ExcessCorrelationUniversalNeuronAnalyzer:
    """
    Identify universal neurons using excess correlation metric.
    
    This implements the exact formula from the paper:
    ϱi = (1/|M|) * Σ_m [max_j ρ^{a,m}_{i,j} - max_j ρ̄^{a,m}_{i,j}]
    
    Where:
    - ρ^{a,m}_{i,j} is regular correlation between neuron i in model a and neuron j in model m
    - ρ̄^{a,m}_{i,j} is baseline correlation with rotated activations
    - The excess correlation is the difference between these two maxima, averaged across models
    """
    
    def __init__(self, excess_correlation_files: List[str], neuron_stats: Dict[str, pd.DataFrame]):
        self.excess_correlation_files = excess_correlation_files
        self.neuron_stats = neuron_stats
    
    def load_excess_correlation_file(self, filepath: str) -> Dict:
        """Load a single excess correlation file"""
        return torch.load(filepath, map_location='cpu')
    
    def compute_excess_correlation_scores(self) -> pd.DataFrame:
        """
        Compute excess correlation scores for all neurons using the paper's formula.
        
        For each neuron i in reference model a, compute:
        ϱi = (1/|M|) * Σ_m [max_j ρ^{a,m}_{i,j} - max_j ρ̄^{a,m}_{i,j}]
        """
        
        # Load all excess correlation data
        excess_correlation_data = {}
        all_models = set()
        
        print("Loading excess correlation files...")
        for filepath in self.excess_correlation_files:
            corr_data = self.load_excess_correlation_file(filepath)
            model1 = corr_data['model1']
            model2 = corr_data['model2']
            all_models.update([model1, model2])
            
            # Store both regular and baseline correlations
            excess_correlation_data[(model1, model2)] = {
                'regular': corr_data['regular_correlation_matrix'],
                'baseline': corr_data['baseline_correlation_matrix'],
                'excess': corr_data['excess_correlation_matrix']
            }
        
        all_models = list(all_models)
        print(f"Found {len(all_models)} models: {all_models}")
        
        # Use first model as reference (model 'a' in the paper)
        reference_model = all_models[0]
        other_models = all_models[1:]
        
        print(f"Using {reference_model} as reference model")
        
        # Get dimensions from reference model
        ref_stats = self.neuron_stats[reference_model]
        n_layers = ref_stats.index.get_level_values('layer').max() + 1
        d_mlp = ref_stats.index.get_level_values('neuron').max() + 1
        
        excess_correlation_scores = []
        
        print("Computing excess correlation scores...")
        for layer in tqdm(range(n_layers), desc="Processing layers"):
            for neuron in range(d_mlp):
                
                # For this neuron, compute excess correlation across all other models
                excess_correlations_across_models = []
                
                for other_model in other_models:
                    # Get correlation data for this model pair
                    pair_key = (reference_model, other_model)
                    if pair_key not in excess_correlation_data:
                        pair_key = (other_model, reference_model)
                    
                    if pair_key in excess_correlation_data:
                        regular_corr = excess_correlation_data[pair_key]['regular']
                        baseline_corr = excess_correlation_data[pair_key]['baseline']
                        
                        # Extract correlations for this specific neuron
                        if pair_key[0] == reference_model:
                            # Reference model is first: [ref_layer, ref_neuron, other_layer, other_neuron]
                            neuron_regular_corrs = regular_corr[layer, neuron, :, :]
                            neuron_baseline_corrs = baseline_corr[layer, neuron, :, :]
                        else:
                            # Reference model is second: [other_layer, other_neuron, ref_layer, ref_neuron]
                            neuron_regular_corrs = regular_corr[:, :, layer, neuron]
                            neuron_baseline_corrs = baseline_corr[:, :, layer, neuron]
                        
                        # Compute max correlations as per paper formula
                        max_regular_corr = neuron_regular_corrs.max().item()
                        max_baseline_corr = neuron_baseline_corrs.max().item()
                        
                        # Excess correlation for this model pair
                        excess_corr = max_regular_corr - max_baseline_corr
                        excess_correlations_across_models.append(excess_corr)
                
                # Average excess correlation across all models (formula from paper)
                if excess_correlations_across_models:
                    mean_excess_correlation = np.mean(excess_correlations_across_models)
                    
                    excess_correlation_scores.append({
                        'layer': layer,
                        'neuron': neuron,
                        'excess_correlation': mean_excess_correlation,
                        'n_models_compared': len(excess_correlations_across_models),
                        'excess_correlations_per_model': excess_correlations_across_models
                    })
        
        excess_df = pd.DataFrame(excess_correlation_scores)
        excess_df.set_index(['layer', 'neuron'], inplace=True)
        
        print(f"Computed excess correlations for {len(excess_df)} neurons")
        print(f"Excess correlation range: {excess_df['excess_correlation'].min():.4f} to {excess_df['excess_correlation'].max():.4f}")
        
        return excess_df
    
    def identify_universal_neurons_by_excess_correlation(self, excess_threshold: float = 0.1,
                                                       top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Identify universal neurons using excess correlation threshold.
        
        Args:
            excess_threshold: Minimum excess correlation to be considered universal
            top_k: If provided, return top k neurons by excess correlation regardless of threshold
        """
        
        # Compute excess correlation scores
        excess_df = self.compute_excess_correlation_scores()
        
        if top_k is not None:
            # Return top k neurons by excess correlation
            universal_neurons = excess_df.nlargest(top_k, 'excess_correlation').reset_index()
            print(f"Selected top {top_k} neurons by excess correlation")
        else:
            # Filter by threshold
            universal_mask = excess_df['excess_correlation'] >= excess_threshold
            universal_neurons = excess_df[universal_mask].reset_index()
            print(f"Found {len(universal_neurons)} neurons with excess correlation >= {excess_threshold}")
        
        # Add additional information for each universal neuron
        enhanced_universal_neurons = []
        
        for _, row in universal_neurons.iterrows():
            layer, neuron = row['layer'], row['neuron']
            
            # Find the specific matching neurons in other models (for compatibility with downstream code)
            matching_neurons = [('reference_model', layer, neuron)]  # Placeholder - could be enhanced
            
            enhanced_universal_neurons.append({
                'reference_layer': layer,
                'reference_neuron': neuron,
                'excess_correlation': row['excess_correlation'],
                'n_models_compared': row['n_models_compared'],
                'matching_neurons': matching_neurons,  # Simplified for now
                'correlations': row['excess_correlations_per_model'],
                'mean_correlation': row['excess_correlation'],  # For backward compatibility
                'n_models': row['n_models_compared'] + 1  # +1 for reference model
            })
        
        result_df = pd.DataFrame(enhanced_universal_neurons)
        
        if len(result_df) > 0:
            print(f"Universal neurons identified:")
            print(f"  Mean excess correlation: {result_df['excess_correlation'].mean():.4f}")
            print(f"  Std excess correlation: {result_df['excess_correlation'].std():.4f}")
            print(f"  Range: {result_df['excess_correlation'].min():.4f} to {result_df['excess_correlation'].max():.4f}")
        
        return result_df
    
    def analyze_universal_properties_with_excess_correlation(self, universal_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze properties of neurons identified by excess correlation"""
        
        analysis_results = []
        
        for model_name, stats_df in self.neuron_stats.items():
            # Mark universal neurons
            is_universal = np.zeros(len(stats_df), dtype=bool)
            
            for _, row in universal_df.iterrows():
                layer, neuron = row['reference_layer'], row['reference_neuron']
                try:
                    idx = stats_df.index.get_loc((layer, neuron))
                    is_universal[idx] = True
                except KeyError:
                    continue
            
            stats_df['is_universal_excess'] = is_universal
            
            # Compute statistics
            for stat_col in ['w_in_norm', 'w_out_norm', 'l2_penalty', 'vocab_var', 'vocab_kurt']:
                if stat_col in stats_df.columns:
                    universal_mean = stats_df[is_universal][stat_col].mean() if is_universal.sum() > 0 else 0
                    regular_mean = stats_df[~is_universal][stat_col].mean() if (~is_universal).sum() > 0 else 0
                    
                    analysis_results.append({
                        'model': model_name,
                        'statistic': stat_col,
                        'universal_mean': universal_mean,
                        'regular_mean': regular_mean,
                        'difference': universal_mean - regular_mean,
                        'n_universal': is_universal.sum(),
                        'n_regular': (~is_universal).sum()
                    })
        
        return pd.DataFrame(analysis_results)


# Example usage functions

def find_existing_excess_correlation_files(output_dir: str, checkpoint_value: Optional[Union[int, str]] = None) -> List[str]:
    """Find all existing excess correlation files in output directory"""
    if checkpoint_value is not None:
        pattern = os.path.join(output_dir, f"excess_correlation_*_checkpoint_{checkpoint_value}.pt")
    else:
        pattern = os.path.join(output_dir, "excess_correlation_*.pt")
    
    correlation_files = glob.glob(pattern)
    return correlation_files


def run_excess_correlation_analysis(
    model_names: List[str],
    dataset_path: str,
    output_dir: str = "universal_neurons_results",
    excess_threshold: float = 0.1,
    checkpoint_value: Optional[Union[int, str]] = None,
    top_k: Optional[int] = None,
    n_rotation_samples: int = 5
):
    """
    Run universal neurons analysis using excess correlation metric from the paper.
    
    Args:
        model_names: List of model names to analyze
        dataset_path: Path to tokenized dataset
        output_dir: Directory to save results
        excess_threshold: Threshold for excess correlation to identify universal neurons
        checkpoint_value: Specific checkpoint to analyze
        top_k: If provided, return top k neurons regardless of threshold
        n_rotation_samples: Number of random rotations for baseline computation
    """
    
    if checkpoint_value is not None:
        output_dir = f"{output_dir}_checkpoint_{checkpoint_value}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("UNIVERSAL NEURONS ANALYSIS WITH EXCESS CORRELATION")
    if checkpoint_value is not None:
        print(f"CHECKPOINT: {checkpoint_value}")
    print("=" * 60)
    
    # Step 1: Load or compute neuron stats (reuse existing implementation)
    print("\n" + "=" * 50)
    print("STEP 1: LOADING NEURON STATISTICS")
    print("=" * 50)
    
    # Import from your existing pipeline
    from universal_neurons_pipeline import NeuronStatsGenerator, load_existing_neuron_stats
    
    neuron_stats = {}
    for model_name in model_names:
        generator = NeuronStatsGenerator(model_name, checkpoint_value=checkpoint_value)
        if generator.stats_file_exists(output_dir):
            print(f"Loading existing stats for {model_name}...")
            stats_df = generator.load_existing_stats(output_dir)
            model_id = generator._get_model_identifier(model_name)
            neuron_stats[model_id] = stats_df
        else:
            print(f"Computing stats for {model_name}...")
            stats_df = generator.generate_full_neuron_dataframe(dataset_path, output_dir)
            model_id = generator.model_identifier
            neuron_stats[model_id] = stats_df
    
    print(f"Loaded/computed stats for {len(neuron_stats)} models")
    
    # Step 2: Compute excess correlations
    print("\n" + "=" * 50)
    print("STEP 2: COMPUTING EXCESS CORRELATIONS")
    print("=" * 50)
    
    # Check for existing files
    existing_files = find_existing_excess_correlation_files(output_dir, checkpoint_value)
    print(f"Found {len(existing_files)} existing excess correlation files")
    
    # Compute missing correlations
    correlator = ExcessCorrelationComputer(
        model_names, 
        checkpoint_value=checkpoint_value,
        n_rotation_samples=n_rotation_samples
    )
    
    all_correlation_files = []
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names[i:], i):
            if i == j:
                continue
            
            expected_file = correlator.get_expected_excess_correlation_filename(model1, model2, output_dir)
            
            if os.path.exists(expected_file):
                print(f"Found existing: {os.path.basename(expected_file)}")
                all_correlation_files.append(expected_file)
            else:
                print(f"Computing excess correlation: {model1} vs {model2}")
                corr_file = correlator.compute_streaming_correlation_with_baseline(
                    model1, model2, dataset_path, output_dir
                )
                all_correlation_files.append(corr_file)
                
                # Force garbage collection
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"Total excess correlation files: {len(all_correlation_files)}")
    
    # Step 3: Identify universal neurons using excess correlation
    print("\n" + "=" * 50)
    print("STEP 3: IDENTIFYING UNIVERSAL NEURONS BY EXCESS CORRELATION")
    print("=" * 50)
    
    analyzer = ExcessCorrelationUniversalNeuronAnalyzer(all_correlation_files, neuron_stats)
    
    # Compute excess correlation scores for all neurons
    excess_scores_df = analyzer.compute_excess_correlation_scores()
    
    # Identify universal neurons
    if top_k is not None:
        print(f"Selecting top {top_k} neurons by excess correlation...")
        universal_df = analyzer.identify_universal_neurons_by_excess_correlation(top_k=top_k)
    else:
        print(f"Using excess correlation threshold: {excess_threshold}")
        universal_df = analyzer.identify_universal_neurons_by_excess_correlation(excess_threshold=excess_threshold)
    
    # Step 4: Analyze properties
    print("\n" + "=" * 50)
    print("STEP 4: ANALYZING UNIVERSAL NEURON PROPERTIES")
    print("=" * 50)
    
    analysis_df = analyzer.analyze_universal_properties_with_excess_correlation(universal_df)
    
    # Save results
    excess_filename = "excess_correlation_scores.csv"
    universal_filename = "universal_neurons_excess.csv"
    analysis_filename = "universal_analysis_excess.csv"
    
    if checkpoint_value is not None:
        excess_filename = f"excess_correlation_scores_checkpoint_{checkpoint_value}.csv"
        universal_filename = f"universal_neurons_excess_checkpoint_{checkpoint_value}.csv"
        analysis_filename = f"universal_analysis_excess_checkpoint_{checkpoint_value}.csv"
    
    excess_scores_df.to_csv(os.path.join(output_dir, excess_filename))
    universal_df.to_csv(os.path.join(output_dir, universal_filename), index=False)
    analysis_df.to_csv(os.path.join(output_dir, analysis_filename), index=False)
    
    print(f"Results saved:")
    print(f"  - {excess_filename}: All excess correlation scores")
    print(f"  - {universal_filename}: Universal neurons identified")
    print(f"  - {analysis_filename}: Property analysis")
    
    # Summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    print(f"Total neurons analyzed: {len(excess_scores_df)}")
    print(f"Universal neurons found: {len(universal_df)}")
    
    if len(universal_df) > 0:
        print(f"Excess correlation range: {universal_df['excess_correlation'].min():.4f} to {universal_df['excess_correlation'].max():.4f}")
        print(f"Mean excess correlation: {universal_df['excess_correlation'].mean():.4f}")
        
        # Layer distribution
        layer_dist = universal_df['reference_layer'].value_counts().sort_index()
        print(f"\nUniversal neurons by layer:")
        for layer, count in layer_dist.items():
            print(f"  Layer {layer}: {count} neurons")
    
    if len(excess_scores_df) > 0:
        print(f"\nOverall excess correlation statistics:")
        print(f"  Mean: {excess_scores_df['excess_correlation'].mean():.4f}")
        print(f"  Std: {excess_scores_df['excess_correlation'].std():.4f}")
        print(f"  Min: {excess_scores_df['excess_correlation'].min():.4f}")
        print(f"  Max: {excess_scores_df['excess_correlation'].max():.4f}")
        
        # Show percentiles
        percentiles = [50, 75, 90, 95, 99]
        print(f"  Percentiles:")
        for p in percentiles:
            val = excess_scores_df['excess_correlation'].quantile(p/100)
            print(f"    {p}th: {val:.4f}")
    
    return {
        'neuron_stats': neuron_stats,
        'excess_correlation_files': all_correlation_files,
        'excess_scores': excess_scores_df,
        'universal_neurons': universal_df,
        'analysis': analysis_df,
        'checkpoint': checkpoint_value
    }


# Integration with existing pipeline
def integrate_excess_correlation_into_pipeline():
    """
    Instructions for integrating excess correlation into your existing pipeline.
    
    This function provides the steps to modify your existing pipeline to use excess correlation.
    """
    
    instructions = """
    TO INTEGRATE EXCESS CORRELATION INTO YOUR EXISTING PIPELINE:
    
    1. REPLACE the regular correlation computation in universal_neurons_pipeline.py:
       
       # Old way:
       correlator = NeuronCorrelationComputer(model_names, checkpoint_value=checkpoint_value)
       correlation_files = correlator.compute_all_correlations(dataset_path, output_dir)
       
       # New way:
       correlator = ExcessCorrelationComputer(model_names, checkpoint_value=checkpoint_value)
       correlation_files = correlator.compute_all_excess_correlations(dataset_path, output_dir)
    
    2. REPLACE the universal neuron analyzer:
       
       # Old way:
       analyzer = UniversalNeuronAnalyzer(correlation_files, neuron_stats)
       universal_df = analyzer.identify_universal_neurons(threshold=0.5, min_models=3)
       
       # New way:
       analyzer = ExcessCorrelationUniversalNeuronAnalyzer(correlation_files, neuron_stats)
       universal_df = analyzer.identify_universal_neurons_by_excess_correlation(excess_threshold=0.1)
    
    3. UPDATE your example_usage.py script:
       
       # Add this import at the top:
       from excess_correlation_implementation import run_excess_correlation_analysis
       
       # Replace the main analysis call:
       results = run_excess_correlation_analysis(
           model_names=MODELS,
           dataset_path=dataset_path,
           output_dir=CONFIG['output_dir'],
           excess_threshold=0.1,  # Instead of correlation_threshold
           checkpoint_value=checkpoint_value,
           top_k=None,  # Or specify a number like 100
           n_rotation_samples=5
       )
    
    4. PARAMETERS TO TUNE:
       
       - excess_threshold: Start with 0.1, adjust based on results
         Higher values = more stringent universal neuron criteria
         
       - n_rotation_samples: 5 is usually sufficient for stable baseline
         More samples = more accurate baseline but slower computation
         
       - top_k: Alternative to threshold - just take top K neurons
    
    5. KEY DIFFERENCES FROM ORIGINAL:
       
       - Files saved as "excess_correlation_*.pt" instead of "correlation_*.pt"
       - Additional files: excess_correlation_scores.csv with all neuron scores
       - Results files have "_excess" suffix to distinguish from regular correlation results
       - Baseline computation adds significant compute time but provides more principled identification
    
    6. INTERPRETING RESULTS:
       
       - Excess correlation > 0.1 typically indicates meaningful universality
       - Negative excess correlation means the neuron correlates LESS than random baseline
       - Look at the distribution of excess correlations to set appropriate thresholds
    """
    
    print(instructions)
    

if __name__ == "__main__":
    # Example usage
    print("Example usage of excess correlation analysis:")
    
    models = [
        "stanford-crfm/alias-gpt2-small-x21",
        "stanford-crfm/battlestar-gpt2-small-x49",
        "stanford-crfm/caprica-gpt2-small-x81"
    ]
    
    # Note: You need to provide a real dataset path
    dataset_path = "path/to/your/tokenized/dataset"
    
    # Run excess correlation analysis
    # results = run_excess_correlation_analysis(
    #     model_names=models,
    #     dataset_path=dataset_path,
    #     output_dir="excess_correlation_results",
    #     excess_threshold=0.1,
    #     checkpoint_value=1000,  # Example checkpoint
    #     n_rotation_samples=5
    # )
    
    # Show integration instructions
    integrate_excess_correlation_into_pipeline()