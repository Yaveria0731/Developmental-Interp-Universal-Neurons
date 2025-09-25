"""
Universal Neurons Analysis Pipeline - Modified for Checkpoint Support
Streamlined version for replicating the universal neurons experiment across training checkpoints
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

# ============================================================================
# 1. NEURON STATISTICS GENERATION (MODIFIED)
# ============================================================================

class NeuronStatsGenerator:
    """Generate comprehensive statistics for all neurons in a model"""
    
    def __init__(self, model_name: str, device: str = "cuda", checkpoint_value: Optional[Union[int, str]] = None):
        self.model_name = model_name
        self.device = device
        self.checkpoint_value = checkpoint_value
        
        # Load model with checkpoint if specified
        if checkpoint_value is not None:
            self.model = HookedTransformer.from_pretrained(
                model_name, 
                device=device, 
                checkpoint_value=checkpoint_value
            )
            self.model_identifier = f"{model_name}_checkpoint_{checkpoint_value}"
        else:
            self.model = HookedTransformer.from_pretrained(model_name, device=device)
            self.model_identifier = model_name
            
        self.model.eval()
        torch.set_grad_enabled(False)
    
    def get_model_device(self):
        """Get the actual device of the model parameters"""
        return next(self.model.parameters()).device
    
    def compute_weight_statistics(self) -> pd.DataFrame:
        """Compute weight-based statistics for all neurons"""
        # Use the actual device of model parameters
        device = self.get_model_device()
        
        # Get weight matrices
        W_in = einops.rearrange(self.model.W_in, 'l d n -> l n d').to(device)
        W_out = self.model.W_out.to(device)
        
        layers, d_mlp, _ = W_in.shape
        
        # Compute basic weight stats
        W_in_norms = torch.norm(W_in, dim=-1)
        W_out_norms = torch.norm(W_out, dim=-1)
        
        # Compute input-output similarity
        dot_product = (W_in * W_out).sum(dim=-1)
        cos_sim = dot_product / (W_in_norms * W_out_norms)
        
        # Weight norm penalty (L2)
        l2_penalty = W_in_norms**2 + W_out_norms**2
        
        # Create dataframe
        index = pd.MultiIndex.from_product(
            [range(layers), range(d_mlp)],
            names=["layer", "neuron"]
        )
        
        stats_df = pd.DataFrame({
            "w_in_norm": W_in_norms.flatten().cpu().numpy(),
            "w_out_norm": W_out_norms.flatten().cpu().numpy(),
            "in_out_sim": cos_sim.flatten().cpu().numpy(),
            "l2_penalty": l2_penalty.flatten().cpu().numpy(),
        }, index=index)
        
        return stats_df
    
    def compute_vocab_composition_stats(self) -> pd.DataFrame:
        """Compute vocabulary composition statistics"""
        device = self.get_model_device()
        
        # Normalize embeddings
        W_U = self.model.W_U / self.model.W_U.norm(dim=0, keepdim=True)
        
        vocab_stats = []
        for layer in range(self.model.cfg.n_layers):
            w_out = self.model.W_out[layer]
            w_out_norm = w_out / w_out.norm(dim=1)[:, None]
            
            # Compute cosine similarity with unembedding
            vocab_cosines = w_out_norm @ W_U
            
            # Compute moments
            mean = vocab_cosines.mean(dim=1)
            var = vocab_cosines.var(dim=1)
            
            # Compute skewness and kurtosis manually
            centered = vocab_cosines - mean[:, None]
            std = torch.sqrt(var)
            normalized = centered / std[:, None]
            skew = (normalized**3).mean(dim=1)
            kurt = (normalized**4).mean(dim=1)
            
            for neuron in range(self.model.cfg.d_mlp):
                vocab_stats.append({
                    'layer': layer,
                    'neuron': neuron,
                    'vocab_mean': mean[neuron].item(),
                    'vocab_var': var[neuron].item(),
                    'vocab_skew': skew[neuron].item(),
                    'vocab_kurt': kurt[neuron].item()
                })
        
        vocab_df = pd.DataFrame(vocab_stats)
        vocab_df.set_index(['layer', 'neuron'], inplace=True)
        return vocab_df
    
    def compute_activation_statistics(self, dataset_path: str, 
                                    batch_size: int = 32) -> pd.DataFrame:
        """Compute activation-based statistics on a dataset"""
        
        device = self.get_model_device()
        
        # Load tokenized dataset
        tokenized_dataset = datasets.load_from_disk(dataset_path)
        
        # Check dataset structure and handle accordingly
        if hasattr(tokenized_dataset, 'column_names') and 'tokens' in tokenized_dataset.column_names:
            # Dataset has 'tokens' column
            dataset_to_use = tokenized_dataset['tokens']
        else:
            # Dataset is directly the tokens
            dataset_to_use = tokenized_dataset
        
        # Convert tokens to tensor format for DataLoader
        def collate_fn(batch):
            # Handle different batch formats
            if isinstance(batch[0], list):
                sequences = batch
            elif isinstance(batch[0], dict) and 'tokens' in batch[0]:
                sequences = [item['tokens'] for item in batch]
            else:
                sequences = batch
            
            # Pad to max length in batch
            max_len = max(len(seq) for seq in sequences)
            padded = [seq + [0] * (max_len - len(seq)) for seq in sequences]
            return torch.tensor(padded, dtype=torch.long)
        
        dataloader = DataLoader(
            dataset_to_use, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
        
        # Initialize accumulators
        n_layers = self.model.cfg.n_layers
        d_mlp = self.model.cfg.d_mlp
        
        activation_sums = torch.zeros(n_layers, d_mlp, device=device)
        activation_sum_sq = torch.zeros(n_layers, d_mlp, device=device)
        sparsity_counts = torch.zeros(n_layers, d_mlp, device=device)
        total_tokens = 0
        
        def save_activations(tensor, hook):
            hook.ctx['activation'] = tensor.detach()
        
        # Set up hooks
        hooks = [(f'blocks.{layer}.mlp.hook_post', save_activations) 
                for layer in range(n_layers)]
        
        for batch in tqdm(dataloader, desc="Computing activation stats"):
            batch = batch.to(device)
            
            with torch.no_grad():
                self.model.run_with_hooks(batch, fwd_hooks=hooks)
            
            # Process activations
            for layer in range(n_layers):
                acts = self.model.hook_dict[f'blocks.{layer}.mlp.hook_post'].ctx['activation']
                acts_flat = acts.view(-1, d_mlp)  # [batch*seq, d_mlp]
                
                # Filter out padding tokens (assuming 0 is padding)
                valid_mask = (batch.view(-1) != 0)
                acts_valid = acts_flat[valid_mask]
                
                if len(acts_valid) > 0:
                    activation_sums[layer] += acts_valid.sum(dim=0)
                    activation_sum_sq[layer] += (acts_valid**2).sum(dim=0)
                    sparsity_counts[layer] += (acts_valid > 0).float().sum(dim=0)
                    if layer == 0:  # Count once
                        total_tokens += len(acts_valid)
            
            self.model.reset_hooks()
        
        # Compute final statistics
        mean_acts = activation_sums / total_tokens
        var_acts = (activation_sum_sq / total_tokens) - mean_acts**2
        sparsity = sparsity_counts / total_tokens
        
        # Create dataframe
        index = pd.MultiIndex.from_product(
            [range(n_layers), range(d_mlp)],
            names=["layer", "neuron"]
        )
        
        activation_df = pd.DataFrame({
            "mean": mean_acts.flatten().cpu().numpy(),
            "var": var_acts.flatten().cpu().numpy(), 
            "sparsity": sparsity.flatten().cpu().numpy(),
        }, index=index)
        
        return activation_df
    
    def generate_full_neuron_dataframe(self, dataset_path: Optional[str] = None) -> pd.DataFrame:
        """Generate complete neuron statistics dataframe"""
        print(f"Generating neuron statistics for {self.model_identifier}")
        
        # Compute weight statistics
        print("Computing weight statistics...")
        weight_stats = self.compute_weight_statistics()
        
        # Compute vocabulary composition statistics  
        print("Computing vocabulary composition statistics...")
        vocab_stats = self.compute_vocab_composition_stats()
        
        # Combine dataframes
        full_stats = pd.concat([weight_stats, vocab_stats], axis=1)
        
        # Add activation statistics if dataset provided
        if dataset_path and os.path.exists(dataset_path):
            print("Computing activation statistics...")
            activation_stats = self.compute_activation_statistics(dataset_path)
            full_stats = pd.concat([full_stats, activation_stats], axis=1)
        
        # Add model name and checkpoint for identification
        full_stats['model'] = self.model_identifier
        if self.checkpoint_value is not None:
            full_stats['checkpoint'] = self.checkpoint_value
        
        return full_stats

# ============================================================================
# 2. CORRELATION COMPUTATION (MODIFIED)
# ============================================================================

class NeuronCorrelationComputer:
    """Compute correlations between neurons across different models"""
    
    def __init__(self, model_names: List[str], device: str = "cuda", checkpoint_value: Optional[Union[int, str]] = None):
        self.model_names = model_names
        self.requested_device = device  # Store requested device
        self.checkpoint_value = checkpoint_value
        self.models = {}
        
        print("Loading models...")
        for name in model_names:
            print(f"Loading {name}...")
            if checkpoint_value is not None:
                model = HookedTransformer.from_pretrained(
                    name, 
                    device=device, 
                    checkpoint_value=checkpoint_value
                )
                model_id = f"{name}_checkpoint_{checkpoint_value}"
            else:
                model = HookedTransformer.from_pretrained(name, device=device)
                model_id = name
                
            model.eval()
            self.models[model_id] = model
        torch.set_grad_enabled(False)
    
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
    
    def compute_pairwise_correlation(self, model1_id: str, model2_id: str,
                               dataset_path: str, batch_size: int = 32) -> torch.Tensor:
        """Compute Pearson correlation between all neuron pairs - Full GPU version for high-end GPUs"""
        
        model1 = self.models[model1_id]
        model2 = self.models[model2_id]
        
        # Get actual devices - should be GPU for L40/A100
        device1 = self.get_model_device(model1)
        device2 = self.get_model_device(model2)
        
        # Use GPU for all computation
        compute_device = device1 if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {compute_device}")
        
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
        
        # Initialize correlation computation - all on GPU now
        m1_sum = torch.zeros(model1.cfg.n_layers, model1.cfg.d_mlp, 
                            dtype=torch.float64, device=compute_device)
        m1_sum_sq = torch.zeros(model1.cfg.n_layers, model1.cfg.d_mlp, 
                            dtype=torch.float64, device=compute_device)
        m2_sum = torch.zeros(model2.cfg.n_layers, model2.cfg.d_mlp, 
                            dtype=torch.float64, device=compute_device) 
        m2_sum_sq = torch.zeros(model2.cfg.n_layers, model2.cfg.d_mlp, 
                            dtype=torch.float64, device=compute_device)
        
        # Full cross-correlation tensor - this is what requires ~11GB
        cross_sum = torch.zeros(
            model1.cfg.n_layers, model1.cfg.d_mlp,
            model2.cfg.n_layers, model2.cfg.d_mlp,
            dtype=torch.float64, device=compute_device
        )
        n_samples = 0
        
        print(f"Allocated correlation tensors on {compute_device}")
        print(f"Cross-correlation tensor shape: {cross_sum.shape}")
        print(f"Estimated GPU memory usage: {cross_sum.numel() * 8 / (1024**3):.1f} GB")
        
        print(f"Computing correlation between {model1_id} and {model2_id}")
        for batch in tqdm(dataloader):
            # Send batch to model devices
            batch = batch.to(device1)
            
            # Get activations
            acts1 = self.get_activations(model1, batch)  # [l1, d1, t]
            
            if device1 != device2:
                batch = batch.to(device2)
            acts2 = self.get_activations(model2, batch)  # [l2, d2, t]
            
            # Filter padding tokens - ensure mask is on same device as activations
            valid_mask = (batch.flatten() != 0)
            
            # Move activations to compute device first, then apply mask
            acts1_compute = acts1.to(compute_device)
            acts2_compute = acts2.to(compute_device)
            valid_mask_compute = valid_mask.to(compute_device)
            
            acts1_filtered = acts1_compute[:, :, valid_mask_compute]
            acts2_filtered = acts2_compute[:, :, valid_mask_compute]
            
            n_tokens = acts1_filtered.shape[-1]
            if n_tokens == 0:  # Skip empty batches
                continue
                
            n_samples += n_tokens
            
            # Update first and second moment statistics
            m1_sum += acts1_filtered.sum(dim=-1)
            m1_sum_sq += (acts1_filtered**2).sum(dim=-1)
            m2_sum += acts2_filtered.sum(dim=-1)
            m2_sum_sq += (acts2_filtered**2).sum(dim=-1)
            
            # Compute cross products - this is where GPU really shines
            # Use einsum for efficient batch computation across all layer pairs
            cross_sum += torch.einsum('ijk,ljk->iljk', acts1_filtered, acts2_filtered)
            
            # Alternative: nested loops (slightly more memory efficient but slower)
            # for l1 in range(model1.cfg.n_layers):
            #     for l2 in range(model2.cfg.n_layers):
            #         cross_sum[l1, :, l2, :] += torch.mm(
            #             acts1_filtered[l1], acts2_filtered[l2].T
            #         )
        
        if n_samples == 0:
            raise ValueError("No valid samples found in dataset")
        
        # Compute Pearson correlation - all on GPU
        print("Computing final correlations on GPU...")
        correlations = torch.zeros_like(cross_sum, device=compute_device)
        
        # Vectorized computation across all layer pairs
        for l1 in range(model1.cfg.n_layers):
            for l2 in range(model2.cfg.n_layers):
                # Numerator: E[XY] - E[X]E[Y]
                numerator = cross_sum[l1, :, l2, :] / n_samples - torch.outer(
                    m1_sum[l1] / n_samples, m2_sum[l2] / n_samples
                )
                
                # Denominator: sqrt(Var[X] * Var[Y])
                var1 = m1_sum_sq[l1] / n_samples - (m1_sum[l1] / n_samples)**2
                var2 = m2_sum_sq[l2] / n_samples - (m2_sum[l2] / n_samples)**2
                denominator = torch.outer(torch.sqrt(var1 + 1e-8), torch.sqrt(var2 + 1e-8))
                
                correlations[l1, :, l2, :] = numerator / denominator
        
        print(f"Correlation computation complete. Returning tensor to CPU...")
        return correlations.cpu()  # Move back to CPU for saving/further processing
    
    def compute_all_correlations(self, dataset_path: str) -> Dict[Tuple[str, str], torch.Tensor]:
        """Compute correlations between all model pairs"""
        correlations = {}
        model_ids = list(self.models.keys())
        
        for i, model1 in enumerate(model_ids):
            for j, model2 in enumerate(model_ids[i:], i):
                if i == j:
                    continue  # Skip self-correlation
                
                pair = (model1, model2)
                print(f"Computing correlation for pair: {pair}")
                
                corr_matrix = self.compute_pairwise_correlation(
                    model1, model2, dataset_path
                )
                correlations[pair] = corr_matrix
        
        return correlations

# ============================================================================
# 3. UNIVERSAL NEURON IDENTIFICATION (UNCHANGED)
# ============================================================================

class UniversalNeuronAnalyzer:
    """Identify and analyze universal neurons"""
    
    def __init__(self, correlation_results: Dict[Tuple[str, str], torch.Tensor],
                 neuron_stats: Dict[str, pd.DataFrame]):
        self.correlation_results = correlation_results
        self.neuron_stats = neuron_stats
    
    def identify_universal_neurons(self, threshold: float = 0.5,
                                 min_models: int = 3) -> pd.DataFrame:
        """Identify neurons that are highly correlated across multiple models"""
        
        universal_neurons = []
        
        # Get model names
        all_models = set()
        for (m1, m2) in self.correlation_results.keys():
            all_models.update([m1, m2])
        all_models = list(all_models)
        
        # For each neuron in first model, find its best matches in other models
        first_model = all_models[0]
        first_stats = self.neuron_stats[first_model]
        
        for (layer, neuron), row in first_stats.iterrows():
            correlations_found = []
            matching_neurons = [(first_model, layer, neuron)]
            
            for other_model in all_models[1:]:
                pair_key = (first_model, other_model)
                if pair_key not in self.correlation_results:
                    pair_key = (other_model, first_model)
                
                if pair_key in self.correlation_results:
                    corr_matrix = self.correlation_results[pair_key]
                    
                    # Find best correlation for this neuron
                    if pair_key[0] == first_model:
                        neuron_corrs = corr_matrix[layer, neuron, :, :]
                    else:
                        neuron_corrs = corr_matrix[:, :, layer, neuron]
                    
                    max_corr = neuron_corrs.max()
                    if max_corr > threshold:
                        max_idx = neuron_corrs.argmax()
                        if pair_key[0] == first_model:
                            best_layer, best_neuron = np.unravel_index(
                                max_idx, neuron_corrs.shape
                            )
                        else:
                            best_layer, best_neuron = np.unravel_index(
                                max_idx, neuron_corrs.shape
                            )
                        
                        correlations_found.append(max_corr.item())
                        matching_neurons.append((other_model, best_layer, best_neuron))
            
            # If neuron is universal across enough models
            if len(correlations_found) >= min_models - 1:
                universal_neurons.append({
                    'reference_model': first_model,
                    'reference_layer': layer,
                    'reference_neuron': neuron,
                    'matching_neurons': matching_neurons,
                    'correlations': correlations_found,
                    'mean_correlation': np.mean(correlations_found),
                    'min_correlation': np.min(correlations_found),
                    'n_models': len(matching_neurons)
                })
        
        return pd.DataFrame(universal_neurons)
    
    def analyze_universal_properties(self, universal_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze properties of universal neurons vs regular neurons"""
        
        analysis_results = []
        
        for model_name, stats_df in self.neuron_stats.items():
            # Mark universal neurons
            is_universal = np.zeros(len(stats_df), dtype=bool)
            
            for _, row in universal_df.iterrows():
                for model, layer, neuron in row['matching_neurons']:
                    if model == model_name:
                        try:
                            idx = stats_df.index.get_loc((layer, neuron))
                            is_universal[idx] = True
                        except KeyError:
                            continue
            
            stats_df['is_universal'] = is_universal
            
            # Compute statistics
            universal_stats = stats_df[is_universal].describe()
            regular_stats = stats_df[~is_universal].describe()
            
            # Compare distributions
            for stat_col in ['w_in_norm', 'w_out_norm', 'l2_penalty', 'vocab_var', 'vocab_kurt']:
                if stat_col in stats_df.columns:
                    universal_mean = stats_df[is_universal][stat_col].mean()
                    regular_mean = stats_df[~is_universal][stat_col].mean()
                    
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

# ============================================================================
# 4. MAIN PIPELINE (MODIFIED)
# ============================================================================

def run_universal_neurons_analysis(
    model_names: List[str],
    dataset_path: str,
    output_dir: str = "universal_neurons_results",
    correlation_threshold: float = 0.5,
    min_models: int = 3,
    checkpoint_value: Optional[Union[int, str]] = None
):
    """Run complete universal neurons analysis pipeline"""
    
    # Modify output directory to include checkpoint info
    if checkpoint_value is not None:
        output_dir = f"{output_dir}_checkpoint_{checkpoint_value}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Generate neuron statistics for each model
    print("=" * 50)
    print("STEP 1: GENERATING NEURON STATISTICS")
    if checkpoint_value is not None:
        print(f"CHECKPOINT: {checkpoint_value}")
    print("=" * 50)
    
    neuron_stats = {}
    for model_name in model_names:
        print(f"\nProcessing {model_name}...")
        generator = NeuronStatsGenerator(model_name, checkpoint_value=checkpoint_value)
        stats_df = generator.generate_full_neuron_dataframe(dataset_path)
        
        # Use the model identifier that includes checkpoint info
        model_id = generator.model_identifier
        neuron_stats[model_id] = stats_df
        
        # Save individual model stats with checkpoint info in filename
        safe_model_name = model_name.replace('/', '_')
        if checkpoint_value is not None:
            filename = f"{safe_model_name}_checkpoint_{checkpoint_value}_neuron_stats.csv"
        else:
            filename = f"{safe_model_name}_neuron_stats.csv"
        stats_df.to_csv(f"{output_dir}/{filename}")
        print(f"Saved stats for {model_id}: {len(stats_df)} neurons")
    
    # Step 2: Compute correlations between models
    print("\n" + "=" * 50)
    print("STEP 2: COMPUTING INTER-MODEL CORRELATIONS")  
    print("=" * 50)
    
    correlator = NeuronCorrelationComputer(model_names, checkpoint_value=checkpoint_value)
    correlations = correlator.compute_all_correlations(dataset_path)
    
    # Save correlations
    correlation_filename = "correlations.pt"
    if checkpoint_value is not None:
        correlation_filename = f"correlations_checkpoint_{checkpoint_value}.pt"
    correlation_file = f"{output_dir}/{correlation_filename}"
    torch.save(correlations, correlation_file)
    print(f"Saved correlations to {correlation_file}")
    
    # Step 3: Identify universal neurons
    print("\n" + "=" * 50)
    print("STEP 3: IDENTIFYING UNIVERSAL NEURONS")
    print("=" * 50)
    
    analyzer = UniversalNeuronAnalyzer(correlations, neuron_stats)
    universal_df = analyzer.identify_universal_neurons(
        threshold=correlation_threshold,
        min_models=min_models
    )
    
    universal_filename = "universal_neurons.csv"
    if checkpoint_value is not None:
        universal_filename = f"universal_neurons_checkpoint_{checkpoint_value}.csv"
    universal_file = f"{output_dir}/{universal_filename}"
    universal_df.to_csv(universal_file, index=False)
    print(f"Found {len(universal_df)} universal neurons")
    print(f"Saved to {universal_file}")
    
    # Step 4: Analyze universal neuron properties
    print("\n" + "=" * 50)
    print("STEP 4: ANALYZING UNIVERSAL NEURON PROPERTIES")
    print("=" * 50)
    
    analysis_df = analyzer.analyze_universal_properties(universal_df)
    analysis_filename = "universal_analysis.csv"
    if checkpoint_value is not None:
        analysis_filename = f"universal_analysis_checkpoint_{checkpoint_value}.csv"
    analysis_file = f"{output_dir}/{analysis_filename}"
    analysis_df.to_csv(analysis_file, index=False)
    print(f"Saved analysis to {analysis_file}")
    
    # Summary
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 50)
    print(f"Models analyzed: {len(model_names)}")
    if checkpoint_value is not None:
        print(f"Checkpoint: {checkpoint_value}")
    print(f"Universal neurons found: {len(universal_df)}")
    print(f"Mean correlation threshold: {correlation_threshold}")
    print(f"Results saved to: {output_dir}")
    
    return {
        'neuron_stats': neuron_stats,
        'correlations': correlations,
        'universal_neurons': universal_df,
        'analysis': analysis_df,
        'checkpoint': checkpoint_value
    }

# ============================================================================
# 5. USAGE EXAMPLE (UNCHANGED FROM ORIGINAL)
# ============================================================================

if __name__ == "__main__":
    # Stanford CRFM GPT2-Small models
    models = [
        "stanford-crfm/alias-gpt2-small-x21",
        "stanford-crfm/battlestar-gpt2-small-x49",
        "stanford-crfm/caprica-gpt2-small-x81",
        "stanford-crfm/darkmatter-gpt2-small-x343",
        "stanford-crfm/expanse-gpt2-small-x777"
    ]
    
    # You'll need to create/download a tokenized dataset
    dataset_path = "path/to/tokenized/dataset"
    
    # Run analysis
    results = run_universal_neurons_analysis(
        model_names=models,
        dataset_path=dataset_path,
        output_dir="universal_neurons_results",
        correlation_threshold=0.5,
        min_models=3,
        checkpoint_value=1000  # Example: analyze at step 1000
    )