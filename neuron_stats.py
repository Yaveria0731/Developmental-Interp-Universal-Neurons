import os
import torch
import numpy as np
import pandas as pd
import einops
from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader


def save_activation_hook(tensor, hook, device='cpu', dtype=torch.float16):
    """Hook function to save activations."""
    hook.ctx['activation'] = tensor.detach().to(dtype).to(device)


def extract_mlp_activations(model, dataloader, device='cpu', 
                           activation_location='mlp.hook_post',
                           filter_tokens=True):
    """Extract MLP activations from all layers."""
    
    hooks = [
        (f'blocks.{layer_ix}.{activation_location}',
         partial(save_activation_hook, device=device))
        for layer_ix in range(model.cfg.n_layers)
    ]
    
    all_activations = []
    
    for step, batch in enumerate(tqdm(dataloader, desc="Extracting activations")):
        batch = batch.to(device)
        
        with torch.no_grad():
            model.run_with_hooks(
                batch,
                fwd_hooks=hooks,
                stop_at_layer=model.cfg.n_layers + 1  # Don't compute logits
            )
        
        # Stack activations from all layers
        batch_activations = torch.stack([
            model.hook_dict[hook_pt[0]].ctx['activation'] 
            for hook_pt in hooks
        ], dim=2)  # Shape: [batch, context, layers, neurons]
        
        model.reset_hooks()
        
        # Reshape to [layers, neurons, (batch * context)]
        batch_activations = einops.rearrange(
            batch_activations, 'batch context layers neurons -> layers neurons (batch context)'
        )
        
        if filter_tokens:
            # Filter out padding and special tokens
            from dataset_utilities import filter_valid_tokens
            valid_mask = filter_valid_tokens(batch.flatten(), model)
            batch_activations = batch_activations[:, :, valid_mask]
        
        all_activations.append(batch_activations.cpu())
        
        # Clear GPU memory
        del batch_activations
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Concatenate all activations
    activations = torch.cat(all_activations, dim=2)
    return activations


def compute_activation_statistics(activations):
    """Compute comprehensive activation statistics."""
    # activations shape: [layers, neurons, tokens]
    
    stats = {}
    
    # Basic statistics
    stats['mean'] = activations.mean(dim=2)
    stats['var'] = activations.var(dim=2)
    stats['std'] = activations.std(dim=2)
    
    # Higher order moments
    centered = activations - stats['mean'].unsqueeze(2)
    stats['skew'] = (centered ** 3).mean(dim=2) / (stats['std'] ** 3)
    stats['kurtosis'] = (centered ** 4).mean(dim=2) / (stats['var'] ** 2)
    
    # Sparsity (fraction of activations > 0)
    stats['sparsity'] = (activations > 0).float().mean(dim=2)
    
    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        stats[f'p{p}'] = torch.quantile(activations, p/100, dim=2)
    
    return stats


def compute_weight_statistics(model):
    """Compute weight-based statistics for each neuron."""
    
    layers, d_mlp = model.cfg.n_layers, model.cfg.d_mlp
    
    # Get weight matrices
    W_in = model.W_in  # [layers, d_model, d_mlp]
    W_out = model.W_out  # [layers, d_mlp, d_model]
    b_in = model.b_in   # [layers, d_mlp]
    
    # Reshape for easier computation
    W_in = einops.rearrange(W_in, 'l d_model d_mlp -> l d_mlp d_model')
    
    stats = {}
    
    # Weight norms
    stats['input_weight_norm'] = torch.norm(W_in, dim=2)
    stats['output_weight_norm'] = torch.norm(W_out, dim=2)
    stats['input_bias'] = b_in
    
    # Weight decay penalty (L2 regularization term)
    stats['l2_penalty'] = stats['input_weight_norm']**2 + stats['output_weight_norm']**2
    
    # Cosine similarity between input and output weights
    W_in_normalized = W_in / (torch.norm(W_in, dim=2, keepdim=True) + 1e-8)
    W_out_normalized = W_out / (torch.norm(W_out, dim=2, keepdim=True) + 1e-8)
    
    # Dot product between normalized input and output weights
    stats['cos_input_output'] = (W_in_normalized * W_out_normalized).sum(dim=2)
    
    return stats


def compute_vocab_composition_statistics(model):
    """Compute neuron-vocabulary composition statistics."""
    
    W_out = model.W_out  # [layers, d_mlp, d_model]
    W_U = model.W_U      # [d_model, d_vocab]
    
    # Normalize weights
    W_out_norm = W_out / (torch.norm(W_out, dim=2, keepdim=True) + 1e-8)
    W_U_norm = W_U / (torch.norm(W_U, dim=0, keepdim=True) + 1e-8)
    
    # Compute composition: how much each neuron contributes to each vocabulary item
    vocab_composition = torch.einsum('l n d, d v -> l n v', W_out_norm, W_U_norm)
    
    stats = {}
    
    # Statistics over vocabulary dimension
    stats['vocab_mean'] = vocab_composition.mean(dim=2)
    stats['vocab_var'] = vocab_composition.var(dim=2)
    stats['vocab_std'] = vocab_composition.std(dim=2)
    
    # Higher order moments
    centered = vocab_composition - stats['vocab_mean'].unsqueeze(2)
    stats['vocab_skew'] = (centered ** 3).mean(dim=2) / (stats['vocab_std'] ** 3 + 1e-8)
    stats['vocab_kurt'] = (centered ** 4).mean(dim=2) / (stats['vocab_var'] ** 2 + 1e-8)
    
    return stats


def create_neuron_dataframe(activation_stats, weight_stats, vocab_stats):
    """Create a comprehensive neuron statistics dataframe."""
    
    layers, d_mlp = activation_stats['mean'].shape
    
    # Create MultiIndex for (layer, neuron)
    index = pd.MultiIndex.from_product(
        [range(layers), range(d_mlp)],
        names=['layer', 'neuron']
    )
    
    # Flatten all statistics
    data = {}
    
    # Activation statistics
    for key, values in activation_stats.items():
        data[f'act_{key}'] = values.flatten().numpy()
    
    # Weight statistics  
    for key, values in weight_stats.items():
        data[f'weight_{key}'] = values.flatten().numpy()
    
    # Vocabulary composition statistics
    for key, values in vocab_stats.items():
        data[key] = values.flatten().numpy()
    
    df = pd.DataFrame(data, index=index)
    return df


def compute_comprehensive_neuron_stats(model, dataset, batch_size=32, device='auto'):
    """Compute comprehensive neuron statistics including activations and weights."""
    
    if device == 'auto':
        device = next(model.parameters()).device
    
    # Create dataloader
    dataloader = DataLoader(dataset['tokens'], batch_size=batch_size, shuffle=False)
    
    print("Computing activation statistics...")
    # Extract activations
    activations = extract_mlp_activations(model, dataloader, device)
    
    # Compute activation statistics
    activation_stats = compute_activation_statistics(activations)
    
    print("Computing weight statistics...")
    # Compute weight statistics
    weight_stats = compute_weight_statistics(model)
    
    print("Computing vocabulary composition statistics...")
    # Compute vocabulary composition statistics
    vocab_stats = compute_vocab_composition_statistics(model)
    
    print("Creating neuron dataframe...")
    # Create comprehensive dataframe
    neuron_df = create_neuron_dataframe(activation_stats, weight_stats, vocab_stats)
    
    return neuron_df


def save_neuron_stats(neuron_df, save_path, model_name, checkpoint=None):
    """Save neuron statistics to file."""
    
    os.makedirs(save_path, exist_ok=True)
    
    if checkpoint is not None:
        filename = f"neuron_stats_{model_name.replace('/', '_')}_checkpoint_{checkpoint}.csv"
    else:
        filename = f"neuron_stats_{model_name.replace('/', '_')}.csv"
    
    filepath = os.path.join(save_path, filename)
    neuron_df.to_csv(filepath)
    
    print(f"Neuron statistics saved to: {filepath}")
    return filepath


def load_neuron_stats(filepath):
    """Load neuron statistics from file."""
    df = pd.read_csv(filepath, index_col=[0, 1])
    return df


def identify_high_activation_neurons(neuron_df, threshold_percentile=95):
    """Identify neurons with high activation statistics."""
    
    # Neurons with high mean activation
    high_mean = neuron_df['act_mean'] > neuron_df['act_mean'].quantile(threshold_percentile/100)
    
    # Neurons with high variance (more dynamic)
    high_var = neuron_df['act_var'] > neuron_df['act_var'].quantile(threshold_percentile/100)
    
    # Neurons with high sparsity (selective activation)
    low_sparsity = neuron_df['act_sparsity'] < neuron_df['act_sparsity'].quantile((100-threshold_percentile)/100)
    
    # Neurons with extreme vocabulary composition
    high_vocab_kurt = neuron_df['vocab_kurt'] > neuron_df['vocab_kurt'].quantile(threshold_percentile/100)
    
    interesting_neurons = {
        'high_mean': neuron_df[high_mean],
        'high_variance': neuron_df[high_var], 
        'selective': neuron_df[low_sparsity],
        'vocab_specialist': neuron_df[high_vocab_kurt]
    }
    
    return interesting_neurons


def analyze_neuron_properties(neuron_df):
    """Analyze overall properties of the neuron population."""
    
    analysis = {}
    
    # Distribution of activation statistics
    for col in neuron_df.columns:
        if col.startswith('act_'):
            analysis[f"{col}_distribution"] = {
                'mean': neuron_df[col].mean(),
                'std': neuron_df[col].std(),
                'min': neuron_df[col].min(),
                'max': neuron_df[col].max(),
                'q25': neuron_df[col].quantile(0.25),
                'q50': neuron_df[col].quantile(0.50),
                'q75': neuron_df[col].quantile(0.75)
            }
    
    # Layer-wise analysis
    layer_stats = neuron_df.groupby('layer').agg({
        'act_mean': ['mean', 'std'],
        'act_sparsity': ['mean', 'std'], 
        'vocab_kurt': ['mean', 'std'],
        'weight_l2_penalty': ['mean', 'std']
    })
    
    analysis['layer_statistics'] = layer_stats
    
    return analysis