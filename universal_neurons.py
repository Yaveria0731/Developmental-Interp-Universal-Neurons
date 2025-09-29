import os
import time
import torch
import numpy as np
import pandas as pd
import einops
from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader
from scipy.stats import special_ortho_group


def save_activation_hook(tensor, hook, device='cpu', dtype=torch.float16):
    """Hook function to save activations."""
    hook.ctx['activation'] = tensor.detach().to(dtype).to(device)


def extract_activations_for_correlation(model, dataloader, device='cpu',
                                      activation_location='mlp.hook_post',
                                      filter_tokens=True):
    """Extract MLP activations optimized for correlation computation."""
    
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
                stop_at_layer=model.cfg.n_layers + 1
            )
        
        # Stack activations: [layers, neurons, (batch * context)]
        batch_activations = torch.stack([
            model.hook_dict[hook_pt[0]].ctx['activation'] 
            for hook_pt in hooks
        ], dim=0)
        
        model.reset_hooks()
        
        # Reshape to [layers, neurons, (batch * context)]
        batch_activations = einops.rearrange(
            batch_activations, 'layers batch context neurons -> layers neurons (batch context)'
        )
        
        if filter_tokens:
            from dataset_utilities import filter_valid_tokens
            valid_mask = filter_valid_tokens(batch.flatten(), model)
            batch_activations = batch_activations[:, :, valid_mask]
        
        all_activations.append(batch_activations.cpu())
        
        del batch_activations
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Concatenate all activations
    activations = torch.cat(all_activations, dim=2)
    return activations


class StreamingPearsonComputer:
    """Streaming computation of Pearson correlations between two models."""
    
    def __init__(self, model_1, model_2, device='cpu'):
        self.device = device
        
        m1_layers = model_1.cfg.n_layers
        m2_layers = model_2.cfg.n_layers  
        m1_dmlp = model_1.cfg.d_mlp
        m2_dmlp = model_2.cfg.d_mlp
        
        # Initialize accumulators
        self.m1_sum = torch.zeros((m1_layers, m1_dmlp), dtype=torch.float64, device=device)
        self.m1_sum_sq = torch.zeros((m1_layers, m1_dmlp), dtype=torch.float64, device=device)
        
        self.m2_sum = torch.zeros((m2_layers, m2_dmlp), dtype=torch.float64, device=device)
        self.m2_sum_sq = torch.zeros((m2_layers, m2_dmlp), dtype=torch.float64, device=device)
        
        self.m1_m2_sum = torch.zeros(
            (m1_layers, m1_dmlp, m2_layers, m2_dmlp),
            dtype=torch.float64, device=device
        )
        self.n = 0
        
    def update_correlation_data(self, batch_1_acts, batch_2_acts):
        """Update correlation statistics with a batch of activations."""
        
        # Move to computation device and convert to float64 for precision
        batch_1_acts = batch_1_acts.to(self.device).to(torch.float64)
        batch_2_acts = batch_2_acts.to(self.device).to(torch.float64)
        
        # Update sums
        self.m1_sum += batch_1_acts.sum(dim=-1)
        self.m1_sum_sq += (batch_1_acts ** 2).sum(dim=-1)
        
        self.m2_sum += batch_2_acts.sum(dim=-1)
        self.m2_sum_sq += (batch_2_acts ** 2).sum(dim=-1)
        
        # Update cross products (memory intensive)
        self.m1_m2_sum += einops.einsum(
            batch_1_acts, batch_2_acts, 'l1 n1 t, l2 n2 t -> l1 n1 l2 n2'
        )
        
        self.n += batch_1_acts.shape[-1]
        
    def compute_correlation(self):
        """Compute final Pearson correlation matrix."""
        layer_correlations = []
        
        # Compute layer-wise for memory efficiency
        for l1 in range(self.m1_sum.shape[0]):
            numerator = self.m1_m2_sum[l1, :, :, :] - (1 / self.n) * einops.einsum(
                self.m1_sum[l1, :], self.m2_sum, 'n1, l2 n2 -> n1 l2 n2'
            )
            
            m1_norm = (self.m1_sum_sq[l1, :] - (1 / self.n) * self.m1_sum[l1, :] ** 2) ** 0.5
            m2_norm = (self.m2_sum_sq - (1 / self.n) * self.m2_sum ** 2) ** 0.5
            
            l_correlation = numerator / einops.einsum(
                m1_norm, m2_norm, 'n1, l2 n2 -> n1 l2 n2'
            )
            
            layer_correlations.append(l_correlation.to(torch.float16))
        
        correlation = torch.stack(layer_correlations, dim=0)
        return correlation


def generate_rotation_matrix(n_layers, d_mlp, device='cpu'):
    """Generate proper orthogonal rotation matrices using QR factorization."""
    rotation_matrices = []
    
    for layer in range(n_layers):
        # Generate random matrix
        random_matrix = torch.randn(d_mlp, d_mlp, device=device, dtype=torch.float64)
        
        # QR decomposition to get orthogonal matrix
        Q, R = torch.linalg.qr(random_matrix)
        
        # Ensure proper rotation (det = +1, not reflection)
        if torch.det(Q) < 0:
            Q[:, 0] *= -1
            
        rotation_matrices.append(Q.to(torch.float32))
    
    return torch.stack(rotation_matrices, dim=0)


def apply_rotation_baseline(activations, rotation_matrices):
    """Apply rotation matrices to activations for baseline."""
    rotated_activations = []
    
    for l in range(activations.shape[0]):
        rotated = torch.einsum(
            'mn,nt->mt',
            rotation_matrices[l],
            activations[l]
        )
        rotated_activations.append(rotated)
    
    return torch.stack(rotated_activations, dim=0)


def compute_pairwise_correlations(model_1, model_2, dataset, batch_size=32,
                                 baseline='rotation', device='auto'):
    """Compute pairwise neuron correlations between two models."""
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Computing correlations on {device}")
    
    # Initialize correlation computer
    corr_computer = StreamingPearsonComputer(model_1, model_2, device=device)
    
    # Prepare rotation matrices for baseline if needed
    if baseline == 'rotation':
        rotation_matrices = generate_rotation_matrix(
            model_2.cfg.n_layers, 
            model_2.cfg.d_mlp, 
            device=device
        )
        print("Generated orthogonal rotation matrices for baseline")
    
    # Create dataloader
    dataloader = DataLoader(dataset['tokens'], batch_size=batch_size, shuffle=False)
    
    start_time = time.time()
    
    for step, batch in enumerate(tqdm(dataloader, desc="Computing correlations")):
        # Extract activations from both models
        batch_dict = {'tokens': batch}
        
        # Model 1 activations
        model_1.to(device)
        dataloader_1 = DataLoader([batch], batch_size=1, shuffle=False)
        m1_activations = extract_activations_for_correlation(
            model_1, dataloader_1, device=device
        )
        
        # Model 2 activations  
        model_2.to(device)
        dataloader_2 = DataLoader([batch], batch_size=1, shuffle=False)
        m2_activations = extract_activations_for_correlation(
            model_2, dataloader_2, device=device
        )
        
        # Apply baseline transformation if needed
        if baseline == 'rotation':
            m2_activations = apply_rotation_baseline(m2_activations, rotation_matrices)
        elif baseline == 'permutation':
            # Shuffle along token dimension
            perm = torch.randperm(m2_activations.shape[-1])
            m2_activations = m2_activations[:, :, perm]
        elif baseline == 'gaussian':
            # Replace with Gaussian noise
            m2_activations = torch.randn_like(m2_activations)
            m2_activations = torch.nn.functional.gelu(m2_activations)
        
        # Update correlation statistics
        corr_computer.update_correlation_data(m1_activations, m2_activations)
        
        # Cleanup
        del m1_activations, m2_activations
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Compute final correlations
    print("Computing final correlation matrix...")
    correlation = corr_computer.compute_correlation()
    
    end_time = time.time()
    print(f"Correlation computation took {end_time - start_time:.2f} seconds")
    
    return correlation


def flatten_correlation_matrix(correlation_data):
    """Flatten correlation matrix from [l1, n1, l2, n2] to [(l1*n1), (l2*n2)]."""
    return einops.rearrange(correlation_data, 'l1 n1 l2 n2 -> (l1 n1) (l2 n2)')


def unflatten_correlation_matrix(correlation_data, m1_layers, m1_neurons, m2_layers, m2_neurons):
    """Unflatten correlation matrix back to [l1, n1, l2, n2] shape."""
    return einops.rearrange(
        correlation_data, '(l1 n1) (l2 n2) -> l1 n1 l2 n2',
        l1=m1_layers, n1=m1_neurons, l2=m2_layers, n2=m2_neurons
    )


def compute_excess_correlation(max_correlation, baseline_correlation):
    """Compute excess correlation (main - baseline)."""
    return max_correlation - baseline_correlation


def identify_universal_neurons(correlation_matrix, baseline_matrix, threshold=0.5):
    """Identify universal neurons based on excess correlation threshold."""
    
    # Flatten correlation matrices
    corr_flat = flatten_correlation_matrix(correlation_matrix)
    baseline_flat = flatten_correlation_matrix(baseline_matrix)
    
    # Compute maximum correlations for each neuron
    max_corr = corr_flat.max(dim=1)[0]
    max_baseline = baseline_flat.max(dim=1)[0]
    
    # Compute excess correlation
    excess_corr = compute_excess_correlation(max_corr, max_baseline)
    
    # Identify universal neurons
    universal_mask = excess_corr > threshold
    universal_indices = torch.where(universal_mask)[0]
    
    # Convert flat indices back to (layer, neuron) pairs
    n_layers = correlation_matrix.shape[0]
    n_neurons = correlation_matrix.shape[1]
    
    universal_neurons = []
    for idx in universal_indices:
        layer = idx // n_neurons
        neuron = idx % n_neurons
        excess_val = excess_corr[idx].item()
        max_corr_val = max_corr[idx].item()
        universal_neurons.append((layer.item(), neuron.item(), excess_val, max_corr_val))
    
    return universal_neurons, excess_corr


def analyze_correlation_patterns(correlation_matrix, baseline_matrix):
    """Analyze patterns in correlation matrices."""
    
    corr_flat = flatten_correlation_matrix(correlation_matrix)
    baseline_flat = flatten_correlation_matrix(baseline_matrix)
    
    analysis = {}
    
    # Distribution statistics
    analysis['correlation_stats'] = {
        'mean': corr_flat.mean().item(),
        'std': corr_flat.std().item(),
        'max': corr_flat.max().item(),
        'min': corr_flat.min().item()
    }
    
    analysis['baseline_stats'] = {
        'mean': baseline_flat.mean().item(),
        'std': baseline_flat.std().item(), 
        'max': baseline_flat.max().item(),
        'min': baseline_flat.min().item()
    }
    
    # Maximum correlations per neuron
    max_corr = corr_flat.max(dim=1)[0]
    max_baseline = baseline_flat.max(dim=1)[0]
    excess_corr = compute_excess_correlation(max_corr, max_baseline)
    
    analysis['excess_correlation_stats'] = {
        'mean': excess_corr.mean().item(),
        'std': excess_corr.std().item(),
        'max': excess_corr.max().item(),
        'min': excess_corr.min().item()
    }
    
    # Count universal neurons at different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    analysis['universal_counts'] = {}
    
    for thresh in thresholds:
        count = (excess_corr > thresh).sum().item()
        percentage = (count / len(excess_corr)) * 100
        analysis['universal_counts'][thresh] = {
            'count': count,
            'percentage': percentage
        }
    
    return analysis


def save_correlation_results(correlation_matrix, baseline_matrix, save_path, 
                           model_1_name, model_2_name, dataset_name, checkpoint_1=None, checkpoint_2=None):
    """Save correlation results to disk."""
    
    os.makedirs(save_path, exist_ok=True)
    
    # Create filename
    ckpt_suffix = ""
    if checkpoint_1 is not None:
        ckpt_suffix += f"_ckpt1_{checkpoint_1}"
    if checkpoint_2 is not None:
        ckpt_suffix += f"_ckpt2_{checkpoint_2}"
    
    base_name = f"correlations_{model_1_name.replace('/', '_')}+{model_2_name.replace('/', '_')}_{dataset_name}{ckpt_suffix}"
    
    # Save matrices
    torch.save(correlation_matrix, os.path.join(save_path, f"{base_name}_main.pt"))
    torch.save(baseline_matrix, os.path.join(save_path, f"{base_name}_baseline.pt"))
    
    # Identify and save universal neurons
    universal_neurons, excess_corr = identify_universal_neurons(correlation_matrix, baseline_matrix)
    
    # Create DataFrame of universal neurons
    if universal_neurons:
        universal_df = pd.DataFrame(universal_neurons, 
                                  columns=['layer', 'neuron', 'excess_correlation', 'max_correlation'])
        universal_df.to_csv(os.path.join(save_path, f"{base_name}_universal_neurons.csv"), index=False)
    
    # Save analysis
    analysis = analyze_correlation_patterns(correlation_matrix, baseline_matrix)
    
    import json
    with open(os.path.join(save_path, f"{base_name}_analysis.json"), 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Results saved to {save_path}")
    print(f"Found {len(universal_neurons)} universal neurons with excess correlation > 0.5")
    
    return universal_df, analysis


def run_universal_neuron_analysis(model_1, model_2, dataset, batch_size=32, 
                                threshold=0.5, baseline='rotation', device='auto',
                                save_path='results/universal_neurons'):
    """Run complete universal neuron analysis."""
    
    print("Starting universal neuron analysis...")
    print(f"Model 1: {model_1.cfg.model_name if hasattr(model_1.cfg, 'model_name') else 'Unknown'}")
    print(f"Model 2: {model_2.cfg.model_name if hasattr(model_2.cfg, 'model_name') else 'Unknown'}")
    print(f"Baseline: {baseline}")
    print(f"Threshold: {threshold}")
    
    # Compute main correlations (no baseline)
    print("\nComputing main correlations...")
    main_correlation = compute_pairwise_correlations(
        model_1, model_2, dataset, batch_size, baseline='none', device=device
    )
    
    # Compute baseline correlations
    print(f"\nComputing baseline correlations ({baseline})...")
    baseline_correlation = compute_pairwise_correlations(
        model_1, model_2, dataset, batch_size, baseline=baseline, device=device
    )
    
    # Identify universal neurons
    print("\nIdentifying universal neurons...")
    universal_neurons, excess_corr = identify_universal_neurons(
        main_correlation, baseline_correlation, threshold
    )
    
    # Analyze results
    analysis = analyze_correlation_patterns(main_correlation, baseline_correlation)
    
    # Print summary
    print(f"\nUniversal Neuron Analysis Summary:")
    print(f"Total neurons analyzed: {main_correlation.shape[0] * main_correlation.shape[1]}")
    print(f"Universal neurons (threshold > {threshold}): {len(universal_neurons)}")
    print(f"Percentage universal: {(len(universal_neurons) / (main_correlation.shape[0] * main_correlation.shape[1])) * 100:.2f}%")
    
    # Print counts at different thresholds
    print(f"\nUniversal neuron counts at different thresholds:")
    for thresh, data in analysis['universal_counts'].items():
        print(f"  {thresh}: {data['count']} ({data['percentage']:.2f}%)")
    
    return main_correlation, baseline_correlation, universal_neurons, analysis


def load_correlation_results(filepath):
    """Load correlation results from disk."""
    return torch.load(filepath, map_location='cpu')


def compare_universal_neurons_across_checkpoints(model_name, dataset, checkpoints, 
                                                save_path='results/checkpoint_analysis'):
    """Compare universal neurons across different training checkpoints."""
    
    from dataset_utilities import setup_model_and_dataset
    
    print(f"Comparing universal neurons across checkpoints: {checkpoints}")
    
    results = {}
    
    for i, ckpt in enumerate(checkpoints):
        print(f"\nProcessing checkpoint {ckpt}...")
        
        # Load model at checkpoint
        model, data = setup_model_and_dataset(model_name, dataset, checkpoint=ckpt)
        
        if i == 0:
            reference_model = model
            reference_ckpt = ckpt
        else:
            # Compare with reference checkpoint
            print(f"Comparing checkpoint {ckpt} with reference {reference_ckpt}")
            
            main_corr, baseline_corr, universal_neurons, analysis = run_universal_neuron_analysis(
                reference_model, model, data, save_path=save_path
            )
            
            results[f"ckpt_{reference_ckpt}_vs_{ckpt}"] = {
                'universal_neurons': universal_neurons,
                'analysis': analysis,
                'main_correlation': main_corr,
                'baseline_correlation': baseline_corr
            }
    
    return results