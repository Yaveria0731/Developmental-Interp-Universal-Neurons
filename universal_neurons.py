"""
Universal Neurons Analysis - Efficient Implementation using Repository's Method
This implements the fast correlation computation approach from the original repository.
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
    Fast streaming correlation computer from the repository.
    Computes full correlation matrices efficiently.
    """
    def __init__(self, model_1, model_2, device='cpu'):
        m1_layers = model_1.cfg.n_layers
        m2_layers = model_2.cfg.n_layers
        m1_dmlp = model_1.cfg.d_mlp
        m2_dmlp = model_2.cfg.d_mlp
        self.device = device

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
        """Update correlation statistics with new batch"""
        for l1 in range(batch_1_acts.shape[0]):
            batch_1_acts_l1 = batch_1_acts[l1].to(torch.float32)

            for l2 in range(batch_2_acts.shape[0]):
                layerwise_result = einops.einsum(
                    batch_1_acts_l1, batch_2_acts[l2].to(torch.float32), 
                    'n1 t, n2 t -> n1 n2'
                )
                self.m1_m2_sum[l1, :, l2, :] += layerwise_result.cpu()

        self.m1_sum += batch_1_acts.sum(dim=-1).cpu()
        self.m1_sum_sq += (batch_1_acts**2).sum(dim=-1).cpu()
        self.m2_sum += batch_2_acts.sum(dim=-1).cpu()
        self.m2_sum_sq += (batch_2_acts**2).sum(dim=-1).cpu()

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


def generate_random_rotation_matrix(d_mlp: int, device: str = "cpu", seed: int = None) -> torch.Tensor:
    """Generate random orthogonal rotation matrix"""
    if seed is not None:
        torch.manual_seed(seed)
    
    random_matrix = torch.randn(d_mlp, d_mlp, dtype=torch.float32, device=device)
    Q, R = torch.linalg.qr(random_matrix)
    
    # Ensure proper rotation (det = 1)
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    
    return Q


class EfficientExcessCorrelationComputer:
    """
    Efficient excess correlation computer using the repository's approach:
    1. Compute all correlations first
    2. Calculate excess correlation from correlation matrices
    """
    
    def __init__(self, model_names: List[str], device: str = "cuda", 
                 checkpoint_value: Optional[Union[int, str]] = None,
                 n_rotation_samples: int = 5):
        self.model_names = model_names
        self.device = device if torch.cuda.is_available() else "cpu"
        self.checkpoint_value = checkpoint_value
        self.n_rotation_samples = n_rotation_samples
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
    
    def compute_correlation_matrix(self, model_a_name: str, model_b_name: str, 
                                 dataset_path: str, batch_size: int = 8,
                                 apply_rotation: bool = False, rotation_seed: int = None) -> torch.Tensor:
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
        if apply_rotation:
            rotation_matrix = generate_random_rotation_matrix(
                model_b.cfg.d_mlp, device=self.device, seed=rotation_seed
            )
        
        print(f"Computing correlations between {model_a_name} and {model_b_name}...")
        
        for batch in tqdm(dataloader, desc="Processing batches"):
            batch = batch.to(self.device)
            
            # Get activations
            acts_a = get_activations(model_a, batch, device=self.device)
            acts_b = get_activations(model_b, batch, device=self.device)
            
            # Apply rotation if needed
            if apply_rotation and rotation_matrix is not None:
                rotated_acts = []
                for l in range(acts_b.shape[0]):
                    # acts_b[l]: [neurons, tokens]
                    rotated = torch.matmul(rotation_matrix, acts_b[l])
                    rotated_acts.append(rotated)
                acts_b = torch.stack(rotated_acts, dim=0)
            
            # Update correlation statistics
            corr_computer.update_correlation_data(acts_a, acts_b)
        
        # Compute final correlation matrix
        correlation_matrix = corr_computer.compute_correlation()
        
        # Clear memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return correlation_matrix
    
    def compute_excess_correlations(self, dataset_path: str, batch_size: int = 8) -> pd.DataFrame:
        """
        Compute excess correlations using the repository's method:
        1. Compute all regular correlations
        2. Compute baseline correlations with rotations
        3. Calculate excess = regular - baseline
        """
        
        if len(self.model_names) < 2:
            raise ValueError("Need at least 2 models for comparison")
        
        reference_model = self.model_names[0]
        comparison_models = self.model_names[1:]
        
        print(f"Computing excess correlations with reference model: {reference_model}")
        
        # Get model info
        self._load_model(reference_model)
        ref_model = self.models[reference_model]
        n_layers = ref_model.cfg.n_layers
        d_mlp = ref_model.cfg.d_mlp
        
        excess_correlation_scores = []
        
        # Process each neuron in the reference model
        for layer in range(n_layers):
            for neuron in range(d_mlp):
                print(f"Processing neuron {layer}.{neuron}")
                
                regular_correlations = []
                baseline_correlations = []
                
                # Compute correlations with each comparison model
                for comp_model in comparison_models:
                    
                    # Regular correlation
                    print(f"  Computing regular correlation with {comp_model}")
                    regular_corr_matrix = self.compute_correlation_matrix(
                        reference_model, comp_model, dataset_path, batch_size, 
                        apply_rotation=False
                    )
                    
                    # Extract max correlation for this neuron
                    neuron_corrs = regular_corr_matrix[layer, neuron, :, :].flatten()
                    max_regular_corr = torch.max(neuron_corrs).item()
                    regular_correlations.append(max_regular_corr)
                    
                    # Baseline correlations with rotations
                    rotation_max_corrs = []
                    for rot_idx in range(self.n_rotation_samples):
                        print(f"  Computing baseline correlation {rot_idx+1}/{self.n_rotation_samples}")
                        
                        rotation_seed = layer * 10000 + neuron * 100 + rot_idx
                        baseline_corr_matrix = self.compute_correlation_matrix(
                            reference_model, comp_model, dataset_path, batch_size,
                            apply_rotation=True, rotation_seed=rotation_seed
                        )
                        
                        # Extract max correlation for this neuron
                        neuron_baseline_corrs = baseline_corr_matrix[layer, neuron, :, :].flatten()
                        max_baseline_corr = torch.max(neuron_baseline_corrs).item()
                        rotation_max_corrs.append(max_baseline_corr)
                    
                    # Average baseline across rotations
                    avg_baseline_corr = np.mean(rotation_max_corrs)
                    baseline_correlations.append(avg_baseline_corr)
                
                # Average across comparison models
                mean_regular_corr = np.mean(regular_correlations)
                mean_baseline_corr = np.mean(baseline_correlations)
                
                # Excess correlation
                excess_correlation = mean_regular_corr - mean_baseline_corr
                
                excess_correlation_scores.append({
                    'layer': layer,
                    'neuron': neuron,
                    'excess_correlation': excess_correlation,
                    'regular_correlation': mean_regular_corr,
                    'baseline_correlation': mean_baseline_corr,
                    'n_models_compared': len(comparison_models),
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
                                 batch_size: int = 8) -> Dict:
    """
    Run complete universal neurons analysis using efficient excess correlation method.
    """
    
    if checkpoint_value is not None:
        output_dir = f"{output_dir}_checkpoint_{checkpoint_value}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("UNIVERSAL NEURONS ANALYSIS - EFFICIENT EXCESS CORRELATION")
    if checkpoint_value is not None:
        print(f"CHECKPOINT: {checkpoint_value}")
    print("=" * 60)
    
    # Step 1: Compute excess correlations
    print("\nStep 1: Computing excess correlations...")
    correlator = EfficientExcessCorrelationComputer(
        model_names, checkpoint_value=checkpoint_value, 
        n_rotation_samples=n_rotation_samples
    )
    
    excess_correlation_df = correlator.compute_excess_correlations(
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
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"Models analyzed: {len(model_names)}")
    print(f"Total neurons: {len(excess_correlation_df)}")
    print(f"Universal neurons found: {len(universal_neurons_df)}")
    print(f"Results saved to: {output_dir}")
    
    return {
        'excess_correlation_scores': excess_correlation_df,
        'universal_neurons': universal_neurons_df,
        'checkpoint': checkpoint_value
    }


# Example usage
if __name__ == "__main__":
    # Test with smaller models first
    models = [
        "gpt2",
        "distilgpt2"
    ]
    
    # Create dataset
    dataset_path = create_tokenized_dataset(
        model_name=models[0],
        n_tokens=100000,  # Small for testing
        output_dir="datasets"
    )
    
    # Run analysis
    results = run_universal_neurons_analysis(
        model_names=models,
        dataset_path=dataset_path,
        excess_threshold=0.05,
        n_rotation_samples=3,
        batch_size=4
    )
    
    print("Analysis complete!")