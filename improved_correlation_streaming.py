"""
Improved NeuronCorrelationComputer with streaming support for full dataset usage
"""

import torch
import datasets
import numpy as np
from torch.utils.data import DataLoader
import einops
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import gc

class StreamingNeuronCorrelationComputer:
    """Compute correlations between neurons across different models with streaming support"""
    
    def __init__(self, model_names: List[str], device: str = "cuda"):
        self.model_names = model_names
        self.device = device
        self.models = {}
        
        print("Loading models...")
        for name in model_names:
            print(f"Loading {name}...")
            model = HookedTransformer.from_pretrained(name, device=device)
            model.eval()
            self.models[name] = model
        torch.set_grad_enabled(False)
    
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
    
    def compute_pairwise_correlation_streaming(
        self, 
        model1_name: str, 
        model2_name: str,
        dataset_path: str, 
        batch_size: int = 16,
        max_samples: Optional[int] = None,
        chunk_size: int = 1000,
        use_fp16: bool = False
    ) -> torch.Tensor:
        """
        Compute Pearson correlation between all neuron pairs using streaming
        
        Args:
            model1_name: First model name
            model2_name: Second model name  
            dataset_path: Path to tokenized dataset
            batch_size: Batch size for processing
            max_samples: Maximum number of samples to use (None for full dataset)
            chunk_size: Number of batches to process before checkpointing
            use_fp16: Use half precision to save memory
        """
        
        model1 = self.models[model1_name]
        model2 = self.models[model2_name]
        
        # Load dataset
        tokenized_dataset = datasets.load_from_disk(dataset_path)
        
        # Limit dataset if specified
        if max_samples is not None:
            tokenized_dataset = tokenized_dataset.select(range(min(max_samples, len(tokenized_dataset))))
        
        dataloader = DataLoader(
            tokenized_dataset['tokens'], 
            batch_size=batch_size, 
            shuffle=False
        )
        
        # Initialize correlation computation with appropriate dtype
        dtype = torch.float16 if use_fp16 else torch.float64
        device_dtype = torch.float32  # Keep on device calculations in fp32
        
        # Initialize accumulators on CPU to save GPU memory
        m1_sum = torch.zeros(model1.cfg.n_layers, model1.cfg.d_mlp, dtype=dtype)
        m1_sum_sq = torch.zeros(model1.cfg.n_layers, model1.cfg.d_mlp, dtype=dtype)
        m2_sum = torch.zeros(model2.cfg.n_layers, model2.cfg.d_mlp, dtype=dtype) 
        m2_sum_sq = torch.zeros(model2.cfg.n_layers, model2.cfg.d_mlp, dtype=dtype)
        
        # For cross products, we'll compute layer by layer to save memory
        cross_sums = {}
        for l1 in range(model1.cfg.n_layers):
            for l2 in range(model2.cfg.n_layers):
                cross_sums[(l1, l2)] = torch.zeros(
                    model1.cfg.d_mlp, model2.cfg.d_mlp, dtype=dtype
                )
        
        n_samples = 0
        batch_count = 0
        
        print(f"Computing correlation between {model1_name} and {model2_name}")
        print(f"Using full dataset: {len(tokenized_dataset)} sequences")
        
        for batch in tqdm(dataloader, desc="Processing batches"):
            batch = batch.to(self.device)
            
            # Get activations
            acts1 = self.get_activations(model1, batch)  # [l1, d1, t]
            acts2 = self.get_activations(model2, batch)  # [l2, d2, t]
            
            # Filter padding tokens
            valid_mask = (batch.flatten() != 0)
            acts1 = acts1[:, :, valid_mask].to(device_dtype)
            acts2 = acts2[:, :, valid_mask].to(device_dtype)
            
            n_tokens = acts1.shape[-1]
            n_samples += n_tokens
            
            # Update correlation statistics - move to CPU to save GPU memory
            m1_sum += acts1.sum(dim=-1).cpu().to(dtype)
            m1_sum_sq += (acts1**2).sum(dim=-1).cpu().to(dtype)
            m2_sum += acts2.sum(dim=-1).cpu().to(dtype)
            m2_sum_sq += (acts2**2).sum(dim=-1).cpu().to(dtype)
            
            # Compute cross products layer by layer to manage memory
            for l1 in range(model1.cfg.n_layers):
                acts1_layer = acts1[l1]  # [d1, t]
                for l2 in range(model2.cfg.n_layers):
                    acts2_layer = acts2[l2]  # [d2, t]
                    
                    # Compute cross product: [d1, d2]
                    cross_product = torch.mm(acts1_layer, acts2_layer.T)
                    cross_sums[(l1, l2)] += cross_product.cpu().to(dtype)
            
            batch_count += 1
            
            # Clear GPU memory periodically
            if batch_count % chunk_size == 0:
                torch.cuda.empty_cache()
                gc.collect()
                print(f"Processed {batch_count} batches, {n_samples:,} tokens")
        
        print(f"Total samples processed: {n_samples:,}")
        
        # Compute final correlations
        print("Computing final correlations...")
        correlations = torch.zeros(
            model1.cfg.n_layers, model1.cfg.d_mlp,
            model2.cfg.n_layers, model2.cfg.d_mlp,
            dtype=torch.float32
        )
        
        for l1 in range(model1.cfg.n_layers):
            for l2 in range(model2.cfg.n_layers):
                # Convert to float32 for final computation
                m1_mean = (m1_sum[l1] / n_samples).float()
                m2_mean = (m2_sum[l2] / n_samples).float()
                
                # Numerator: E[XY] - E[X]E[Y]
                numerator = (cross_sums[(l1, l2)] / n_samples).float() - torch.outer(m1_mean, m2_mean)
                
                # Denominator: sqrt(Var[X] * Var[Y])
                var1 = (m1_sum_sq[l1] / n_samples).float() - m1_mean**2
                var2 = (m2_sum_sq[l2] / n_samples).float() - m2_mean**2
                
                # Clamp variances to avoid numerical issues
                var1 = torch.clamp(var1, min=1e-8)
                var2 = torch.clamp(var2, min=1e-8)
                
                denominator = torch.outer(torch.sqrt(var1), torch.sqrt(var2))
                correlations[l1, :, l2, :] = numerator / denominator
        
        # Clamp correlations to valid range
        correlations = torch.clamp(correlations, min=-1.0, max=1.0)
        
        return correlations
    
    def compute_all_correlations_streaming(
        self, 
        dataset_path: str,
        batch_size: int = 16,
        max_samples: Optional[int] = None,
        use_fp16: bool = False,
        save_individual: bool = True,
        output_dir: str = "correlations"
    ) -> Dict[Tuple[str, str], torch.Tensor]:
        """Compute correlations between all model pairs with streaming"""
        
        correlations = {}
        
        for i, model1 in enumerate(self.model_names):
            for j, model2 in enumerate(self.model_names[i+1:], i+1):
                pair = (model1, model2)
                print(f"\nComputing correlation for pair: {pair}")
                
                try:
                    corr_matrix = self.compute_pairwise_correlation_streaming(
                        model1, model2, dataset_path,
                        batch_size=batch_size,
                        max_samples=max_samples,
                        use_fp16=use_fp16
                    )
                    correlations[pair] = corr_matrix
                    
                    # Save individual correlation matrices to disk
                    if save_individual:
                        import os
                        os.makedirs(output_dir, exist_ok=True)
                        filename = f"{output_dir}/corr_{model1.replace('/', '_')}_{model2.replace('/', '_')}.pt"
                        torch.save(corr_matrix, filename)
                        print(f"Saved correlation matrix to {filename}")
                    
                    # Clear memory
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"OOM error for pair {pair}. Try reducing batch_size or using fp16=True")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        return correlations


def memory_efficient_correlation_analysis(
    model_names: List[str],
    dataset_path: str,
    output_dir: str = "universal_neurons_results",
    correlation_threshold: float = 0.5,
    min_models: int = 3,
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    use_fp16: bool = True
):
    """
    Memory-efficient version of the universal neurons analysis
    
    Args:
        model_names: List of model names to analyze
        dataset_path: Path to tokenized dataset
        output_dir: Output directory for results
        correlation_threshold: Threshold for universal neurons
        min_models: Minimum number of models a neuron must appear in
        batch_size: Batch size (reduce if OOM)
        max_samples: Maximum samples to use (None for full dataset)
        use_fp16: Use half precision to save memory
    """
    
    import os
    from universal_neurons_pipeline import NeuronStatsGenerator, UniversalNeuronAnalyzer
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Generate neuron statistics (unchanged)
    print("=" * 50)
    print("STEP 1: GENERATING NEURON STATISTICS")
    print("=" * 50)
    
    neuron_stats = {}
    for model_name in model_names:
        print(f"\nProcessing {model_name}...")
        generator = NeuronStatsGenerator(model_name)
        stats_df = generator.generate_full_neuron_dataframe(dataset_path)
        neuron_stats[model_name] = stats_df
        
        # Save individual model stats
        stats_df.to_csv(f"{output_dir}/{model_name.replace('/', '_')}_neuron_stats.csv")
        print(f"Saved stats for {model_name}: {len(stats_df)} neurons")
    
    # Step 2: Compute correlations with streaming
    print("\n" + "=" * 50)
    print("STEP 2: COMPUTING CORRELATIONS WITH STREAMING")  
    print("=" * 50)
    
    correlator = StreamingNeuronCorrelationComputer(model_names)
    correlations = correlator.compute_all_correlations_streaming(
        dataset_path, 
        batch_size=batch_size,
        max_samples=max_samples,
        use_fp16=use_fp16,
        save_individual=True,
        output_dir=f"{output_dir}/individual_correlations"
    )
    
    # Save all correlations
    correlation_file = f"{output_dir}/correlations.pt"
    torch.save(correlations, correlation_file)
    print(f"Saved all correlations to {correlation_file}")
    
    # Continue with rest of analysis...
    print("\n" + "=" * 50)
    print("STEP 3: IDENTIFYING UNIVERSAL NEURONS")
    print("=" * 50)
    
    analyzer = UniversalNeuronAnalyzer(correlations, neuron_stats)
    universal_df = analyzer.identify_universal_neurons(
        threshold=correlation_threshold,
        min_models=min_models
    )
    
    universal_file = f"{output_dir}/universal_neurons.csv"
    universal_df.to_csv(universal_file, index=False)
    print(f"Found {len(universal_df)} universal neurons")
    print(f"Saved to {universal_file}")
    
    # Step 4: Analyze properties
    analysis_df = analyzer.analyze_universal_properties(universal_df)
    analysis_file = f"{output_dir}/universal_analysis.csv"
    analysis_df.to_csv(analysis_file, index=False)
    print(f"Saved analysis to {analysis_file}")
    
    return {
        'neuron_stats': neuron_stats,
        'correlations': correlations,
        'universal_neurons': universal_df,
        'analysis': analysis_df
    }


# Example usage with different memory configurations:

def run_with_memory_config(models, dataset_path, memory_level="medium"):
    """Run analysis with different memory configurations"""
    
    configs = {
        "high_memory": {
            "batch_size": 32,
            "max_samples": None,
            "use_fp16": False
        },
        "medium_memory": {
            "batch_size": 16, 
            "max_samples": None,
            "use_fp16": True
        },
        "low_memory": {
            "batch_size": 8,
            "max_samples": 2000000,  # 2M samples
            "use_fp16": True
        },
        "very_low_memory": {
            "batch_size": 4,
            "max_samples": 1000000,  # 1M samples
            "use_fp16": True
        }
    }
    
    config = configs[memory_level]
    print(f"Running with {memory_level} configuration: {config}")
    
    return memory_efficient_correlation_analysis(
        model_names=models,
        dataset_path=dataset_path,
        **config
    )
