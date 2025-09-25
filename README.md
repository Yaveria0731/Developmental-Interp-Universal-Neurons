# Universal Neurons Analysis - Setup Guide

This streamlined codebase replicates the universal neurons experiment from the research paper, focusing on finding neurons that exhibit similar behavior across different GPT2-small models from Stanford CRFM.

## Quick Start

1. **Install dependencies:**
```bash
pip install torch transformer-lens datasets pandas numpy matplotlib seaborn plotly tqdm einops
```

2. **Run quick test:**
```bash
python example_usage.py --test
```

3. **Run full analysis:**
```bash
python example_usage.py
```

## Requirements

### Python Packages
```
torch>=1.9.0
transformer-lens>=1.14.0
datasets>=2.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
tqdm>=4.62.0
einops>=0.4.0
```

### Hardware Requirements
- **Memory**: 16GB+ RAM recommended for full analysis
- **GPU**: CUDA-capable GPU recommended (but not required)
- **Storage**: ~5GB for datasets and results

## Core Functionalities

### 1. Neuron Statistics Generation (`NeuronStatsGenerator`)

Generates comprehensive statistics for all neurons in a model:

- **Weight statistics**: Input/output norms, L2 penalties, cosine similarities
- **Vocabulary composition**: Statistics about neuron interactions with vocabulary
- **Activation statistics**: Mean, variance, and sparsity on dataset

**Usage:**
```python
from universal_neurons_pipeline import NeuronStatsGenerator

generator = NeuronStatsGenerator("stanford-crfm/alias-gpt2-small-x21")
stats_df = generator.generate_full_neuron_dataframe("path/to/dataset")
```

### 2. Correlation Computation (`NeuronCorrelationComputer`)

Computes Pearson correlations between neurons across different models:

- Streams activations to handle memory constraints  
- Computes correlations between all model pairs
- Saves correlation matrices for analysis

**Usage:**
```python
from universal_neurons_pipeline import NeuronCorrelationComputer

models = ["model1", "model2", "model3"]
correlator = NeuronCorrelationComputer(models)
correlations = correlator.compute_all_correlations("path/to/dataset")
```

### 3. Universal Neuron Identification (`UniversalNeuronAnalyzer`)

Identifies neurons that are highly correlated across multiple models:

- Finds neurons above correlation threshold in multiple models
- Analyzes properties of universal vs regular neurons
- Provides detailed statistics and comparisons

**Usage:**
```python
from universal_neurons_pipeline import UniversalNeuronAnalyzer

analyzer = UniversalNeuronAnalyzer(correlations, neuron_stats)
universal_df = analyzer.identify_universal_neurons(threshold=0.6, min_models=3)
```

### 4. Visualization and Analysis (`UniversalNeuronVisualizer`)

Creates comprehensive visualizations:

- Correlation distribution plots
- Property comparison histograms  
- Interactive correlation heatmaps
- Network graphs of universal connections
- Interactive dashboard

**Usage:**
```python
from dataset_utilities import UniversalNeuronVisualizer

visualizer = UniversalNeuronVisualizer(results)
visualizer.create_analysis_dashboard("dashboard.html")
```

## Stanford CRFM Models

The analysis focuses on these 5 GPT2-small models from Stanford CRFM:

1. `stanford-crfm/alias-gpt2-small-x21`
2. `stanford-crfm/battlestar-gpt2-small-x49` 
3. `stanford-crfm/caprica-gpt2-small-x81`
4. `stanford-crfm/darkmatter-gpt2-small-x343`
5. `stanford-crfm/expanse-gpt2-small-x777`

All models have the same architecture but different training runs, making them ideal for studying universal computational patterns.

## File Structure

```
universal_neurons_analysis/
├── universal_neurons_pipeline.py    # Core analysis pipeline
├── dataset_utilities.py             # Dataset creation and visualization 
├── example_usage.py                 # Complete example script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file

# Generated during analysis:
├── datasets/                        # Tokenized datasets
├── universal_neurons_results/       # Analysis results
│   ├── universal_neurons.csv        # Universal neuron mappings
│   ├── universal_analysis.csv       # Statistical analysis
│   ├── correlations.pt             # Correlation matrices
│   ├── *_neuron_stats.csv          # Per-model neuron statistics
│   └── plots/                      # Visualizations
│       ├── dashboard.html          # Interactive dashboard
│       ├── correlation_distribution.png
│       ├── properties_comparison.png
│       └── *.png                   # Various plots
```

## Expected Results

Based on the original paper, you should expect:

- **Universality Rate**: ~10-30% of neurons showing high correlation across models
- **Layer Distribution**: Universal neurons more common in middle-to-late layers
- **Property Differences**: Universal neurons tend to have:
  - Higher vocabulary variance (more selective)
  - Higher kurtosis (more peaked activation distributions)
  - Different weight norm patterns

## Advanced Usage

### Custom Analysis

```python
# Load existing results
from dataset_utilities import load_analysis_results
results = load_analysis_results("universal_neurons_results")

# Find neurons similar to a specific one
from dataset_utilities import find_similar_neurons
similar = find_similar_neurons(
    results['neuron_stats']['model_name'],
    reference_layer=10, 
    reference_neuron=500,
    top_k=10
)

# Compute importance scores
from dataset_utilities import compute_neuron_importance_scores
importance = compute_neuron_importance_scores(stats_df)
```

### Memory Optimization

For large-scale analysis, consider:

1. **Reduce dataset size**: Use fewer tokens (1M-2M instead of 5M)
2. **Batch processing**: Process model pairs sequentially
3. **Layer-wise computation**: Compute correlations layer by layer
4. **Precision reduction**: Use float16 instead of float32

```python
# Memory-efficient correlation computation
correlator = NeuronCorrelationComputer(models, device='cpu')  # Use CPU
correlations = correlator.compute_all_correlations(
    dataset_path, batch_size=16  # Smaller batches
)
```

### Custom Thresholds

Adjust parameters based on your research questions:

```python
# More stringent universality
results = run_universal_neurons_analysis(
    model_names=models,
    dataset_path=dataset_path,
    correlation_threshold=0.8,  # Higher threshold
    min_models=4                # Must appear in 4+ models
)

# More permissive universality  
results = run_universal_neurons_analysis(
    model_names=models,
    dataset_path=dataset_path,
    correlation_threshold=0.4,  # Lower threshold
    min_models=2                # Must appear in 2+ models
)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size: `batch_size=8`
   - Use CPU: `device='cpu'`
   - Reduce dataset size

2. **Models not downloading**:
   - Check internet connection
   - Ensure HuggingFace access
   - Try smaller models first: `["gpt2", "distilgpt2"]`

3. **No universal neurons found**:
   - Lower correlation threshold: `correlation_threshold=0.3`
   - Reduce minimum models: `min_models=2`
   - Check if models are actually different

4. **Slow correlation computation**:
   - Use smaller dataset: `n_tokens=500_000`
   - Increase batch size: `batch_size=64`
   - Use GPU if available

### Performance Tips

- **Use GPU**: Significant speedup for correlation computation
- **Smaller datasets**: 1M tokens often sufficient for finding patterns
- **Checkpoint results**: Save intermediate results to avoid re-computation
- **Parallel processing**: Run different model pairs on different GPUs

## Research Applications

This pipeline enables various research directions:

### 1. Universality Studies
- Compare universality across different model families
- Study how universality changes with model size
- Investigate universality in fine-tuned vs. pre-trained models

### 2. Interpretability Research
- Analyze what universal neurons compute
- Study their role in language understanding
- Compare their activation patterns on different tasks

### 3. Model Analysis
- Understand similarities/differences between training runs
- Identify robust vs. fragile computational patterns
- Study the effect of training dynamics on universality

### 4. Intervention Experiments
```python
# Use universal neurons for targeted interventions
universal_neurons = results['universal_neurons']
top_universal = universal_neurons.nlargest(10, 'mean_correlation')

# Extract neuron coordinates for interventions
intervention_targets = []
for _, row in top_universal.iterrows():
    layer = row['reference_layer'] 
    neuron = row['reference_neuron']
    intervention_targets.append((layer, neuron))

# Use these coordinates in activation patching experiments
```

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{universal_neurons_2024,
  title={Universal Neurons in GPT2 Language Models},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## Contributing

To extend this codebase:

1. **Add new correlation metrics**: Extend `NeuronCorrelationComputer`
2. **Add new visualizations**: Extend `UniversalNeuronVisualizer`  
3. **Add new model families**: Modify model loading in pipeline
4. **Add new analysis methods**: Extend `UniversalNeuronAnalyzer`

## License

This code is provided for research purposes. Please respect the licenses of the underlying libraries (transformer-lens, PyTorch, etc.).