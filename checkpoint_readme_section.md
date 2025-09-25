# Checkpoint Analysis Support

The universal neurons pipeline now supports analyzing models at different training checkpoints to study how universality evolves during training.

## Usage

### Single Checkpoint Analysis

```bash
# Analyze at a specific checkpoint
python example_usage.py --checkpoint 1000

# Test mode with checkpoint
python example_usage.py --test --checkpoint 5000
```

### Multi-Checkpoint Comparison

```bash
# Compare across multiple checkpoints
python example_usage.py --compare-checkpoints 1000 2000 5000 10000
```

### Programmatic Usage

```python
from universal_neurons_pipeline import run_universal_neurons_analysis

# Analyze at checkpoint 1000
results = run_universal_neurons_analysis(
    model_names=models,
    dataset_path=dataset_path,
    checkpoint_value=1000  # Specify checkpoint
)

# Results are saved to universal_neurons_results_checkpoint_1000/
```

## Output Organization

Results for different checkpoints are saved in separate directories:
- `universal_neurons_results/` (no checkpoint specified)
- `universal_neurons_results_checkpoint_1000/`
- `universal_neurons_results_checkpoint_2000/`
- etc.

Each directory contains:
- `*_checkpoint_1000_neuron_stats.csv` - Neuron statistics
- `correlations_checkpoint_1000.pt` - Correlation matrices
- `universal_neurons_checkpoint_1000.csv` - Universal neuron mappings
- `plots/dashboard_checkpoint_1000.html` - Interactive dashboard

## Checkpoint Comparison Tools

### Loading Results Across Checkpoints

```python
from dataset_utilities import load_analysis_results, CheckpointComparisonVisualizer

# Load results from multiple checkpoints
results_by_checkpoint = {}
for checkpoint in [1000, 2000, 5000]:
    results_dir = f"universal_neurons_results_checkpoint_{checkpoint}"
    results_by_checkpoint[checkpoint] = load_analysis_results(results_dir)

# Create comparison visualizer
comparator = CheckpointComparisonVisualizer(results_by_checkpoint)
```

### Visualization Tools

```python
# Plot universality evolution
comparator.plot_universality_evolution("universality_evolution.png")

# Plot layer distribution changes
comparator.plot_layer_distribution_evolution("layer_evolution.png")

# Create interactive comparison dashboard
comparator.create_checkpoint_comparison_dashboard("checkpoint_comparison.html")
```

### Analysis Tools

```python
from dataset_utilities import analyze_checkpoint_progression, find_persistent_universal_neurons

# Analyze progression statistics
progression_df = analyze_checkpoint_progression(results_by_checkpoint)
print(progression_df)

# Find neurons that remain universal across checkpoints
persistent_neurons = find_persistent_universal_neurons(
    results_by_checkpoint, 
    min_checkpoints=3
)
print(f"Found {len(persistent_neurons)} persistent universal neurons")
```

## Checkpoint Values

The `checkpoint_value` parameter can be:
- Integer: `1000`, `5000` (training steps)
- String: `"step_1000"`, `"epoch_5"` (depends on model's checkpoint naming)

Check the specific model's documentation for available checkpoint formats.

## Research Applications

### Training Dynamics Study
- Track when universal neurons first appear
- Study stability/instability of universality during training
- Identify critical training phases for universality emergence

### Curriculum Effects
- Compare universality at different training stages
- Study effect of learning rate schedules on universality
- Analyze relationship between loss and universality

### Model Comparison
- Compare universality development across different model sizes
- Study effect of different training procedures
- Analyze checkpoints from models with different hyperparameters

## Example Research Workflow

```python
# 1. Analyze multiple checkpoints
checkpoints = [100, 500, 1000, 2000, 5000, 10000]
for checkpoint in checkpoints:
    run_universal_neurons_analysis(
        model_names=models,
        dataset_path=dataset_path,
        checkpoint_value=checkpoint
    )

# 2. Load and compare results
results_by_checkpoint = {}
for checkpoint in checkpoints:
    results_dir = f"universal_neurons_results_checkpoint_{checkpoint}"
    results_by_checkpoint[checkpoint] = load_analysis_results(results_dir)

# 3. Analyze progression
progression = analyze_checkpoint_progression(results_by_checkpoint)
progression.to_csv("universality_progression.csv")

# 4. Find persistent neurons
persistent = find_persistent_universal_neurons(results_by_checkpoint, min_checkpoints=4)
persistent.to_csv("persistent_universal_neurons.csv")

# 5. Create visualizations
comparator = CheckpointComparisonVisualizer(results_by_checkpoint)
comparator.create_checkpoint_comparison_dashboard("training_analysis.html")
```

This enables studying questions like:
- **When do universal neurons first emerge?**
- **Do they remain stable throughout training?**
- **How does universality correlate with training loss?**
- **Which neurons are most persistent across training?**