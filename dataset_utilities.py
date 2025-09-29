import os
import numpy as np
import torch
import datasets
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate


def get_model_family(model_name):
    """Determine model family from model name."""
    if 'gpt2' in model_name.lower():
        return 'gpt2'
    elif 'pythia' in model_name.lower():
        return 'pythia'
    else:
        raise ValueError(f'Unsupported model family for: {model_name}')

def download_and_save_pile_dataset(model_name, output_dir='token_datasets', 
                                   hf_dataset='monology/pile-uncopyrighted',
                                   split='train', max_samples=None):
    """
    Download pile-uncopyrighted from HuggingFace and tokenize it for the given model.
    
    Args:
        model_name: Name of the model to use for tokenization
        output_dir: Base directory to save tokenized dataset
        hf_dataset: HuggingFace dataset identifier
        split: Dataset split to use
        max_samples: Maximum number of samples to process (None for all)
    """
    import datasets
    from transformers import AutoTokenizer
    
    print(f"Loading dataset {hf_dataset} (split: {split})...")
    dataset = datasets.load_dataset(hf_dataset, split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Load tokenizer
    print(f"Loading tokenizer for {model_name}...")
    model = HookedTransformer.from_pretrained(model_name, device='cpu')
    tokenizer = model.tokenizer
    max_length = model.cfg.n_ctx
    
    print(f"Tokenizing dataset (max_length={max_length})...")
    
    def tokenize_function(examples):
        # Tokenize the texts
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
        )
        return {'tokens': tokenized['input_ids']}
    
    # Tokenize in batches
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Save to disk
    model_family = get_model_family(model_name)
    save_path = os.path.join(output_dir, model_family, 'pile')
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Saving tokenized dataset to {save_path}...")
    tokenized_dataset.save_to_disk(save_path)
    
    print(f"Dataset saved successfully!")
    print(f"Total sequences: {len(tokenized_dataset)}")
    print(f"Path: {save_path}")
    
    return save_path

def load_tokenized_dataset(dataset_path, model_config, batch_size=32):
    """Load and prepare tokenized dataset."""
    tokenized_dataset = datasets.load_from_disk(dataset_path)
    
    max_length = model_config.n_ctx
    
    # Ensure tokens are properly formatted and clipped to vocab size
    tokenized_dataset = tokenized_dataset.map(
        lambda x: {
            "tokens": [
                np.clip(
                    seq[:max_length] + [0] * (max_length - len(seq)),  # Pad with zeros
                    0,
                    model_config.d_vocab - 1
                )
                for seq in x["tokens"]
            ]
        },
        batched=True,
        batch_size=batch_size
    )
    
    tokenized_dataset.set_format(type='torch', columns=['tokens'])
    
    if 'tokens' not in tokenized_dataset.column_names:
        raise KeyError("Missing tokens column in dataset")
    
    return tokenized_dataset


def create_dataloader(dataset, batch_size=32, shuffle=False):
    """Create DataLoader from tokenized dataset."""
    return DataLoader(
        dataset['tokens'], 
        batch_size=batch_size, 
        shuffle=shuffle
    )


def filter_valid_tokens(tokens, model, filter_padding=True, filter_newlines=True):
    """Filter out padding tokens and optionally newlines."""
    valid_mask = torch.ones_like(tokens, dtype=torch.bool)
    
    if filter_padding:
        # Filter padding tokens (usually token id 0 or pad_token_id)
        pad_token_id = getattr(model.tokenizer, 'pad_token_id', 0)
        if pad_token_id is None:
            pad_token_id = 0
        valid_mask &= (tokens != pad_token_id)
    
    if filter_newlines:
        # Filter newline tokens
        try:
            newline_token = model.to_single_token('\n')
            valid_mask &= (tokens != newline_token)
        except:
            # If can't get newline token, skip this filter
            pass
    
    return valid_mask


def get_dataset_path(model_name, dataset_name, base_dir='token_datasets'):
    """Get the path to a tokenized dataset."""
    model_family = get_model_family(model_name)
    return os.path.join(base_dir, model_family, dataset_name)


def prepare_pile_dataset(model_name, dataset_name='pile', base_dir='token_datasets'):
    """Prepare a Pile dataset for the given model."""
    dataset_path = get_dataset_path(model_name, dataset_name, base_dir)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    return dataset_path


def validate_dataset_tokens(dataset, model_config):
    """Validate that dataset tokens are within valid range."""
    sample_batch = dataset['tokens'][:10]  # Check first 10 sequences
    
    # Check for tokens outside vocab range
    max_token = sample_batch.max().item()
    min_token = sample_batch.min().item()
    
    if max_token >= model_config.d_vocab:
        print(f"Warning: Found tokens ({max_token}) >= vocab size ({model_config.d_vocab})")
    
    if min_token < 0:
        print(f"Warning: Found negative token ids ({min_token})")
    
    return True


def chunk_dataset_by_memory(dataset, chunk_size=1000):
    """Yield chunks of dataset to manage memory usage."""
    total_size = len(dataset['tokens'])
    
    for i in range(0, total_size, chunk_size):
        end_idx = min(i + chunk_size, total_size)
        chunk = {
            'tokens': dataset['tokens'][i:end_idx]
        }
        yield chunk, i, end_idx


def estimate_memory_usage(model_config, batch_size, sequence_length):
    """Estimate memory usage for activations."""
    # Rough estimate: layers * neurons * batch_size * sequence_length * 2 bytes (float16)
    memory_bytes = (model_config.n_layers * 
                   model_config.d_mlp * 
                   batch_size * 
                   sequence_length * 2)
    
    memory_gb = memory_bytes / (1024**3)
    return memory_gb


def setup_model_and_dataset(model_name, dataset_name, batch_size=32, 
                           checkpoint=None, device='auto',
                           auto_download=False):
    """Setup model and dataset together with validation."""
    
    # Load model
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        checkpoint_value=checkpoint
    )
    model.eval()
    torch.set_grad_enabled(False)
    
    # Load dataset
    dataset_path = prepare_pile_dataset(model_name, dataset_name)
    
    # Check if dataset exists, if not and auto_download is True, download it
    if not os.path.exists(dataset_path):
        if auto_download:
            print(f"Dataset not found at {dataset_path}")
            print("Downloading and tokenizing pile-uncopyrighted dataset...")
            dataset_path = download_and_save_pile_dataset(model_name)
        else:
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}. "
                f"Run with auto_download=True or manually download using "
                f"download_and_save_pile_dataset('{model_name}')"
            )
    
    dataset = load_tokenized_dataset(dataset_path, model.cfg, batch_size)
    
    # Validate
    validate_dataset_tokens(dataset, model.cfg)
    
    # Estimate memory
    seq_len = dataset['tokens'][0].shape[0] if len(dataset['tokens']) > 0 else 512
    memory_gb = estimate_memory_usage(model.cfg, batch_size, seq_len)
    
    print(f"Estimated memory usage: {memory_gb:.2f} GB")
    print(f"Model: {model_name} (checkpoint: {checkpoint})")
    print(f"Dataset: {dataset_name} ({len(dataset['tokens'])} sequences)")
    print(f"Device: {device}")
    
    return model, dataset