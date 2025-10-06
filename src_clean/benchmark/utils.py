"""
Unified utility functions for benchmark scripts.
Provides consistent embedding processing for different types of embeddings:
- Time series foundation model embeddings (Chronos, Moirai, Astromer) with time dimension
- Handcrafted features (no time dimension)  
- Pre-computed/random embeddings (already processed)
"""
import numpy as np
from typing import Dict, Any, Union, Tuple, List
from functools import partial


def get_available_bands(example: Dict[str, Any]) -> list:
    """
    Detect available bands from the dataset example.
    
    Args:
        example: Dictionary containing embedding data
    
    Returns:
        List of available band names (e.g., ['g', 'r'] or ['g', 'r', 'i'])
    """
    # Check for embedding columns (pre-computed embeddings)
    embedding_bands = []
    for key in example.keys():
        if key.endswith('_embedding'):
            band = key.replace('_embedding', '')
            embedding_bands.append(band)
    
    if embedding_bands:
        return sorted(embedding_bands)
    
    # Check for embeddings_* columns (raw time series embeddings)
    embeddings_bands = []
    for key in example.keys():
        if key.startswith('embeddings_'):
            band = key.replace('embeddings_', '')
            embeddings_bands.append(band)
    
    if embeddings_bands:
        return sorted(embeddings_bands)
    
    # Check bands_data structure 
    if 'bands_data' in example and isinstance(example['bands_data'], dict):
        return sorted(example['bands_data'].keys())
    
    # Default fallback to g, r
    return ['g', 'r']


def compute_embedding(
    example: Dict[str, Any], 
    band_combination: str = "avg", 
    hand_crafted: bool = False,
    target_bands: list = None,
    scaler=None,
    return_format: str = "combined"
) -> Union[np.ndarray, Dict[str, Any]]:
    """
    **THE MAIN UNIFIED EMBEDDING FUNCTION**
    
    This is the single function that ALL scripts should use for embedding processing.
    Handles pre-computed avg_embedding, time averaging, band combination, and multi-band support.
    
    Args:
        example: Dictionary containing embedding data
        band_combination: How to combine bands - "concat", "avg", or specific band name ("g", "r", "i", "z")
        hand_crafted: Whether the embeddings are handcrafted features (no time dimension)
        target_bands: Specific bands to use (default: auto-detect from data)
        scaler: Optional StandardScaler for normalization
        return_format: "combined" returns just the embedding array, "dict" returns dict with embedding added
    
    Returns:
        If return_format="combined": numpy array of combined embedding
        If return_format="dict": example dict with 'avg_embedding' field added
    """
    
    # **NEW**: Check if avg_embedding is already computed (fast path)
    if "avg_embedding" in example and example["avg_embedding"] is not None:
        avg_emb = example["avg_embedding"]
        
        if band_combination == "concat":
            # Concatenate all available bands in sorted order
            vectors = []
            for band in sorted(avg_emb.keys()):
                if avg_emb[band] is not None:
                    vectors.append(np.array(avg_emb[band], dtype=np.float32))
            combined_embedding = np.concatenate(vectors) if vectors else None
            
        elif band_combination == "avg":
            # Average all available bands
            vectors = []
            for band in avg_emb.keys():
                if avg_emb[band] is not None:
                    vectors.append(np.array(avg_emb[band], dtype=np.float32))
            combined_embedding = np.mean(vectors, axis=0) if vectors else None
            
        elif band_combination in avg_emb:
            # Return specific band
            if avg_emb[band_combination] is not None:
                combined_embedding = np.array(avg_emb[band_combination], dtype=np.float32)
            else:
                combined_embedding = None
        else:
            # Fallback to first available band
            available_bands = [b for b in avg_emb.keys() if avg_emb[b] is not None]
            if available_bands:
                combined_embedding = np.array(avg_emb[available_bands[0]], dtype=np.float32)
            else:
                combined_embedding = None
        
        if combined_embedding is None:
            raise ValueError(f"No valid embedding data found in avg_embedding for combination '{band_combination}'")
        
        # Apply scaling if provided
        if scaler is not None:
            combined_embedding = scaler.transform(combined_embedding.reshape(1, -1)).flatten()
        
        # Return in requested format
        if return_format == "combined":
            return combined_embedding
        else:  # return_format == "dict"
            example['combined_embedding'] = combined_embedding
            return example
    
    # **FALLBACK**: Original processing for datasets without avg_embedding
    # Auto-detect available bands if not specified
    if target_bands is None:
        target_bands = get_available_bands(example)
    
    # Validate band_combination
    valid_combinations = ["concat", "avg"] + target_bands
    if band_combination not in valid_combinations:
        print(f"Warning: '{band_combination}' not in {valid_combinations}, using first available band")
        band_combination = target_bands[0] if target_bands else "g"
    
    # Step 1: Process each available band
    processed_bands = {}
    
    for band in target_bands:
        # Extract raw embedding data
        emb_raw = None
        if f"embeddings_{band}" in example:
            emb_raw = example[f"embeddings_{band}"]
        elif f"{band}_embedding" in example:
            emb_raw = example[f"{band}_embedding"]
        
        if emb_raw is None:
            continue
            
        # Convert to numpy array
        emb_array = np.array(emb_raw, dtype=np.float32)
        
        # Handle different data formats
        if hand_crafted:
            # Handcrafted features: use directly
            processed_emb = emb_array
        else:
            # Foundation model embeddings: handle time averaging
            if emb_array.ndim == 3:  # Shape: (1, time, dim)
                processed_emb = np.mean(emb_array.squeeze(0), axis=0)
            elif emb_array.ndim == 2:  # Shape: (time, dim)
                processed_emb = np.mean(emb_array, axis=0)
            else:  # Shape: (dim,) - already processed
                processed_emb = emb_array
        
        processed_bands[band] = processed_emb
    
    # Step 2: Combine bands
    if len(processed_bands) == 0:
        raise ValueError("No valid embedding data found for any band")
    
    if band_combination == "concat":
        # Concatenate all available bands in sorted order
        sorted_bands = sorted(processed_bands.keys())
        combined_embedding = np.concatenate([processed_bands[band] for band in sorted_bands])
    elif band_combination == "avg":
        # Element-wise average of all bands
        band_arrays = list(processed_bands.values())
        combined_embedding = np.mean(band_arrays, axis=0)
    elif band_combination in processed_bands:
        # Use specific band
        combined_embedding = processed_bands[band_combination]
    else:
        # Fallback to first available band
        sorted_bands = sorted(processed_bands.keys())
        combined_embedding = processed_bands[sorted_bands[0]]
    
    # Step 3: Apply scaling if provided
    if scaler is not None:
        combined_embedding = scaler.transform(combined_embedding.reshape(1, -1)).flatten()
    
    # Step 4: Return in requested format
    if return_format == "combined":
        return combined_embedding
    else:  # return_format == "dict"
        example['avg_embedding'] = combined_embedding
        return example


def compute_embedding_batch(
    batch: Dict[str, Any], 
    band_combination: str = "concat",
    hand_crafted: bool = False,
    target_bands: list = None,
    return_format: str = "dict"
) -> Dict[str, Any]:
    """
    **TRUE VECTORIZED BATCH PROCESSING**
    
    Processes entire batches efficiently using vectorized numpy operations.
    This is the proper way to use batched=True for performance gains.
    
    Args:
        batch: Dictionary containing batch of embedding data
        band_combination: How to combine bands - "concat", "avg", or specific band
        hand_crafted: Whether the embeddings are handcrafted features
        target_bands: Specific bands to use (auto-detect if None)
        
    Returns:
        Dictionary with 'avg_embedding' containing processed embeddings for all examples
    """
    # Auto-detect available bands from first example if not specified
    print("auto-detecting bands...")
    if target_bands is None:
        first_example = {key: batch[key][0] if isinstance(batch[key], list) and len(batch[key]) > 0 else batch[key] 
                        for key in batch.keys()}
        target_bands = get_available_bands(first_example)
    
    # Validate band_combination
    valid_combinations = ["concat", "avg"] + target_bands
    if band_combination not in valid_combinations:
        print(f"Warning: '{band_combination}' not in {valid_combinations}, using first available band")
        band_combination = target_bands[0] if target_bands else "g"
    
    # Step 1: Process each band with TRUE vectorization
    processed_bands = {}

    print("Processing bands...")
    for band in target_bands:
        # Extract embeddings for this band from entire batch
        if f"embeddings_{band}" in batch:
            emb_batch = batch[f"embeddings_{band}"]
        elif f"{band}_embedding" in batch:
            emb_batch = batch[f"{band}_embedding"]
        else:
            continue
            
        # TRUE VECTORIZED PROCESSING - no loops!
        # Convert entire batch to numpy array at once
        emb_array_batch = np.array(emb_batch, dtype=np.float32)
        
        if hand_crafted:
            # Handcrafted features: use directly (already correct shape)
            processed_embeddings = emb_array_batch
        else:
            # Foundation model embeddings: handle time averaging vectorized
            if emb_array_batch.ndim == 4:  # Shape: (batch, 1, time, dim)
                # Remove singleton dimension and average over time for entire batch
                processed_embeddings = np.mean(emb_array_batch.squeeze(1), axis=1)
            elif emb_array_batch.ndim == 3:  # Shape: (batch, time, dim)
                # Average over time dimension for entire batch
                processed_embeddings = np.mean(emb_array_batch, axis=1)
            else:  # Shape: (batch, dim) - already processed
                processed_embeddings = emb_array_batch
        
        # Store processed embeddings for this band
        processed_bands[band] = processed_embeddings
    print("Finished processing bands.")
    print("Combining bands...")
    # Step 2: Combine bands across all examples at once (vectorized!)
    if len(processed_bands) == 0:
        raise ValueError("No valid embedding data found for any band")
    
    if band_combination == "concat":
        # Concatenate all available bands - vectorized across all examples
        sorted_bands = sorted(processed_bands.keys())
        combined_embeddings = np.concatenate([processed_bands[band] for band in sorted_bands], axis=1)
    elif band_combination == "avg":
        # Element-wise average of all bands - vectorized across all examples
        band_arrays = list(processed_bands.values())
        combined_embeddings = np.mean(band_arrays, axis=0)
    elif band_combination in processed_bands:
        # Use specific band
        combined_embeddings = processed_bands[band_combination]
    else:
        # Fallback to first available band
        sorted_bands = sorted(processed_bands.keys())
        combined_embeddings = processed_bands[sorted_bands[0]]
    
    # Convert to list format for HuggingFace datasets
    batch['avg_embedding'] = combined_embeddings.tolist()
    return batch


def remove_outliers(dataset, hand_crafted: bool = False):
    """
    Remove known outlier examples from the dataset.
    
    Args:
        dataset: HuggingFace dataset with train/validation/test splits
        hand_crafted: Whether using handcrafted features (different outlier indices)
    
    Returns:
        Dataset with outliers removed
    """
    if not hand_crafted:
        print("Removing outliers from time series embedding dataset")
        bad_idx_trn, bad_idx_val, bad_idx_tst = 23082, 473, 7880
        trn_idx_to_select = list(range(bad_idx_trn)) + list(range(bad_idx_trn+1, len(dataset["train"]))) 
        val_idx_to_select = list(range(bad_idx_val)) + list(range(bad_idx_val+1, len(dataset["validation"]))) 
        tst_idx_to_select = list(range(bad_idx_tst)) + list(range(bad_idx_tst+1, len(dataset["test"])))
    else:
        print("Removing outliers from hand-crafted feature dataset")
        bad_idx_trn = [3010, 9693, 16524, 22151]
        bad_idx_val = [449]
        bad_idx_tst = [1158]
        trn_idx_to_select = list(sorted(set(range(len(dataset["train"]))) - set(bad_idx_trn)))
        val_idx_to_select = list(sorted(set(range(len(dataset["validation"]))) - set(bad_idx_val)))
        tst_idx_to_select = list(sorted(set(range(len(dataset["test"]))) - set(bad_idx_tst)))

    dataset["train"] = dataset["train"].select(trn_idx_to_select)
    dataset["validation"] = dataset["validation"].select(val_idx_to_select)
    dataset["test"] = dataset["test"].select(tst_idx_to_select)
    
    print(f"Selected {len(dataset['train'])} train, {len(dataset['validation'])} validation, {len(dataset['test'])} test samples")
    return dataset


def get_label_mapping(dataset, sort_labels=True):
    """
    Get label mapping from descriptive class names in the dataset.
    Works with any dataset that has descriptive class names in 'class_str' field.
    
    Args:
        dataset: HuggingFace dataset with 'class_str' field containing descriptive names
        sort_labels: If True, sort labels alphabetically for consistency
    
    Returns:
        Tuple of (label2idx dict, text_labels list)
    """
    # Get unique class labels from the training split
    unique_labels = sorted(set(dataset['train']['class_str'])) if sort_labels else list(set(dataset['train']['class_str']))
    
    # Create mapping from label to index
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    return label2idx, unique_labels


def add_label_indices(dataset, num_proc: int = 4, sort_labels: bool = True):
    """
    Add numerical label indices to dataset splits based on descriptive class names.
    
    Args:
        dataset: HuggingFace dataset with descriptive class names in 'class_str'
        num_proc: Number of processes for mapping
        sort_labels: If True, sort labels alphabetically for consistency
    
    Returns:
        Dataset with label_idx column added
    """
    # Get label mapping directly from the dataset
    label2idx, text_labels = get_label_mapping(dataset, sort_labels=sort_labels)
    
    def add_label(example):
        return {"label_idx": label2idx[example["class_str"]]}
    
    standard_splits = ['train', 'validation', 'test']
    for split in standard_splits:
        if split in dataset:
            dataset[split] = dataset[split].map(add_label, num_proc=num_proc)
    
    return dataset, label2idx, text_labels


# ==============================================================================
# SIMPLIFIED INTERFACE - All scripts use these 4 functions only!
# ==============================================================================

# For clustering script - use this instead of local cal_avg_embedding
def cal_avg_embedding(example, concat=None, scenario=None, hand_crafted=False, target_bands=None):
    """For clustering script compatibility - supports both old concat parameter and new scenario parameter."""
    if scenario is not None:
        # New scenario-based approach (like classification scripts)
        band_combination = scenario
    elif concat is not None:
        # Old concat boolean approach (for backward compatibility)
        band_combination = "concat" if concat else "avg"
    else:
        # Default to concat if neither specified
        band_combination = "concat"
        
    return compute_embedding(example, band_combination=band_combination, 
                           hand_crafted=hand_crafted, target_bands=target_bands,
                           return_format="dict")

# For classification scripts - use this instead of ScenarioDataset logic  
def get_scenario_embedding(example, scenario, hand_crafted=False, scaler=None, target_bands=None):
    """For ScenarioDataset compatibility."""
    return compute_embedding(example, band_combination=scenario, 
                           hand_crafted=hand_crafted, target_bands=target_bands,
                           scaler=scaler, return_format="combined")

# For RF script - use lambda functions instead for better performance
def process_embeddings_batch(batch, band_combination="concat", hand_crafted=False):
    """
    For RF script compatibility - now with band combination control like other scripts.
    
    RECOMMENDED: Use lambda functions directly instead:
    dataset.map(lambda example: {"embedding": compute_embedding(example, band_combination="concat")})
    """
    return compute_embedding_batch(batch, band_combination=band_combination, hand_crafted=hand_crafted)
