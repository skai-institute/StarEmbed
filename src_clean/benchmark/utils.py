"""
Unified utility functions for benchmark scripts.
Provides consistent embedding processing for different types of embeddings:
- Time series foundation model embeddings (Chronos, Moirai, Astromer) with time dimension
- Handcrafted features (no time dimension)
- Pre-computed/random embeddings (already processed)
"""
import numpy as np
from typing import Dict, Any, Union, Tuple
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


def process_embeddings(
    example: Dict[str, Any], 
    scenario: str = "avg", 
    hand_crafted: bool = False,
    return_separate: bool = True,
    target_bands: list = None
) -> Dict[str, Any]:
    """
    Unified function to process embeddings from different sources with support for any bands.
    
    Args:
        example: Dictionary containing embedding data
        scenario: How to combine bands - "concat", "avg", or specific band name
        hand_crafted: Whether the embeddings are handcrafted features
        return_separate: If True, return {band}_embedding for each band separately
                        If False, return combined avg_embedding
        target_bands: Specific bands to use (default: auto-detect from data)
    
    Returns:
        Dictionary with processed embeddings added
    """
    # Auto-detect available bands if not specified
    if target_bands is None:
        target_bands = get_available_bands(example)
    
    # Validate scenario
    valid_scenarios = ["concat", "avg"] + target_bands
    assert scenario in valid_scenarios, f"Invalid scenario: {scenario}. Valid options: {valid_scenarios}"
    
    # Step 1: Extract and process embeddings for each available band
    processed_embeddings = {}
    
    for band in target_bands:
        # Try different naming patterns for embeddings
        emb_raw = None
        
        # Pattern 1: embeddings_{band} (raw time series)
        if f"embeddings_{band}" in example:
            emb_raw = example[f"embeddings_{band}"]
        # Pattern 2: {band}_embedding (pre-computed)
        elif f"{band}_embedding" in example:
            emb_raw = example[f"{band}_embedding"]
        
        if emb_raw is None:
            print(f"Warning: No embedding data found for band '{band}', skipping...")
            continue
            
        # Step 2: Convert to numpy and squeeze
        emb_array = np.squeeze(np.array(emb_raw, dtype=np.float32))
        
        # Step 3: Handle time dimension averaging for time series embeddings
        if hand_crafted:
            # Handcrafted features: use directly (no time dimension)
            avg_emb = emb_array
        else:
            # Time series or pre-computed embeddings
            if emb_array.ndim > 1:
                # Time series embeddings: average over time dimension
                avg_emb = emb_array.mean(0)
            else:
                # Pre-computed/random embeddings: use directly
                avg_emb = emb_array
        
        processed_embeddings[band] = avg_emb
        
        # Step 4: Store processed embeddings separately if requested
        if return_separate:
            example[f'{band}_embedding'] = avg_emb
    
    # Step 5: Combine bands according to scenario
    if len(processed_embeddings) == 0:
        raise ValueError("No valid embedding data found for any band")
    
    if scenario == "concat":
        # Concatenate all available bands in sorted order
        sorted_bands = sorted(processed_embeddings.keys())
        combined = np.concatenate([processed_embeddings[band] for band in sorted_bands], axis=0)
    elif scenario == "avg":
        # Average all available bands
        band_arrays = list(processed_embeddings.values())
        combined = np.mean(band_arrays, axis=0)
    elif scenario in processed_embeddings:
        # Use specific band
        combined = processed_embeddings[scenario]
    else:
        # Fallback to first available band if requested band not found
        first_band = sorted(processed_embeddings.keys())[0]
        print(f"Warning: Band '{scenario}' not found, using '{first_band}' instead")
        combined = processed_embeddings[first_band]
    
    if not return_separate:
        # Store combined embedding (for clustering)
        example['avg_embedding'] = combined
    else:
        # Store combined for scenario-based processing
        example['combined_embedding'] = combined
    
    return example


def process_embeddings_batch(
    batch: Dict[str, Any], 
    hand_crafted: bool = False,
    target_bands: list = None
) -> Dict[str, Any]:
    """
    Batch version for faster processing of multiple examples with support for any bands.
    
    Args:
        batch: Dictionary containing batch of embedding data
        hand_crafted: Whether the embeddings are handcrafted features
        target_bands: Specific bands to process (default: auto-detect from first example)
    
    Returns:
        Dictionary with processed {band}_embedding arrays for each available band
    """
    # Auto-detect available bands from first example if not specified
    if target_bands is None:
        first_example = {key: batch[key][0] if isinstance(batch[key], list) and len(batch[key]) > 0 else batch[key] 
                        for key in batch.keys()}
        target_bands = get_available_bands(first_example)
    
    # Initialize storage for each band
    band_embeddings = {band: [] for band in target_bands}
    
    # Get batch size from any available embedding field
    batch_size = None
    for band in target_bands:
        if f"embeddings_{band}" in batch:
            batch_size = len(batch[f"embeddings_{band}"])
            break
        elif f"{band}_embedding" in batch:
            batch_size = len(batch[f"{band}_embedding"])
            break
    
    if batch_size is None:
        raise ValueError("No embedding data found in batch")
    
    # Process each example in the batch
    for idx in range(batch_size):
        for band in target_bands:
            # Try different naming patterns for embeddings
            emb_raw = None
            
            # Pattern 1: embeddings_{band} (raw time series)
            if f"embeddings_{band}" in batch:
                emb_raw = batch[f"embeddings_{band}"][idx]
            # Pattern 2: {band}_embedding (pre-computed)
            elif f"{band}_embedding" in batch:
                emb_raw = batch[f"{band}_embedding"][idx]
            
            if emb_raw is None:
                print(f"Warning: No embedding data found for band '{band}' at index {idx}, skipping...")
                continue
                
            # Convert to numpy and squeeze
            emb_array = np.squeeze(np.array(emb_raw, dtype=np.float32))
            
            # Handle time dimension averaging for time series embeddings
            if hand_crafted:
                # Handcrafted features: use directly
                avg_emb = emb_array
            else:
                # Time series or pre-computed embeddings
                if emb_array.ndim > 1:
                    # Time series embeddings: average over time
                    avg_emb = emb_array.mean(0)
                else:
                    # Pre-computed embeddings: use directly
                    avg_emb = emb_array
                    
            band_embeddings[band].append(avg_emb)
    
    # Store processed embeddings in batch format
    for band in target_bands:
        if band_embeddings[band]:  # Only add if we have data for this band
            batch[f'{band}_embedding'] = band_embeddings[band]
    
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


# Convenience functions for specific use cases
def prepare_classification_embeddings(example, scenario: str = "avg", hand_crafted: bool = False):
    """Convenience wrapper for classification tasks."""
    return process_embeddings(example, scenario=scenario, hand_crafted=hand_crafted, return_separate=True)


def prepare_clustering_embeddings(example, concat: bool = True, hand_crafted: bool = False):
    """Convenience wrapper for clustering tasks."""
    scenario = "concat" if concat else "avg"
    return process_embeddings(example, scenario=scenario, hand_crafted=hand_crafted, return_separate=False)


# Legacy functions for backward compatibility (these handle the special clustering case)
def cal_avg_embedding(example, concat=False, hand_crafted=False, target_bands=None):
    """
    Legacy function for clustering - maintains the original interface but with multi-band support.
    
    Args:
        example: Dictionary containing embedding data
        concat: If True, concatenate bands; if False, average them
        hand_crafted: Whether the embeddings are handcrafted features
        target_bands: Specific bands to use (default: auto-detect)
    
    Returns:
        Dictionary with avg_embedding added
    """
    # Auto-detect available bands if not specified
    if target_bands is None:
        target_bands = get_available_bands(example)
    
    # Process embeddings for each available band
    processed_bands = {}
    
    for band in target_bands:
        # Handle the special clustering data structure
        emb_raw = None
        if f'embeddings_{band}' in example:
            emb_raw = example[f'embeddings_{band}']
        elif f'{band}_embedding' in example:
            emb_raw = example[f'{band}_embedding']
        
        if emb_raw is None:
            continue
            
        emb_array = np.array(emb_raw)
        
        if len(emb_array) == 1:
            # Single time series case (shape: (1, time_steps, embedding_dim))
            avg_band_embedding = np.mean(emb_array.squeeze(0), axis=0)
        else:
            # Multiple time series or handcrafted features
            if not hand_crafted:
                avg_band_embedding = np.mean(emb_array, axis=0)
            else:
                avg_band_embedding = emb_array
        
        processed_bands[band] = avg_band_embedding
    
    # Combine bands according to concat parameter
    if len(processed_bands) == 0:
        raise ValueError("No valid embedding data found for any band")
    
    if concat:
        # Concatenate all available bands in sorted order
        sorted_bands = sorted(processed_bands.keys())
        avg_embedding = np.concatenate([processed_bands[band] for band in sorted_bands])
    else:
        # Average all available bands
        band_arrays = list(processed_bands.values())
        avg_embedding = np.mean(band_arrays, axis=0)
    
    example['avg_embedding'] = avg_embedding
    return example
