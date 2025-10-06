#!/usr/bin/env python3
"""
Simple script to squeeze redundant dimensions from chronos embeddings.
Uses lambda functions for clean, fast batch processing.
"""

import numpy as np
import shutil
from pathlib import Path
from datasets import load_from_disk
import argparse

def squeeze_embeddings_lambda(dataset_path, split_name):
    """Squeeze embeddings using lambda function - clean and fast."""
    split_path = Path(dataset_path) / split_name
    
    if not split_path.exists():
        print(f"Split {split_name} not found")
        return False
    
    print(f"Processing {split_name}...")
    dataset = load_from_disk(str(split_path))
    
    # Check if squeezing is needed
    if len(dataset) > 0:
        first_example = dataset[0]
        needs_squeezing = any(
            f'embeddings_{band}' in first_example and 
            len(np.array(first_example[f'embeddings_{band}']).shape) == 3
            for band in ['g', 'r']
        )
        
        if not needs_squeezing:
            print(f"  No squeezing needed for {split_name}")
            return True
    
    # Create backup
    backup_path = split_path.with_suffix('.backup')
    if not backup_path.exists():
        print(f"  Creating backup...")
        split_path.rename(backup_path)
    
    # Squeeze using lambda - this is the magic!
    squeezed = dataset.map(
        lambda batch: {
            **batch,  # Keep all existing fields
            **{
                f'embeddings_{band}': [
                    np.squeeze(np.array(emb), axis=0).tolist() 
                    if len(np.array(emb).shape) == 3 and np.array(emb).shape[0] == 1 
                    else emb
                    for emb in batch[f'embeddings_{band}']
                ]
                for band in ['g', 'r'] 
                if f'embeddings_{band}' in batch
            }
        },
        batched=True,
        batch_size=3000,  # Larger batch for speed
        desc=f"Squeezing {split_name}",
        num_proc=10
    )
    
    # Save to temporary location first (can't overwrite loaded dataset)
    temp_path = split_path.with_suffix('.temp')
    squeezed.save_to_disk(str(temp_path))
    
    # Remove original and rename temp to original
    if split_path.exists():
        shutil.rmtree(str(split_path))
    temp_path.rename(split_path)
    
    print(f"  ✓ Squeezed and saved {split_name}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Squeeze chronos embeddings with lambda")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--base_path", 
                       default="/projects/b1094/StarEmbed/embeddings/descriptive_name_embeddings")
    
    args = parser.parse_args()
    
    dataset_path = Path(args.base_path) / args.dataset
    splits = ['train', 'validation', 'test', 'anom']
    
    print(f"Processing {args.dataset}")
    for split in splits:
        squeeze_embeddings_lambda(dataset_path, split)
    
    print("✓ All done!")

if __name__ == "__main__":
    main()
