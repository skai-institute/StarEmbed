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

def _squeeze_list(x):
    # x is usually a nested list; when shape is (1, L, D), it's [ [...], ... ] with len==1
    return x[0] if isinstance(x, list) and len(x) == 1 else x

def _squeeze_batch(batch):
    out = dict(batch)
    for band in ("g", "r"):
        key = f"embeddings_{band}"
        if key in batch:
            xs = batch[key]
            out[key] = [_squeeze_list(v) for v in xs]
    return out

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

    # ---- RUN MAP FIRST (do NOT rename/move the folder yet) ----  # CHANGED
    # Optional: only pass embedding columns to workers (less IPC)   # CHANGED
    input_cols = [k for k in ('embeddings_g','embeddings_r') if k in dataset.column_names]  # CHANGED

    squeezed = dataset.map(
        _squeeze_batch,
        batched=True,
        batch_size=500,                 # tune as needed
        desc=f"Squeezing {split_name}",
        num_proc=4,                     # was 16; fewer procs = less IPC/memory  # CHANGED
    )
    
    # Save to temporary location first (can't overwrite loaded dataset)
    temp_path = split_path.with_suffix('.temp')
    if temp_path.exists():
        shutil.rmtree(str(temp_path))
    squeezed.save_to_disk(str(temp_path))

    # NOW create backup and swap in the temp                      # CHANGED
    backup_path = split_path.with_suffix('.backup')
    if not backup_path.exists():
        print(f"  Creating backup...")
        split_path.rename(backup_path)
    else:
        # If a backup already exists, remove current original to make room
        shutil.rmtree(str(split_path))

    # Replace original with temp
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
