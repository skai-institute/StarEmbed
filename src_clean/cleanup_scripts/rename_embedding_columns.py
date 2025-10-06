#!/usr/bin/env python3
"""
Rename embedding columns to match standard naming convention.

Changes:
  g_embedding -> embeddings_g
  r_embedding -> embeddings_r
  i_embedding -> embeddings_i (if exists)
  z_embedding -> embeddings_z (if exists)

Usage:
  python rename_embedding_columns.py --dataset DATASET_PATH
"""

import argparse
import shutil
from pathlib import Path
from datasets import load_from_disk
from tqdm.auto import tqdm

def rename_columns_batch(batch, rename_mapping):
    """Rename columns in a batch."""
    result = {}
    
    # Process each column in the batch
    for key, values in batch.items():
        # Check if this column needs to be renamed
        if key in rename_mapping:
            # Use the new name
            new_name = rename_mapping[key]
            result[new_name] = values
            # Don't include the old name
        else:
            # Keep the column as-is
            result[key] = values
    
    return result

def process_split(dataset_path, split_name, rename_mapping):
    """Process a single split of the dataset."""
    split_path = Path(dataset_path) / split_name
    
    if not split_path.exists():
        print(f"Split {split_name} not found")
        return False
    
    print(f"Processing {split_name}...")
    dataset = load_from_disk(str(split_path))
    
    # Check what columns exist and need renaming
    if len(dataset) > 0:
        first_example = dataset[0]
        columns_to_rename = []
        
        for old_name, new_name in rename_mapping.items():
            if old_name in first_example:
                columns_to_rename.append((old_name, new_name))
                print(f"  Found {old_name} -> will rename to {new_name}")
        
        if not columns_to_rename:
            print(f"  No columns to rename in {split_name}")
            return True
    
    # Create backup
    backup_path = split_path.with_suffix('.backup_rename')
    if not backup_path.exists():
        print(f"  Creating backup...")
        shutil.copytree(str(split_path), str(backup_path))
    
    # Rename columns - explicitly remove old columns that are being renamed
    old_columns_to_remove = [old for old, new in rename_mapping.items() if old in dataset[0]]
    
    processed = dataset.map(
        lambda batch: rename_columns_batch(batch, rename_mapping),
        batched=True,
        batch_size=1000,
        desc=f"Renaming columns in {split_name}",
        num_proc=4,
        remove_columns=old_columns_to_remove
    )
    
    # Save to temporary location first
    temp_path = split_path.with_suffix('.temp')
    processed.save_to_disk(str(temp_path))
    
    # Replace original
    if split_path.exists():
        shutil.rmtree(str(split_path))
    temp_path.rename(split_path)
    
    print(f"  ✓ Renamed columns in {split_name}")
    
    # Show what we have now
    if len(processed) > 0:
        first_example = processed[0]
        print(f"  Columns after renaming:")
        for old_name, new_name in rename_mapping.items():
            if new_name in first_example:
                if hasattr(first_example[new_name], '__len__') and not isinstance(first_example[new_name], str):
                    try:
                        import numpy as np
                        shape = np.array(first_example[new_name]).shape
                        print(f"    {new_name}: shape {shape}")
                    except:
                        print(f"    {new_name}: present")
                else:
                    print(f"    {new_name}: present")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Rename embedding columns to standard naming")
    parser.add_argument("--dataset", required=True, 
                       help="Full path to dataset directory")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"],
                       help="Splits to process")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be renamed without making changes")
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 1
    
    # Define the renaming mapping
    rename_mapping = {
        "g_embedding": "embeddings_g",
        "r_embedding": "embeddings_r", 
        "i_embedding": "embeddings_i",
        "z_embedding": "embeddings_z"
    }
    
    print(f"Processing dataset: {dataset_path}")
    print(f"Splits: {args.splits}")
    print(f"Renaming mapping: {rename_mapping}")
    
    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        for split in args.splits:
            split_path = dataset_path / split
            if split_path.exists():
                dataset = load_from_disk(str(split_path))
                if len(dataset) > 0:
                    first_example = dataset[0]
                    print(f"\n{split} split:")
                    for old_name, new_name in rename_mapping.items():
                        if old_name in first_example:
                            print(f"  Would rename: {old_name} -> {new_name}")
        return 0
    
    success_count = 0
    for split in args.splits:
        if process_split(dataset_path, split, rename_mapping):
            success_count += 1
    
    print(f"\n✓ Successfully processed {success_count}/{len(args.splits)} splits")
    print("Standard naming convention applied:")
    for old_name, new_name in rename_mapping.items():
        print(f"  {old_name} -> {new_name}")
    
    return 0 if success_count == len(args.splits) else 1

if __name__ == "__main__":
    exit(main())
