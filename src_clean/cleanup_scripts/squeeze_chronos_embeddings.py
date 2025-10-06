#!/usr/bin/env python3
"""
Script to remove redundant dimensions from chronos dataset embeddings.
Converts embeddings from (1, time_length, dim) to (time_length, dim) format.
"""

import os
import json
from pathlib import Path
from datasets import load_from_disk, Dataset, DatasetDict
import numpy as np
from tqdm import tqdm
import argparse


def is_chronos_dataset(dataset_path):
    """Check if this is a chronos dataset by looking at the name."""
    return "chronos" in str(dataset_path).lower()


def squeeze_embeddings_batch(batch):
    """Squeeze the first dimension of embeddings if it's size 1. Batch processing version."""
    for band in ['g', 'r']:
        embedding_key = f'embeddings_{band}'
        if embedding_key in batch:
            embeddings_list = batch[embedding_key]
            squeezed_batch = []
            
            for embeddings in embeddings_list:
                if isinstance(embeddings, list) and len(embeddings) > 0:
                    # Convert to numpy for easier manipulation
                    emb_array = np.array(embeddings)
                    # If first dimension is 1, squeeze it out
                    if len(emb_array.shape) == 3 and emb_array.shape[0] == 1:
                        squeezed = np.squeeze(emb_array, axis=0)
                        squeezed_batch.append(squeezed.tolist())
                    else:
                        squeezed_batch.append(embeddings)
                else:
                    squeezed_batch.append(embeddings)
            
            batch[embedding_key] = squeezed_batch
    return batch


def update_dataset_info(dataset_path, split_name):
    """Update the dataset_info.json to reflect the new embedding structure."""
    info_path = dataset_path / split_name / "dataset_info.json"
    
    if info_path.exists():
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        # Update the embedding features structure
        for band in ['g', 'r']:
            embedding_key = f'embeddings_{band}'
            if embedding_key in info['features']:
                # Change from 3-level nested to 2-level nested structure
                # From: List[List[List[float32]]] to List[List[float32]]
                info['features'][embedding_key] = {
                    "feature": {
                        "feature": {
                            "dtype": "float32",
                            "_type": "Value"
                        },
                        "_type": "List"
                    },
                    "_type": "List"
                }
        
        # Write back the updated info
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"Updated dataset_info.json for {split_name}")


def process_split(dataset_path, split_name, dry_run=False):
    """Process a single split of the dataset."""
    split_path = dataset_path / split_name
    
    if not split_path.exists():
        print(f"Split {split_name} not found in {dataset_path}")
        return False
    
    print(f"Processing split: {split_name}")
    
    try:
        # Load the split
        dataset = load_from_disk(str(split_path))
        
        # Check if embeddings need squeezing by examining the first example
        if len(dataset) > 0:
            first_example = dataset[0]
            needs_squeezing = False
            
            for band in ['g', 'r']:
                embedding_key = f'embeddings_{band}'
                if embedding_key in first_example:
                    embeddings = first_example[embedding_key]
                    if isinstance(embeddings, list) and len(embeddings) > 0:
                        emb_array = np.array(embeddings)
                        if len(emb_array.shape) == 3 and emb_array.shape[0] == 1:
                            needs_squeezing = True
                            print(f"  Found {embedding_key} with shape {emb_array.shape} - needs squeezing")
                        else:
                            print(f"  Found {embedding_key} with shape {emb_array.shape} - no squeezing needed")
            
            if not needs_squeezing:
                print(f"  No squeezing needed for {split_name}")
                return True
            
            if dry_run:
                print(f"  [DRY RUN] Would squeeze embeddings for {len(dataset)} examples in {split_name}")
                return True
            
            # Apply the squeezing transformation with batch processing
            print(f"  Squeezing embeddings for {len(dataset)} examples...")
            squeezed_dataset = dataset.map(
                squeeze_embeddings_batch, 
                batched=True, 
                batch_size=1000,
                desc=f"Squeezing {split_name}"
            )
            
            # Save the modified dataset
            backup_path = split_path.with_suffix('.backup')
            if not backup_path.exists():
                print(f"  Creating backup at {backup_path}")
                os.rename(str(split_path), str(backup_path))
            
            squeezed_dataset.save_to_disk(str(split_path))
            print(f"  Saved squeezed dataset to {split_path}")
            
            # Update dataset_info.json
            update_dataset_info(dataset_path, split_name)
            
        return True
        
    except Exception as e:
        print(f"Error processing split {split_name}: {e}")
        return False


def process_dataset(dataset_path, dry_run=False):
    """Process all splits in a dataset."""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"Dataset path not found: {dataset_path}")
        return False
    
    if not is_chronos_dataset(dataset_path):
        print(f"Skipping non-chronos dataset: {dataset_path.name}")
        return True
    
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_path.name}")
    print(f"{'='*60}")
    
    splits = ['train', 'validation', 'test', 'anom']
    success_count = 0
    
    for split in splits:
        if process_split(dataset_path, split, dry_run):
            success_count += 1
    
    print(f"\nCompleted {success_count}/{len(splits)} splits for {dataset_path.name}")
    return success_count == len(splits)


def main():
    parser = argparse.ArgumentParser(description="Squeeze redundant dimensions from chronos dataset embeddings")
    parser.add_argument("--base-path", 
                       default="/projects/b1094/StarEmbed/embeddings/descriptive_name_embeddings",
                       help="Base path to embeddings directory")
    parser.add_argument("--dataset", 
                       help="Specific dataset name to process (if not provided, processes all chronos datasets)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    base_path = Path(args.base_path)
    
    if not base_path.exists():
        print(f"Base path not found: {base_path}")
        return 1
    
    if args.dataset:
        # Process specific dataset
        dataset_path = base_path / args.dataset
        success = process_dataset(dataset_path, args.dry_run)
        return 0 if success else 1
    else:
        # Process all chronos datasets
        chronos_datasets = [d for d in base_path.iterdir() 
                           if d.is_dir() and is_chronos_dataset(d)]
        
        if not chronos_datasets:
            print("No chronos datasets found")
            return 1
        
        print(f"Found {len(chronos_datasets)} chronos datasets:")
        for dataset in chronos_datasets:
            print(f"  - {dataset.name}")
        
        if args.dry_run:
            print("\n[DRY RUN MODE] - No changes will be made")
        
        success_count = 0
        for dataset_path in chronos_datasets:
            if process_dataset(dataset_path, args.dry_run):
                success_count += 1
        
        print(f"\n{'='*60}")
        print(f"SUMMARY: Successfully processed {success_count}/{len(chronos_datasets)} datasets")
        print(f"{'='*60}")
        
        return 0 if success_count == len(chronos_datasets) else 1


if __name__ == "__main__":
    exit(main())
