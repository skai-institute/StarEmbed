#!/usr/bin/env python3
"""
Create Chronos embeddings for MACHO light curves validation split.

Transforms:
  Input:  hf_macho_70-10-20/ (preprocessed splits)
  Output: csdr1_raw4_catflags_filtered_embs_chronos_tiny_macho_ctx{ctx}_band{band}/

Process: Filters by band and length, normalizes time series data, 
generates embeddings using Chronos T5/Bolt models.

Usage:
  python create_chronos_embeddings.py --ctx 64 --model t5
  python create_chronos_embeddings.py --ctx 128 --model bolt
"""

import pandas as pd
import torch
from chronos import BaseChronosPipeline
import datasets
from tqdm import tqdm
import numpy as np
from functools import partial
import argparse

def length_filter(example, min_length=160):
    """Filter examples by minimum length requirement."""
    field_length = len(example['target'])
    
    if field_length < min_length:
        print(f"datapoint: has length {field_length}")
        return False
        
    return True

def filter_single_band(datapoint, band='r'):
    """Filter datapoints that have the required band data."""
    if band == 'gr':
        if datapoint['bands_data']['g'] is None or datapoint['bands_data']['r'] is None:
            print(f"datapoint: has no g or r")
            return False
    else:
        if datapoint['bands_data'][band] is None:
            print(f"datapoint: has no {band}")
            return False
    return True

def filter_single_length_datapoint(datapoint):
    """Filter out datapoints with length 1 in g or r bands."""
    if len(datapoint['bands_data']['g']['target']) == 1 or len(datapoint['bands_data']['r']['target']) == 1:
        print(f"datapoint: has length 1")
        return False
    return True

def add_embeddings(example, pipeline, ctx=64, band='r'):
    """Generate Chronos embeddings for each band in the example."""
    if band == 'gr':
        bands = ['g', 'r']
    else:
        bands = [band]
        
    for band in bands:
        # Normalize the data
        target = np.array(example['bands_data'][band]['target'])
        mu = np.mean(target)
        sigma = np.std(target)
        normalized_target = (target - mu) / sigma

        seq_len = len(target)
        pad_len = max(0, ctx - seq_len)

        # Left-pad target
        if pad_len:
            normalized_target = np.concatenate([np.zeros(pad_len, dtype=float), normalized_target])
        normalized_target = normalized_target[-ctx:]  # keep exactly ctx
        
        # Convert to tensor for model input
        context = torch.tensor(normalized_target)
        
        # Generate embeddings
        embeddings, tokenizer_state = pipeline.embed(context)
        
        # Add embeddings to the example (convert tensor to numpy for storage)
        example[f"embeddings_{band}"] = embeddings.to(torch.float32).cpu().numpy() # maybe need to squeeze here
    
    return example

def main():
    parser = argparse.ArgumentParser(description="Generate Chronos embeddings for MACHO dataset")
    parser.add_argument("--ctx", type=int, default=64, help="Context length for embeddings")
    parser.add_argument("--model", type=str, default='t5', choices=['t5', 'bolt'], 
                       help="Model type: t5 or bolt")
    args = parser.parse_args()
    
    # Model selection
    # d_model: tiny models: 256, mini models: 384, bolt_small: 512, bolt_base: 768, small: 512, base: 768, large: 1024
    if args.model == 't5':
        model_name = "amazon/chronos-t5-tiny"
    elif args.model == 'bolt':
        model_name = "amazon/chronos-bolt-tiny"
    else:
        model_name = args.model

    print(f"Loading Chronos model: {model_name}")
    # Load model
    pipeline = BaseChronosPipeline.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )

    print("Loading MACHO dataset...")
    # Load dataset
    dataset = datasets.load_from_disk('/projects/p32795/hongyu/hf_macho_70-10-20')

    dataset_dict = {}
    if isinstance(dataset, datasets.DatasetDict):
        for split in ['validation']:
            dataset_split = dataset[split]
            # dataset_split = dataset_split.select(range(5))  # just for testing

            print(f"Generating embeddings for {split}...")
            ctx = args.ctx
            band = 'r'  # MACHO only has r-band
            
            print(f"Filtering by band: {band}")
            dataset_split = dataset_split.filter(lambda x: filter_single_band(x, band))
            
            print(f"Processing {len(dataset_split)} examples with context length {ctx}")
            updated_dataset = dataset_split.map(
                partial(add_embeddings, pipeline=pipeline, ctx=ctx, band=band), 
                desc="Generating embeddings"
            )
            dataset_dict[split] = updated_dataset
            
        # Save the updated dataset
        dataset_to_save = datasets.DatasetDict(dataset_dict)
        
    elif isinstance(dataset, datasets.Dataset):
        band = 'C'
        ctx = args.ctx
        dataset_to_save = dataset.filter(lambda x: filter_single_band(x, band))
        dataset_to_save = dataset_to_save.map(
            partial(add_embeddings, pipeline=pipeline, ctx=ctx, band=band), 
            desc="Generating embeddings"
        )

    output_path = f"/projects/p32795/hongyu/csdr1_raw4_catflags_filtered_embs_chronos_tiny_macho_ctx{ctx}_band{band}"

    print(f"Saving dataset with embeddings to {output_path}")
    dataset_to_save.save_to_disk(output_path)

    print("Done!")

if __name__ == "__main__":
    main() 