#!/usr/bin/env python3
"""
Compute average embeddings over time dimension for all bands in a dataset.

Transforms:
  Input:  embeddings_g: (time, dim) or (1, time, dim)
          embeddings_r: (time, dim) or (1, time, dim)
  Output: avg_embedding: {"g": [dim], "r": [dim]}

Process: For each example, averages embeddings over time dimension
and stores result in new column as a dictionary by band.

Usage:
  python compute_avg_embeddings.py --dataset DATASET_NAME
  python compute_avg_embeddings.py --dataset DATASET_NAME --bands g r i
"""

import numpy as np
import argparse
import shutil
from pathlib import Path
from datasets import load_from_disk
from tqdm.auto import tqdm
from functools import partial
# def compute_avg_embeddings_batch(batch, bands=['g', 'r']):
#     """Compute average embeddings over time dimension. Assumes all embeddings are (time, dim)."""
#     batch_size = len(batch[list(batch.keys())[0]])
#     avg_embeddings = []
    
#     for i in range(batch_size):
#         example_avg = {}
        
#         for band in bands:
#             embedding_key = f'embeddings_{band}'
#             if embedding_key in batch and batch[embedding_key][i] is not None:
#                 # Simple: embeddings are always (time, dim), just average over time
#                 emb_array = np.array(batch[embedding_key][i], dtype=np.float32)  # (time, dim)
#                 avg_emb = emb_array.mean(axis=0, dtype=np.float32)              # (dim,)
#                 example_avg[band] = avg_emb.tolist()
#             else:
#                 example_avg[band] = None
        
#         avg_embeddings.append(example_avg)
    
#     return {"avg_embedding": avg_embeddings}

def avg_batch(batch, bands=('g', 'r')):
    B = len(next(iter(batch.values())))
    band_avgs = {}

    for b in bands:
        k = f'embeddings_{b}'
        if k in batch and batch[k] is not None:
            arr = np.asarray(batch[k], dtype=np.float32)   # (B,T,D) or (B,1,T,D)
            if arr.ndim == 4 and arr.shape[1] == 1:
                arr = arr[:, 0]                            # -> (B,T,D)
            assert arr.ndim == 3, f"Unexpected ndim for {k}: {arr.shape}"
            band_avgs[b] = arr.mean(axis=1, dtype=np.float32)  # (B,D)

    # ---- key change: tolist() ONCE per band, not per example ----
    band_lists = {b: band_avgs[b].tolist() for b in band_avgs}  # {b: [(D,), ...] length B}

    out = []
    for i in range(B):
        ex = {}
        for b in bands:
            ex[b] = band_lists[b][i] if b in band_lists else None
        out.append(ex)

    return {"avg_embedding": out}

# --- Enhanced avg_batch_columns with combined embedding functionality ---
def avg_batch_columns(*cols, column_names=None, band_combination="avg", add_combined=True):
    """
    Process embedding columns and optionally create combined embeddings.
    
    Args:
        cols: positional columns in the same order as input_columns
        column_names: pass the same list you give to input_columns  
        band_combination: How to combine bands - "concat", "avg", or specific band name ("g", "r", etc.)
        add_combined: Whether to add a "combined_embedding" column for direct use in benchmarks
        
    Returns: 
        Dictionary with individual band embeddings and optionally combined_embedding
        { "avg_embedding_g": (B,D), "avg_embedding_r": (B,D), "combined_embedding": (B,D_combined) }
    """
    out = {}
    band_arrays = {}
    
    # Step 1: Process each band individually (same as before)
    for name, values in zip(column_names, cols):
        # name is like "embeddings_g" or "embeddings_r"
        band = name.split("_", 1)[1]   # "g" or "r" (assumes "embeddings_<band>")
        arr = np.asarray(values, dtype=np.float32)     # (B,T,D) or (B,1,T,D)
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0]                             # -> (B,T,D)
        # assert arr.ndim == 3, f"Unexpected shape for {name}: {arr.shape}"
        
        if arr.ndim == 3:
            # Time-average each band: (B,T,D) -> (B,D)
            avg_emb = arr.mean(axis=1, dtype=np.float32)
            out[f"avg_embedding_{band}"] = avg_emb
            band_arrays[band] = avg_emb
        elif arr.ndim == 2:
            # print("already 1d for each data")
            # Already (B,D), just store directly, this happens for handcrafted feature and random embedding
            out[f"avg_embedding_{band}"] = arr
            band_arrays[band] = arr
    
    # Step 2: Create combined embedding if requested
    if add_combined and len(band_arrays) > 0:
        if band_combination == "concat":
            # Concatenate all bands in sorted order: (B,D) + (B,D) -> (B,2*D)
            sorted_bands = sorted(band_arrays.keys())
            combined_emb = np.concatenate([band_arrays[band] for band in sorted_bands], axis=1)
            
        elif band_combination == "avg":
            # Element-wise average of all bands: mean((B,D), (B,D)) -> (B,D)
            sorted_bands = sorted(band_arrays.keys())
            combined_emb = np.mean([band_arrays[band] for band in sorted_bands], axis=0)
            
        elif band_combination in band_arrays:
            # Use specific band: (B,D)
            combined_emb = band_arrays[band_combination]
            
        else:
            # Fallback to first available band
            sorted_bands = sorted(band_arrays.keys())
            combined_emb = band_arrays[sorted_bands[0]]
            print(f"Warning: band_combination '{band_combination}' not found, using '{sorted_bands[0]}'")
        
        # Keep as numpy array - HuggingFace datasets handles this efficiently
        out["combined_embedding"] = combined_emb
    
    return out

def process_split(dataset_path, split_name, bands, batch_size, band_combination="concat", add_combined=True):
    """Process a single split of the dataset."""
    split_path = Path(dataset_path) / split_name
    
    if not split_path.exists():
        print(f"Split {split_name} not found")
        return False
    
    print(f"Processing {split_name}...")
    dataset = load_from_disk(str(split_path))
    
    # Check what embedding columns exist
    if len(dataset) > 0:
        first_example = dataset[0]
        available_bands = [band for band in bands if f'embeddings_{band}' in first_example]
        
        if not available_bands:
            print(f"  No embedding columns found for bands {bands} in {split_name}")
            return False
        
        print(f"  Found embedding columns for bands: {available_bands}")
        
        # Show example shape
        for band in available_bands:
            if first_example[f'embeddings_{band}'] is not None:
                shape = np.array(first_example[f'embeddings_{band}']).shape
                print(f"    embeddings_{band}: {shape}")
    
    # Create backup
    backup_path = split_path.with_suffix('.backup_avg')
    if not backup_path.exists():
        print(f"  Creating backup...")
        shutil.copytree(str(split_path), str(backup_path))
    
    # Compute average embeddings using lambda (like embed.py style)
    # processed = dataset.map(
    #     lambda batch: {
    #         "avg_embedding": [
    #             {
    #                 band: np.array(batch[f'embeddings_{band}'][i]).mean(axis=0).tolist()
    #                 if f'embeddings_{band}' in batch and batch[f'embeddings_{band}'][i] is not None
    #                 else None
    #                 for band in available_bands
    #             }
    #             for i in range(len(batch[list(batch.keys())[0]]))
    #         ]
    #     },
    #     batched=True,
    #     batch_size=batch_size,
    #     desc=f"Computing avg embeddings for {split_name}"
    # )
    # processed = dataset.map(
    #     partial(avg_batch, bands=available_bands),
    #     batched=True,
    #     batch_size=batch_size,  # tune 1000–3000 for T=200,D=256,2 bands
    #     desc=f"Computing avg embeddings for {split_name}",
    #     num_proc=4,
    # )

    embed_cols = [f"embeddings_{b}" for b in available_bands]
    processed = dataset.map(
        avg_batch_columns,
        batched=True,
        batch_size=batch_size,
        num_proc=4,                              # 2–4 is a good sweet spot
        input_columns=embed_cols,                # pass only needed columns
        fn_kwargs={
            "column_names": embed_cols,
            "band_combination": band_combination,
            "add_combined": add_combined
        },
        desc=f"Computing avg embeddings for {split_name}",
    )

    
    # Save to temporary location first
    temp_path = split_path.with_suffix('.temp')
    processed.save_to_disk(str(temp_path))
    
    # Replace original
    if split_path.exists():
        shutil.rmtree(str(split_path))
    temp_path.rename(split_path)
    
    print(f"  ✓ Computed and saved average embeddings for {split_name}")
    
    # Show example of result
    if len(processed) > 0:
        row0 = processed[0]
        
        # Show individual band embeddings
        for b in available_bands:
            key = f"avg_embedding_{b}"
            if key in row0 and row0[key] is not None:
                print(f"    {key}: shape {np.array(row0[key]).shape}")
        
        # Show combined embedding if created
        if add_combined and "combined_embedding" in row0 and row0["combined_embedding"] is not None:
            combined_shape = np.array(row0["combined_embedding"]).shape
            print(f"    combined_embedding ({band_combination}): shape {combined_shape}")
            print(f"    → Ready for direct use in linear_classifier.py and other benchmarks!")

    
    return True

def main():
    parser = argparse.ArgumentParser(description="Compute average embeddings over time dimension")
    parser.add_argument("--dataset", required=True, help="Dataset name to process")
    parser.add_argument("--base_path", 
                       default="/projects/b1094/StarEmbed/embeddings/descriptive_name_embeddings",
                       help="Base path to embeddings directory")
    parser.add_argument("--bands", nargs="+", default=["g", "r"],
                       help="Bands to process (default: g r)")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test", "anom"],
                       help="Splits to process")
    parser.add_argument("--batch_size", type=int, default=1000,
                       help="Batch size for processing")
    parser.add_argument("--band_combination", choices=["concat", "avg", "g", "r", "i", "z"], 
                       default="concat",
                       help="How to combine bands for combined_embedding (default: avg)")
    parser.add_argument("--no_combined", action="store_true",
                       help="Skip creating combined_embedding column")
    
    args = parser.parse_args()
    
    dataset_path = Path(args.base_path) / args.dataset
    
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 1
    
    add_combined = not args.no_combined
    
    print(f"Processing dataset: {args.dataset}")
    print(f"Bands: {args.bands}")
    print(f"Splits: {args.splits}")
    print(f"Band combination: {args.band_combination}")
    print(f"Add combined_embedding: {add_combined}")
    
    success_count = 0
    for split in args.splits:
        if process_split(dataset_path, split, args.bands, args.batch_size, 
                        args.band_combination, add_combined):
            success_count += 1
    
    print(f"\n✓ Successfully processed {success_count}/{len(args.splits)} splits")
    print("Columns added:")
    print("  avg_embedding_g, avg_embedding_r, ... : Individual band embeddings")
    if add_combined:
        print(f"  combined_embedding ({args.band_combination}) : Ready-to-use combined embedding!")
        print("  → linear_classifier.py can now skip compute_embedding() calls for massive speedup!")
    
    return 0 if success_count == len(args.splits) else 1

if __name__ == "__main__":
    exit(main())
