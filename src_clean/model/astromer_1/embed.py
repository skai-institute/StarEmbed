#!/usr/bin/env python3
"""
Create ASTROMER embeddings for light curves using SingleBandEncoder.

Transforms:
  Input:  HuggingFace DatasetDict with light curves
  Output: Dataset with ASTROMER embeddings per band

Process: Builds time series windows, encodes using ASTROMER SingleBandEncoder
for specified bands and splits, and saves dataset with embeddings.

Usage:
  # Run from directory containing weights folder
  cd /projects/p32626/uni2ts/data/
  
  python embed.py \
    --input_path /path/to/input/dataset \
    --output_path /path/to/output/dataset \
    --model_name macho \
    --bands g r \
    --splits test \
    --duration 200 \
    --enc_batch 512 \
    --preproc_procs 4 \
    --working_dir src/model/astromer_1
"""

import numpy as np
import argparse
import os
from tqdm.auto import tqdm
from datasets import load_from_disk, DatasetDict, Dataset
from ASTROMER.models import SingleBandEncoder
from datasets.utils.logging import enable_progress_bar

def make_windows(example, band, duration=200):
    """Build time series windows from band data."""
    bd = example["bands_data"].get(band)
    if bd is None:
        example[f"windows_{band}"] = None
    else:
        mjd = np.array(bd["mjd"],                    dtype=np.float32)
        mag = np.array(bd["target"],                 dtype=np.float32)
        err = np.array(bd["past_feat_dynamic_real"], dtype=np.float32)
        arr = np.stack([mjd, mag, err], axis=1)      # (L,3)
        L   = arr.shape[0]
        if L >= duration:
            win = arr[-duration:]
        else:
            pad = np.zeros((duration - L, 3), dtype=np.float32)
            win = np.vstack([arr, pad])
        win = win - win.mean(axis=0, keepdims=True)
        example[f"windows_{band}"] = win
    return example

def encode_batch(batch, band, model, duration=200):
    """Encode batch of windows using ASTROMER model."""
    windows = batch[f"windows_{band}"]  # List[Optional[ndarray]], len=B, each (duration, 3) or None
    ids     = batch["sourceid"]         # List[int], len=B

    valid_ids, wins_valid = [], []
    for i, w in enumerate(windows):
        if w is None: 
            continue
        wins_valid.append(np.array(w, dtype=np.float32))  # (duration, 3)
        valid_ids.append(i)                              # int

    # wins_valid: List[ndarray], len=V, each (duration, 3) where V = num valid windows
    # valid_ids: List[int], len=V

    if not valid_ids:
        # No valid windows for this band - return None for entire batch
        emb = [None] * len(windows)  # List[None], len=B
    else:
        out = model.encode(
            wins_valid,                  # Input: List[ndarray], len=V, each (duration, 3)
            oids_list=[ids[i] for i in valid_ids],  # List[int], len=V
            batch_size=len(wins_valid),  # int: V
            concatenate=False
        )[0].cpu().numpy()               # Output: (V, duration, D) where D=hidden_size
        
        # Create list with None for invalid objects, embeddings for valid ones
        emb = []                         # Will be List[Optional[List]], len=B
        valid_idx = 0
        for i, w in enumerate(windows):  # i: 0 to B-1, w: (duration,3) or None
            if w is None:
                emb.append(None)         # Individual object has no data for this band
            else:
                emb.append(out[valid_idx].tolist())  # (duration, D) -> List[List[float]]
                valid_idx += 1

    # emb: List[Optional[List]], len=B, each None or List[List[float]] of shape (duration, D)
    return { f"embeddings_{band}": emb }

def main():
    parser = argparse.ArgumentParser(description="Create ASTROMER embeddings using SingleBandEncoder")
    parser.add_argument("--input_path", type=str, required=True,
                       help="Path to input HuggingFace dataset")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save output dataset with embeddings")
    parser.add_argument("--model_name", type=str, default="macho",
                       help="ASTROMER model name for from_pretraining")
    parser.add_argument("--bands", type=str, nargs="+", default=["r"],
                       help="Bands to process")
    parser.add_argument("--splits", type=str, nargs="+", default=["validation"],
                       help="Dataset splits to process (use 'all' for all available splits)")
    parser.add_argument("--duration", type=int, default=200,
                       help="Window size (number of time points)")
    parser.add_argument("--enc_batch", type=int, default=5120,
                       help="Batch size for encoding")
    parser.add_argument("--preproc_procs", type=int, default=16,
                       help="CPU processes for preprocessing")
    parser.add_argument("--working_dir", type=str, 
                       default="/projects/p32626/uni2ts/data/",
                       help="Working directory containing weights folder")
    
    args = parser.parse_args()

    # Change to working directory where weights folder exists
    if args.working_dir:
        print(f"Changing to working directory: {args.working_dir}")
        os.chdir(args.working_dir)

    # enable HF-datasets tqdm
    enable_progress_bar()

    print(f"Initializing ASTROMER model: {args.model_name}")
    model = SingleBandEncoder().from_pretraining(args.model_name)

    print(f"Loading dataset from: {args.input_path}")
    data = load_from_disk(args.input_path)

    # Handle Dataset vs DatasetDict
    if isinstance(data, DatasetDict):
        splits = data
        available_splits = list(splits.keys())
    else:
        # Single Dataset - wrap in DatasetDict with "train" as default name
        splits = DatasetDict({"train": data})
        available_splits = ["train"]

    # Handle "all" splits
    if "all" in args.splits:
        splits_to_process = available_splits
    else:
        splits_to_process = args.splits
        # Validate splits exist
        for split_name in splits_to_process:
            if split_name not in available_splits:
                raise ValueError(f"Split '{split_name}' not found. Available: {available_splits}")

    print(f"Available splits: {available_splits}")
    print(f"Processing splits: {splits_to_process}")
    print(f"Processing bands: {args.bands}")

    # 3) PREPROCESS: BUILD WINDOWS FOR EACH BAND
    print("Building time series windows...")
    for band in tqdm(args.bands, desc="Building windows for bands"):
        for split_name in splits_to_process:
            splits[split_name] = splits[split_name].map(
                lambda ex, b=band: make_windows(ex, b, args.duration),
                num_proc=args.preproc_procs,
                desc=f"make_windows[{band}][{split_name}]"
            )

    # 4) ENCODE WINDOWS
    print("Encoding windows with ASTROMER...")
    for band in tqdm(args.bands, desc="Encoding bands"):
        for split_name in splits_to_process:
            splits[split_name] = splits[split_name].map(
                lambda batch, b=band: encode_batch(batch, b, model, args.duration),
                batched=True,
                batch_size=args.enc_batch,
                num_proc=1,
                remove_columns=[f"windows_{band}"],
                desc=f"encode[{band}][{split_name}]"
            )

    # 5) SAVE FINAL DATASET
    print("Saving dataset with embeddings...")
    out_dict = {}
    for split_name in splits_to_process:
        out_dict[split_name] = splits[split_name]
    
    out = DatasetDict(out_dict)
    out.save_to_disk(args.output_path)

    print("Done! Split sizes:", {s: len(out[s]) for s in out})
    print(f"Dataset with embeddings saved to: {args.output_path}")

if __name__ == "__main__":
    main() 