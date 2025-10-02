#!/usr/bin/env python3
"""
Embed a multi-band HuggingFace DatasetDict with ASTROMER (main-code repo).

Transforms:
  Input:  HuggingFace DatasetDict with multi-band light curves  
  Output: Dataset with ASTROMER embeddings per band

Process: Builds fixed-length (200, 3) windows for each band, zero-centers
per column (MJD, mag, mag_err), runs inference with TensorFlow ASTROMER model.

Key features:
- Uses main-code repo utilities for identical preprocessing
- Filters to keep only examples with both g and r bands
- Generates both full embeddings and averaged embeddings per band

Usage:

# this script use the code below astromer_2 main-code repo
# make sure you have the main-code repo cloned and the path is correct
cd /projects/p32626/uni2ts/data/main-code
  python astromer_2_embedding.py \
    --input_path /path/to/input/dataset \
    --output_path /path/to/output/dataset \
    --model_weights /projects/p32626/uni2ts/data/weights/macho-clean \
    --bands g r \
    --splits all \
    --duration 200 \
    --enc_batch 1024 \
    --preproc_procs 16 \
    --filter_gr
"""

import numpy as np
import tensorflow as tf
from datasets import load_from_disk, DatasetDict, Dataset
import argparse
from pathlib import Path

# Import ASTROMER utilities
from presentation.pipelines.steps.model_design import load_pt_model
from src.data.loaders import load_numpy, format_inp_astromer
from src.data import preprocessing as pp
from src.data.masking import mask_dataset

def keep_gr(batch):
    """Filter to keep only examples where both g and r bands exist."""
    return [
        (bd.get("g") is not None and bd.get("r") is not None)
        for bd in batch["bands_data"]
    ]

def make_windows(example, band, duration=200):
    """Build (duration, 3) window per band and attach to example."""
    bd = example["bands_data"].get(band)
    if bd is None:
        example[f"windows_{band}"] = None
        return example

    mjd = np.asarray(bd["mjd"], dtype=np.float32)
    mag = np.asarray(bd["target"], dtype=np.float32)
    err = np.asarray(bd["past_feat_dynamic_real"], dtype=np.float32)
    arr = np.stack([mjd, mag, err], axis=1)  # (L,3)

    # Take last duration obs (or left-pad with zeros)
    if arr.shape[0] >= duration:
        win = arr[-duration:]
    else:
        pad = np.zeros((duration - arr.shape[0], 3), dtype=np.float32)
        win = np.vstack([arr, pad])

    # Zero-mean each column
    win -= win.mean(axis=0, keepdims=True)
    example[f"windows_{band}"] = win
    return example

def assert_embedding_shape(emb, window_size=200, hidden_dim=256):
    """Validate embedding shape."""
    assert isinstance(emb, np.ndarray), f"Expected NumPy array, got {type(emb)}"
    assert emb.ndim == 3, f"Expected 3 dims, got {emb.ndim}"
    B, T, D = emb.shape
    assert T == window_size, f"Expected time-axis={window_size}, got {T}"
    assert D == hidden_dim, f"Expected hidden-dim={hidden_dim}, got {D}"

def encode_batch(batch, band, encoder, duration=200, enc_batch=1024):
    """
    HF map-style function to encode windows with ASTROMER.
    
    Args:
        batch: HF batch with windows
        band: band name
        encoder: ASTROMER encoder model
        duration: window size
        enc_batch: encoding batch size
    
    Returns:
        Dict with embeddings and averaged embeddings
    """
    windows = batch[f"windows_{band}"]  # list[(200,3)] or None
    ids = batch["item_id"]

    # Gather valid curves
    valid_idx, wins_valid = [], []
    for i, w in enumerate(windows):
        if w is None:
            continue
        wins_valid.append(np.asarray(w, dtype=np.float32))
        valid_idx.append(i)

    B = len(windows)  # batch size in HF sense
    D = encoder.output_shape[-1]  # hidden dim

    # Pre-allocate output tensors (zero for missing curves)
    emb_full = np.zeros((B, duration, D), dtype=np.float32)
    avg_full = np.zeros((B, D), dtype=np.float32)

    if not valid_idx:  # nothing to encode in this mini-batch
        return {f"embeddings_{band}": emb_full.tolist(),
                f"avg_{band}": avg_full.tolist()}

    # Pipeline that exactly mirrors get_loader() (no TFRecord I/O)
    tf_dataset = load_numpy(wins_valid)  # (None,3)
    tf_dataset = pp.to_windows(tf_dataset, window_size=duration,
                              sampling=True)  # no overlap
    tf_dataset = tf_dataset.map(pp.standardize)  # already 0-mean, ok
    tf_dataset, shapes = mask_dataset(tf_dataset,  # no masking
                                     msk_frac=0.0,
                                     rnd_frac=0.0,
                                     same_frac=0.0,
                                     window_size=duration)
    tf_dataset = tf_dataset.padded_batch(
        enc_batch, padded_shapes=shapes, drop_remainder=False)
    tf_dataset = tf_dataset.map(
        lambda x: format_inp_astromer(x, aversion="base"),
        num_parallel_calls=tf.data.AUTOTUNE)

    # Run encoder
    outputs = []
    for batch_x, _ in tf_dataset:
        z = encoder(batch_x, training=False, z_by_layer=False)  # (b,200,D)
        z = z.numpy()
        outputs.append(z)
    windows_emb = np.concatenate(outputs, axis=0)  # (V,200,D)
    avg_emb = windows_emb.mean(axis=1)  # (V,D)

    # Scatter back to the HF-batch positions
    for slot, idx in enumerate(valid_idx):
        emb_full[idx] = windows_emb[slot]
        avg_full[idx] = avg_emb[slot]
        
    assert_embedding_shape(emb_full, window_size=duration, hidden_dim=D)

    return {f"embeddings_{band}": emb_full.tolist(),
            f"avg_{band}": avg_full.tolist()}

def main():
    parser = argparse.ArgumentParser(description="Create ASTROMER embeddings for multi-band dataset")
    parser.add_argument("--input_path", type=str, required=True,
                       help="Path to input HuggingFace dataset")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save output dataset with embeddings")
    parser.add_argument("--model_weights", type=str, 
                       default="/projects/p32626/uni2ts/data/weights/macho-clean",
                       help="Path to ASTROMER model weights")
    parser.add_argument("--bands", type=str, nargs="+", default=["g", "r"],
                       help="Bands to process")
    parser.add_argument("--duration", type=int, default=200,
                       help="Window size (number of time points)")
    parser.add_argument("--enc_batch", type=int, default=1024,
                       help="GPU batch size for encoding")
    parser.add_argument("--preproc_procs", type=int, default=16,
                       help="CPU processes for preprocessing")
    parser.add_argument("--filter_gr", action="store_true",
                       help="Filter to keep only examples with both g and r bands")
    parser.add_argument("--splits", type=str, nargs="+", default=["all"],
                       help="Dataset splits to process (use 'all' for all available splits)")
    
    args = parser.parse_args()

    print(f"Loading ASTROMER model from: {args.model_weights}")
    # Load pre-trained model
    model, _ = load_pt_model(args.model_weights)
    encoder = model.get_layer("encoder")

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

    # Filter to keep only examples where both g and r exist (if requested)
    if args.filter_gr:
        print("Filtering to keep only examples with both g and r bands...")
        for split_name in splits_to_process:
            original_size = len(splits[split_name])
            splits[split_name] = splits[split_name].filter(
                keep_gr, batched=True, batch_size=2000)
            new_size = len(splits[split_name])
            print(f"  {split_name}: {original_size} -> {new_size} examples")

    print(f"Building {args.duration}-point windows for bands: {args.bands}")
    # Build windows for each band
    for band in args.bands:
        for split_name in splits_to_process:
            splits[split_name] = splits[split_name].map(
                lambda ex, b=band: make_windows(ex, b, args.duration),
                num_proc=args.preproc_procs,
                desc=f"Building {args.duration}-pt windows for band {band} [{split_name}]"
            )

    print("Encoding windows with ASTROMER...")
    # Run GPU encoding (single process to own the GPU)
    for band in args.bands:
        print(f"  Processing band: {band}")
        for split_name in splits_to_process:
            splits[split_name] = splits[split_name].map(
                lambda batch, b=band: encode_batch(batch, b, encoder, 
                                                 args.duration, args.enc_batch),
                batched=True,
                batch_size=args.enc_batch,
                num_proc=1,  # keep GPU safe
                remove_columns=[f"windows_{band}"],
                desc=f"Encoding band {band} [{split_name}]"
            )

    print(f"Saving dataset to: {args.output_path}")
    # Save only processed splits
    out_dict = {}
    for split_name in splits_to_process:
        out_dict[split_name] = splits[split_name]
    
    out = DatasetDict(out_dict)
    out.save_to_disk(args.output_path)
    print("Done! Split sizes:", {k: len(v) for k, v in out.items()})

if __name__ == "__main__":
    main() 