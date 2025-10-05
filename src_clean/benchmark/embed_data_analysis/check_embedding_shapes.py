#!/usr/bin/env python3
import os
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from datasets import Dataset
import glob

# Define the datasets to check
datasets = [
    "hf_csdr1_multiband_raw4_embeddings_astromer_1_subclass_pad_correct",
    "hf_csdr1_multiband_raw4_embeddings_astromer_2_gr_sampling_True", 
    "csdr1_raw4_catflags_filtered_embs_chronos_t5_tiny_trn_val_tst_ctx200_bandgr",
    "csdr1_raw4_catflags_filtered_embs_chronos_bolt_tiny_trn_val_tst_ctx200_bandgr",
    "csdr1_raw_embs_moiral_small_trn_val_tst_ctx200_pdt64_psz16_bandgr"
]

base_path = "/projects/b1094/StarEmbed/embeddings/embeddings_with_anom"

for dataset_name in datasets:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print('='*60)
    
    dataset_path = os.path.join(base_path, dataset_name, "train")
    
    if not os.path.exists(dataset_path):
        print(f"Path not found: {dataset_path}")
        continue
    
    # Load dataset using HuggingFace datasets
    try:
        from datasets import Dataset
        ds = Dataset.load_from_disk(dataset_path)
        print(f"Dataset loaded with {len(ds)} samples")
        print(f"Columns: {ds.column_names}")
        
        for emb_col in ['embeddings_g', 'embeddings_r']:
            if emb_col in ds.column_names:
                sample = ds[0][emb_col]
                print(f"\n{emb_col} sample shape analysis:")
                print(f"  Type: {type(sample)}")
                
                def analyze_nested_structure(data, level=1):
                    if isinstance(data, list) and data:
                        print(f"  Level {level} length: {len(data)}")
                        if isinstance(data[0], list):
                            analyze_nested_structure(data[0], level + 1)
                        else:
                            print(f"  Final element type: {type(data[0])}")
                            # Calculate total shape
                            shape_parts = []
                            current = data
                            while isinstance(current, list) and current:
                                shape_parts.append(len(current))
                                current = current[0]
                            print(f"  Shape: {tuple(shape_parts)}")
                
                analyze_nested_structure(sample)
            else:
                print(f"\n{emb_col}: Not found in this dataset")
                
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Fallback to individual arrow files
        arrow_files = glob.glob(os.path.join(dataset_path, "data-*.arrow"))
        if arrow_files:
            try:
                ds = Dataset.from_file(arrow_files[0])
                print(f"Loaded single arrow file with {len(ds)} samples")
                
                for emb_col in ['embeddings_g', 'embeddings_r']:
                    if emb_col in ds.column_names:
                        sample = ds[0][emb_col]
                        print(f"\n{emb_col} sample shape analysis:")
                        print(f"  Type: {type(sample)}")
                        if isinstance(sample, list):
                            print(f"  Length: {len(sample)}")
                            if sample and isinstance(sample[0], list):
                                print(f"  Inner length: {len(sample[0])}")
                                if sample[0] and isinstance(sample[0][0], list):
                                    print(f"  Innermost length: {len(sample[0][0])}")
                                    print(f"  Shape: ({len(sample)}, {len(sample[0])}, {len(sample[0][0])})")
                                else:
                                    print(f"  Shape: ({len(sample)}, {len(sample[0])})")
                            else:
                                print(f"  Shape: ({len(sample)},)")
            except Exception as e2:
                print(f"Fallback approach also failed: {e2}")
