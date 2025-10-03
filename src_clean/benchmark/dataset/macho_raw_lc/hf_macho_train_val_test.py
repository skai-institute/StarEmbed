#!/usr/bin/env python3
"""
Process HuggingFace MACHO dataset: flatten metadata and create train/val/test splits.

Transforms:
  Input:  data/hf_macho_light_curves/
  Output: hf_macho_70-10-20/ (70% train, 10% val, 20% test)

Process: Flattens bands_data structure, extracts time series fields, 
and creates random train/validation/test splits.

Usage:
  python hf_macho_train_val_test.py
"""

from datasets import load_from_disk, DatasetDict

# which time-series fields to keep (same as before)
TS_KEYS = [
    "target",
    "past_feat_dynamic_real",
    "feat_dynamic_real",
    "mjd",
    "length",
]

def flatten_metadata(example):
    # pick the first non‐None band as reference
    ref = next((d for d in example["bands_data"].values() if d and d.get("item_id")), None)

    out = {
        "sourceid": example["sourceid"],
        "item_id":  ref["item_id"]   if ref else None,
        "start":    ref["start"]     if ref else None,
        "freq":     ref["freq"]      if ref else None,
        "period":   ref["period"]    if ref else None,
        "objid":    ref["ps1_objid"] if ref else None,
        # drop class & csdr1_id entirely
        "bands_data": []
    }

    for band, d in example["bands_data"].items():
        if not d or not d.get("item_id"):
            continue
        ts = {
            k: list(d[k]) if hasattr(d[k], "__iter__") and not isinstance(d[k], str) else d[k]
            for k in TS_KEYS
        }
        ts["band"] = band
        out["bands_data"].append(ts)

    return out

def main():
    print("Loading dataset...")
    # 1) load & flatten
    ds = load_from_disk("/projects/p32626/uni2ts/data/main-code/data/hf_macho_light_curves")
    
    print("Flattening metadata...")
    ds = ds.map(flatten_metadata, num_proc=4, remove_columns=["bands_data"])  # you can remove old columns

    print("Creating train/test split...")
    # 2) random train/test split 80/20
    split1 = ds.train_test_split(test_size=0.2, seed=42)
    trainval = split1["train"]
    test     = split1["test"]

    print("Creating train/validation split...")
    # 3) split train → train/val 0.125 of 80% = 10% total
    split2 = trainval.train_test_split(test_size=0.125, seed=42)
    train = split2["train"]
    val   = split2["test"]

    new_ds = DatasetDict({
        "train":      train,
        "validation": val,
        "test":       test
    })

    print(f"Dataset splits: train={len(train)}, val={len(val)}, test={len(test)}")
    
    # 4) save
    print("Saving processed dataset...")
    new_ds.save_to_disk("hf_macho_70-10-20")
    print("Done! Dataset saved to hf_macho_70-10-20/")

if __name__ == "__main__":
    main() 