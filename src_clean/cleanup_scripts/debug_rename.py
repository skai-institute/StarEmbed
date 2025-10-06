#!/usr/bin/env python3
"""Debug script to test column renaming"""

from datasets import load_from_disk

# Test the renaming function
def rename_columns_batch(batch, rename_mapping):
    """Rename columns in a batch."""
    print(f"Input batch keys: {list(batch.keys())}")
    result = {}
    
    # Process each column in the batch
    for key, values in batch.items():
        # Check if this column needs to be renamed
        if key in rename_mapping:
            # Use the new name
            new_name = rename_mapping[key]
            result[new_name] = values
            print(f"  Renamed {key} -> {new_name}")
            # Don't include the old name
        else:
            # Keep the column as-is
            result[key] = values
            print(f"  Kept {key}")
    
    print(f"Output batch keys: {list(result.keys())}")
    return result

# Test with a small sample
ds = load_from_disk('/projects/b1094/StarEmbed/embeddings/descriptive_name_embeddings/csdr1_raw4_catflags_filtered_embs_hand_crafted_trn_val_tst_bandgr/train')
small_ds = ds.select(range(2))

# Test our function
rename_mapping = {
    "g_embedding": "embeddings_g",
    "r_embedding": "embeddings_r"
}

print("Before renaming:")
print(f"Columns: {list(small_ds[0].keys())}")

result = small_ds.map(
    lambda batch: rename_columns_batch(batch, rename_mapping),
    batched=True,
    batch_size=2
)

print("\nAfter renaming:")
print(f"Columns: {list(result[0].keys())}")
