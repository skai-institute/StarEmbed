import os
import glob
from datasets import load_from_disk
import pandas as pd
from collections import Counter
import re
from collections import defaultdict

"""
After doing the cross-match between the scope and the ztf data, we save the matched light curves as a huggingface dataset. 

This file contains functions to do some inspection tasks on the saved hf dataset or on the scope csv files.
"""


def cal_scope_data_size(output_dir: str = "/projects/p32795/weijian/queried_scope_from_ztf/matched_data_full2"):
    """
    
    This function is to count the total number of stars from the matched light curves.
    """

    # 1) Point this at wherever you saved your shards:
    shard_pattern = os.path.join(output_dir, "shard_*")

    # 2) Find and sort all the shard directories
    shard_dirs = sorted(glob.glob(shard_pattern))

    total_points = 0
    for shard in shard_dirs:
        # load each shard (this uses memory‑mapping under the hood)
        ds = load_from_disk(shard)
        total_points += len(ds)

    print(f"Found {len(shard_dirs)} shards, total data points = {total_points}")



def inspect_csvs(csv_pattern: str = "/projects/b1094/rehemtulla/SkAI/SCoPe/*/*/field_*_vs.csv"):
    """
    This function is to give a quick overview of the scope data regarding the file size and the number of rows.

    For each CSV matching csv_pattern, print:
     - filename
     - size on disk (MB)
     - number of lines (rows)
    """
    paths = sorted(glob.glob(csv_pattern))
    print(f"Found {len(paths)} CSV files:\n")
    for p in paths:
        size_bytes = os.path.getsize(p)
        size_mb = size_bytes / (1024 * 1024)
        try:
            # Fast row count without loading full DF
            with open(p, 'rb') as f:
                row_count = sum(1 for _ in f) - 1  # subtract 1 for header
        except Exception:
            row_count = "?"

        print(f"{os.path.basename(p):30s}  {size_mb:8.2f} MB  rows: {row_count}")

def inspect_duplicate_field_ids(csv_pattern: str = "/projects/b1094/rehemtulla/SkAI/SCoPe/*/*/field_*_vs.csv"):
    """
    This function is to check if there are duplicate field IDs in the scope data.
    """
    paths = sorted(glob.glob(csv_pattern))
    ids = [re.search(r'field_(\d+)_vs\.csv', os.path.basename(p)).group(1)
        for p in paths]
    dups = {fid: cnt for fid, cnt in Counter(ids).items() if cnt > 1}
    if dups:
        # list the duplicate field IDs and paths and their file size
        for fid, cnt in dups.items():
            print(f"⚠️ Duplicate field ID {fid} found {cnt} times:")
            for p in paths:
                if re.search(r'field_(\d+)_vs\.csv', os.path.basename(p)).group(1) == fid:
                    print(f"  {p} {os.path.getsize(p) / (1024 * 1024):.2f} MB")
    else:
        print("All field IDs are unique.")

def verify_field_id_coverage(csv_pattern: str = "/projects/b1094/rehemtulla/SkAI/SCoPe/*/*/field_*_vs.csv",
                             shard_pattern: str = "/projects/p32795/weijian/queried_scope_from_ztf/matched_data_full2/shard_*"):
    """
    This function is to verify if all the field IDs in the scope data are present in the matched light curves (in case the processing is incomplete).
    """

    # 1) Gather input CSV paths keyed by field_id
    input_paths = defaultdict(list)
    for path in glob.glob(csv_pattern):
        m = re.search(r'field_(\d+)_vs\.csv$', os.path.basename(path))
        if not m:
            continue
        fid = m.group(1)
        input_paths[fid].append(path)

    # 2) Gather shard dirs keyed by field_id
    shard_paths = defaultdict(list)
    for path in glob.glob(shard_pattern):
        # basename should be like "shard_423" or "shard_023"
        m = re.search(r'shard_(\d+)$', os.path.basename(path))
        if not m:
            continue
        fid = m.group(1).lstrip("0")  # strip leading zeros, if any
        shard_paths[fid].append(path)

    input_ids = set(input_paths)
    shard_ids = set(shard_paths)

    # 3) Compute mismatches
    missing_shards = input_ids - shard_ids
    extra_shards   = shard_ids - input_ids

    # 4) Report
    print("👉 CSVs with NO matching shard:")
    for fid in sorted(missing_shards, key=int):
        for p in input_paths[fid]:
            print("  ", p)

    print("\n👉 Shards with NO matching CSV:")
    for fid in sorted(extra_shards, key=int):
        for p in shard_paths[fid]:
            print("  ", p)

if __name__ == "__main__":
    csv_pattern = "/projects/b1094/rehemtulla/SkAI/SCoPe/*/*/field_*_vs.csv"
    output_dir = "/projects/p32795/weijian/queried_scope_from_ztf/matched_data_full2"

    # call the function in need
    # ...
    cal_scope_data_size(output_dir)

    

    




