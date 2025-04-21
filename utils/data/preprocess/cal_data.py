# import os
# import glob
# from datasets import load_from_disk

# # 1) Point this at wherever you saved your shards:
# output_dir = "/projects/p32795/weijian/queried_scope_from_ztf/"
# shard_pattern = os.path.join(output_dir, "shard_*")

# # 2) Find and sort all the shard directories
# shard_dirs = sorted(glob.glob(shard_pattern))

# total_points = 0
# for shard in shard_dirs:
#     # load each shard (this uses memory‑mapping under the hood)
#     ds = load_from_disk(shard)
#     total_points += len(ds)

# print(f"Found {len(shard_dirs)} shards, total data points = {total_points}")


import glob
import os
import pandas as pd

def inspect_csvs(csv_pattern):
    """
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

if __name__ == "__main__":
    csv_pattern = "/projects/b1094/rehemtulla/SkAI/SCoPe/*/*/field_*_vs.csv"  # adjust to your actual pattern
    # inspect_csvs(csv_pattern)

    # fields = glob.glob(csv_pattern, recursive=True)
    # # count the number of fields
    # print(fields)
    # print(f"Found {len(fields)} fields")

    import glob, re
    from collections import Counter

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

    




