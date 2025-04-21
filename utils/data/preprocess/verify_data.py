import glob
import os
import re
from collections import defaultdict

# Adjust these to your actual locations
csv_pattern = "/projects/b1094/rehemtulla/SkAI/SCoPe/*/*/field_*_vs.csv"
shard_pattern  = "/projects/p32795/weijian/queried_scope_from_ztf/matched_data_full2/shard_*"

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
