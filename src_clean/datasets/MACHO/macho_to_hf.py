#!/usr/bin/env python3
"""
Convert MACHO light curves from Parquet to HuggingFace dataset format.

Transforms:
  Input:  data/raw_data/macho/macho/light_curves/shard_*.parquet
  Output: data/hf_macho_light_curves/

Process: Groups observations by star ID, structures as time series with r-band 
photometry data, and converts to HF dataset format using parallel processing.

Usage:
  python macho_to_hf.py
  
  Script automatically resumes from where it left off by skipping existing output shards.
"""
import os
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
from tqdm import tqdm

from datasets import Dataset, Features, Sequence, Value

def mjd_to_datetime(mjd_array: np.ndarray) -> pd.DatetimeIndex:
    """Convert MJD array to pandas Timestamps."""
    epoch = pd.Timestamp("1858-11-17")
    return epoch + pd.to_timedelta(mjd_array, unit="D")

def process_and_dump_shard(shard_path: Path, out_dir: Path, features: Features, shard_idx: int) -> int:
    """
    Read one Parquet shard, group by newID, build HF records, convert to a
    tiny Dataset, and save it as shard 'shard_idx'.
    Returns the number of stars processed.
    """
    df = pd.read_parquet(shard_path)
    records = []
    for star_id, grp in df.groupby("newID"):
        grp = grp.sort_values("mjd")
        mjd = grp["mjd"].values.astype(np.float32)
        mag = grp["mag"].values.astype(np.float32)
        err = grp["errmag"].values.astype(np.float32)

        timestamps = mjd_to_datetime(mjd)
        delta_days = np.concatenate(
            [[0.0], np.diff(timestamps.values.astype("datetime64[D]").astype(int))],
            axis=0
        ).astype(np.float32)

        start = pd.Timestamp(timestamps[0])
        length = len(mjd)

        r_band = {
            "item_id": str(star_id),
            "start": start,
            "freq": "1D",
            "target": mag,
            "past_feat_dynamic_real": err,
            "feat_dynamic_real": delta_days,
            "period": np.float32(0.0),
            "ps1_objid": str(star_id),
            "mjd": mjd,
            "class": "",
            "csdr1_id": "",
            "length": np.int32(length),
            "sourceid": str(star_id),
        }

        records.append({
            "sourceid": np.int64(star_id),
            "bands_data": {"g": None, "r": r_band, "i": None}
        })

    ds = Dataset.from_list(records, features=features)
    shard_dir = out_dir / f"shard_{shard_idx:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(shard_dir), num_shards=1, num_proc=1)
    return len(records)

# Configuration
data_dir = Path("/projects/p32626/uni2ts/data/main-code/data/raw_data/macho/macho/light_curves")
out_dir  = Path("/projects/p32626/uni2ts/data/main-code/data/hf_macho_light_curves")
out_dir.mkdir(parents=True, exist_ok=True)

band_schema = {
    "item_id": Value("string"), "start": Value("timestamp[ns]"), "freq": Value("string"),
    "target": Sequence(Value("float32")), "past_feat_dynamic_real": Sequence(Value("float32")),
    "feat_dynamic_real": Sequence(Value("float32")), "period": Value("float32"),
    "ps1_objid": Value("string"), "mjd": Sequence(Value("float32")),
    "class": Value("string"), "csdr1_id": Value("string"),
    "length": Value("int32"), "sourceid": Value("string"),
}
features = Features({
    "sourceid": Value("int64"),
    "bands_data": {"g": band_schema, "r": band_schema, "i": band_schema}
})

shard_paths = sorted(data_dir.glob("shard_*.parquet"))
n_shards  = len(shard_paths)
n_workers = min(8, max(1, os.cpu_count() - 2))

# 1) Figure out which shards still need processing
# skip any shard where the subdirectory already exists
pending = []
for idx, path in enumerate(shard_paths):
    shard_dir = out_dir / f"shard_{idx:03d}"
    if not shard_dir.exists():
        pending.append((idx, path))
    else:
        print(f"Skipping shard {idx:03d}, output folder already present")


print(f"Will process {len(pending)}/{n_shards} shards")
bar = tqdm(total=len(pending), desc="Shards written", unit="shard")

# 2) Only submit the pending ones
with ProcessPoolExecutor(max_workers=n_workers) as exe:
    future_to_idx = {
        exe.submit(process_and_dump_shard, path, out_dir, features, idx): idx
        for idx, path in pending
    }
    for future in as_completed(future_to_idx):
        idx = future_to_idx[future]
        try:
            count = future.result()
            print(f"Shard {idx:03d} written with {count} records")
        except Exception as e:
            print(f"Error in shard {idx:03d}: {e}")
        bar.update(1)

bar.close()

# Flatten
print("Flattening shards")
for idx in range(n_shards):
    sub = out_dir / f"shard_{idx:03d}"
    for arrow in sub.glob("*.arrow"):
        dest = out_dir / f"data-{idx:05d}-of-{n_shards:05d}.arrow"
        arrow.rename(dest)
    if idx == 0:
        for name in ("dataset_info.json", "state.json"):
            src = sub / name
            dst = out_dir / name
            if src.exists():
                src.rename(dst)
    shutil.rmtree(sub)

print(f"All done. Files in {out_dir}")
