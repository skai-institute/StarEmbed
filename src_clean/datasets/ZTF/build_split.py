

from datasets import load_from_disk, DatasetDict

# which time-series fields to keep
TS_KEYS = [
    "target",
    "past_feat_dynamic_real",
    "feat_dynamic_real",
    "mjd",
    "length",
]

# pull out star-level properties from /p32795/weijian/hf_csdr1_raw4_catflags_filtered_with_labels_multiband as suggested in 
# https://skai-institute.slack.com/archives/C080W11MD1T/p1746753352405109?thread_ts=1746734877.730749&cid=C080W11MD1T
def flatten_metadata(example):
    # 1) pick a reference band for the shared metadata
    ref = None
    for band, d in example["bands_data"].items():
        if d is not None and d.get("item_id") is not None:
            ref = d
            break

    # 2) build top-level fields
    out = {
        "sourceid": example["sourceid"],
        # if ref is None, these will be None; otherwise pulled from ref
        "item_id":  ref["item_id"]   if ref else None,
        "start":    ref["start"]     if ref else None,
        "freq":     ref["freq"]      if ref else None,
        "period":   ref["period"]    if ref else None,
        "objid":    ref["ps1_objid"] if ref else None,
        "class":    ref["class"]     if ref else None,
        "csdr1_id": ref["csdr1_id"]  if ref else None,
        # we’ll rebuild bands_data as a list
        "bands_data": []
    }

    # 3) for each band that exists, append a dict to the list
    for band, d in example["bands_data"].items():
        if d is None or d.get("item_id") is None:
            continue
        # collect only the TS fields, converting to plain Python lists
        ts = {
            k: list(d[k]) if hasattr(d[k], "__iter__") and not isinstance(d[k], str) else d[k]
            for k in TS_KEYS
        }
        ts["band"] = band
        out["bands_data"].append(ts)

    return out

# 1) load & flatten as before
dataset = load_from_disk("/projects/p32795/weijian/hf_csdr1_raw4_catflags_filtered_with_labels_multiband")
dataset = dataset.map(flatten_metadata, num_proc=4) # pull out star-level properties

# 2) turn 'class' into a ClassLabel feature
dataset = dataset.class_encode_column("class")

# now `dataset.features["class"]` will be of type ClassLabel

# 3) you can safely stratify by it
split1 = dataset.train_test_split(
    test_size=0.2,
    stratify_by_column="class",
    seed=42
)
trainval = split1["train"]
test     = split1["test"]

split2 = trainval.train_test_split(
    test_size=1/8,                # 0.125 of 80% → 10% total
    stratify_by_column="class",
    seed=42
)
train = split2["train"]
val   = split2["test"]

new_ds = DatasetDict({
    "train":      train,
    "validation": val,
    "test":       test
})

# 4) (optional) save back to disk
new_ds.save_to_disk("6_20_hf_csdr1_multiband_70-10-20")
