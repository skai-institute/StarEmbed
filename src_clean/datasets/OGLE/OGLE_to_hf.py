from datasets import Dataset, Features, Value, Sequence
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import argparse
import time
import json

from OGLE_reading_utils import (
    load_catalog, merge_remarks, merge_ident, read_light_curve, get_period_feature_columns
)

# Standardized StarEmbed schema with some columns unique to OGLE
band_schema = Features({
    "mjd": Sequence(feature=Value("float64")),
    "target": Sequence(feature=Value("float64")),  # mag
    "past_feat_dynamic_real": Sequence(feature=Value("float64")),  # mag unc
    "feat_dynamic_real": Sequence(feature=Value("float64")),  # delta t between observations
    "length": Value("int64"),
})

schema = Features({
    # From catalog file
    "sourceid": Value("string"),
    "avg_mag_I": Value("float64"),
    "avg_mag_V": Value("float64"),

    "parent_type": Value("string"),
    "sub_type": Value("string"),
    "class_str": Value("string"),
    "region": Value("string"),

    # From light curve files
    "bands_data": {
        "I": band_schema,
        "V": band_schema,
    },

    # From remarks file
    "remarks": Value("string"),

    # From ident file
    "OGLE_IV_id": Value("string"),
    "OGLE_III_id": Value("string"),
    "OGLE_II_id": Value("string"),
    "other_id": Value("string"),
    "ra": Value("string"),
    "dec": Value("string"),
} | {feature: Value("float64") for feature in get_period_feature_columns(3)})


def create_dataset(num_workers):
    catalogs_to_process = [
        # region, parent_type, sub_type
        ("blg", "hb", "hb"),
        # ("BLG", "TRANSITS", "TRANSITS")
    ]

    # with open("all_OGLE_collections.json", "r") as f:
    #     OGLE_collections = json.load(f)
    # types_to_process = ["CEP", "RRLYR", "DSCT", "T2CEP", "ACEP"]
    # for type in types_to_process:
    #     catalogs_to_process.extend(OGLE_collections[type])

    # Create empty lists to store dataset entries
    dataset_entries = []

    # List of IDs that don't have light curves
    no_lc_ids = []

    print(f"Processing {len(catalogs_to_process)} catalogs")
    for catalog_to_process in catalogs_to_process:
        region = catalog_to_process[0].lower()
        parent_type = catalog_to_process[1].lower()
        sub_type = catalog_to_process[2]

        start_time = time.time()
        print(f"  {region.upper()} {parent_type.upper()} {sub_type}")
        print("  Started catalog-level data")

        cat = load_catalog(*catalog_to_process)
        duration = time.time() - start_time
        print(f"  Loaded catalog ({duration:.2f}s; {len(cat) / duration:.0f} stars/s)")
        start_time = time.time()
        
        cat = merge_remarks(*catalog_to_process, cat)
        duration = time.time() - start_time
        print(f"  Merged remarks ({duration:.2f}s; {len(cat) / duration:.0f} stars/s)")
        start_time = time.time()

        cat = merge_ident(*catalog_to_process, cat)
        cat.reset_index(drop=True, inplace=True)
        duration = time.time() - start_time
        print(f"  Merged ident ({duration:.2f}s; {len(cat) / duration:.0f} stars/s)")
        print(f"  Finished catalog-level data")

        lc_base_dir = f"../../../data/ogle4_raw/OCVS/{region}/{parent_type}/"
        template_lc_glob_path = [
            lc_base_dir + f"*phot*/BAND/{star_ID}.dat"
            for star_ID in cat['sourceid']
        ]

        # Use multiprocessing to read light curves in parallel
        with Pool(processes=num_workers) as pool:
            # Map read_light_curve function over template_lc_glob_path
            multiband_lcs = list(tqdm(
                pool.imap(read_light_curve, template_lc_glob_path),
                total=len(template_lc_glob_path),
                desc=f"  Processing {catalog_to_process[0]} {catalog_to_process[2]} light curves",
                unit="stars"
            ))

        # Process results and create entries
        start_time = time.time()
        print("  Started collating light curve and catalog data")

        # Create a mapping from sourceid to star_info for quick lookup
        star_info_map = cat.set_index('sourceid').to_dict(orient='index')

        valid_mask = [lc is not None for lc in multiband_lcs]
        valid_star_ids = cat['sourceid'][valid_mask]
        valid_multiband_lcs = [lc for lc in multiband_lcs if lc is not None]

        dataset_entries.extend([
            star_info_map[star_id] | {"bands_data": lc} | {"sourceid": star_id}
            for star_id, lc in zip(valid_star_ids, valid_multiband_lcs)
        ])

        # Track IDs without light curves
        no_lc_ids.extend(cat['sourceid'][~np.array(valid_mask)])
        print(f"  Finished collating ({time.time() - start_time:.2f}s)\n")

    # Create HuggingFace dataset
    start_time = time.time()
    print("\nRegistering all stars in a huggingface dataset")
    dataset = Dataset.from_list(dataset_entries, features=schema)
    print(f"Created dataset with {len(dataset_entries)} entries ({time.time() - start_time:.2f}s)")

    if len(no_lc_ids) > 0:
        print(f"No lightcurve data found for {len(no_lc_ids)} IDs")
    else:
        print("Found lightcurve data for all stars")

    return dataset


if __name__ == "__main__":
    global_start_time = time.time()
    parser = argparse.ArgumentParser(description='Process OGLE data to HuggingFace format')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for parallel processing (default: 4)')

    args = parser.parse_args()
    num_workers = args.num_workers

    dataset = create_dataset(num_workers)
    dataset.save_to_disk(
        "../../../data/ogle4",
        num_proc=num_workers,  # save_to_disk does not support multiprocessing
        max_shard_size="100MB",
    )
    print(f"Done writing OGLE data to HF format ({time.time() - global_start_time:.2f}s)\n")
