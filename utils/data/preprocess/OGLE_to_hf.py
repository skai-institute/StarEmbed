from datasets import Dataset, Features, Value, Sequence

from read_OGLE import (
    load_catalog, merge_remarks, merge_ident, read_light_curve, get_period_feature_columns
)


# Standardized StarEmbed schema with some columns unique to Catalina
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


def create_dataset():
    catalogs_to_process = [
        # region, parent_type, sub_type
        ("blg", "cep", "cep1O"),
        ("blg", "cep", "cepF"),
        ("blg", "cep", "cep1O2O"),
        ("blg", "cep", "cepF1O"),
        ("blg", "cep", "cep2O3O"),
        ("blg", "cep", "cep1O2O3O"),

        ("gd", "cep", "cep1O"),
        ("gd", "cep", "cepF"),
        ("gd", "cep", "cep1O2O"),
        ("gd", "cep", "cepF1O"),
        ("gd", "cep", "cep2O3O"),
        ("gd", "cep", "cep1O2O3O"),
        ("gd", "cep", "cepF1O2O"),
    ]

    # Create empty lists to store dataset entries
    dataset_entries = []

    # List of IDs that don't have light curves
    no_lc_ids = []

    for catalog_to_process in catalogs_to_process:
        cat = load_catalog(*catalog_to_process)
        cat = merge_remarks(*catalog_to_process, cat)
        cat = merge_ident(*catalog_to_process, cat)

        for star_ID in cat['sourceid']:
            star_info = cat[cat['sourceid'] == star_ID].to_dict(orient='records')[0]

            # Get light curve, create entry
            multiband_lc = read_light_curve(*catalog_to_process, star_ID)

            if multiband_lc is None:
                no_lc_ids.append(star_ID)
                continue

            # Create entry following schema
            entry = star_info | {"bands_data": multiband_lc}

            dataset_entries.append(entry)

    # Create HuggingFace dataset
    dataset = Dataset.from_list(dataset_entries, features=schema)

    print(f"Created dataset with {len(dataset_entries)} entries")
    print(f"No lightcurve data found for {len(no_lc_ids)} IDs")
    return dataset


if __name__ == "__main__":
    dataset = create_dataset()
    dataset.save_to_disk(
        "../../../data/ogle4_hf",
        num_proc=4,
        max_shard_size="100MB"
    )
    print("Done writing OGLE data to HF format\n")
