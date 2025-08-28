from datasets import Dataset, Features, Value, Sequence
from astropy.coordinates import Angle
import astropy.units as u
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse

class_map = {
    1: "EW", 2: "EA", 3: "Beta_Lyrae", 4: "RRab",
    5: "RRc", 6: "RRd", 7: "Blazhko", 8: "RS CVn",
    9: "ACEP", 10: "Cep-II", 11: "HADS", 12: "LADS",
    13: "LPV", 14: "ELL", 15: "Hump", 16: "PCEB",
    17: "EA_UP"
}


def read_catalina_lightcurves(lcs_path):
    CSDR_lcs = pd.read_csv(
        lcs_path,
        names=["ID", "mjd", "mag", "magunc", "RA", "Dec"]
    )

    # Remove rows with negative mjd
    CSDR_lcs = CSDR_lcs[CSDR_lcs['mjd'] > 0]

    # Sort by mjd - for matching with catalog later
    CSDR_lcs.sort_values(by='mjd', inplace=True)
    CSDR_lcs.reset_index(drop=True, inplace=True)

    print(len(CSDR_lcs), "total observations;", len(CSDR_lcs['ID'].unique()), "unique stars")
    # CSDR_lcs.head()
    return CSDR_lcs


def read_catalina_catalog(cat_path):
    CSDR_cat = pd.read_csv(
        cat_path, header=1, sep='\s+',
        names=[
            'CS_ID', 'Numerical_ID', 'RA', 'Dec', 'magV', 'P[d]', 'Amp', 'n_obs', 'class'
        ], dtype=str
    )

    # Remove entries with missing periods
    CSDR_cat = CSDR_cat[CSDR_cat['P[d]'] != "\\\\nodata"]

    # Convert RA and Dec to decimal degrees
    CSDR_cat['RA_deg_decimal'] = Angle(CSDR_cat['RA'], unit=u.hourangle).degree
    CSDR_cat['Dec_deg_decimal'] = Angle(CSDR_cat['Dec'], unit=u.deg).degree

    # Convert class integers to strings
    CSDR_cat['class_str'] = CSDR_cat['class'].astype(int).map(class_map)

    print(len(CSDR_cat['Numerical_ID'].unique()), "unique stars")
    # CSDR_cat.head()
    return CSDR_cat


def create_dataset(lcs_path, cat_path):
    lcs = read_catalina_lightcurves(lcs_path)
    cat = read_catalina_catalog(cat_path)

    # Standardized StarEmbed schema with some columns unique to Catalina
    band_schema = Features({
        "mjd": Sequence(feature=Value("float64")),
        "target": Sequence(feature=Value("float64")),
        "past_feat_dynamic_real": Sequence(feature=Value("float64")),
        "feat_dynamic_real": Sequence(feature=Value("float64")),
        "length": Value("int64"),
    })

    schema = Features({
        "sourceid": Value("string"),
        "numerical_id": Value("string"),
        "bands_data": {
            "C": band_schema,  # Catalina observations do not use a filter, denoted as C for "clear"
        },
        "avg_mag_V": Value("float64"),
        "period": Value("float64"),
        "class_str": Value("string"),
        "class_int": Value("int64"),
        "ra": Value("float64"),
        "dec": Value("float64")
    })

    # Create empty lists to store dataset entries
    dataset_entries = []

    # IDs with no lightcurve data
    no_lc_ids = []

    # Iterate through each catalog entry
    for _, cat_row in tqdm(cat.iterrows(), total=len(cat)):
        numerical_id = cat_row['Numerical_ID']

        # Get corresponding lightcurve data
        lc_data = lcs[lcs['ID'] == int(numerical_id)]

        if len(lc_data) == 0:
            # print(f"No lightcurve data found for {numerical_id}")
            no_lc_ids.append(numerical_id)
            continue

        # Create entry following schema
        entry = {
            "sourceid": cat_row['CS_ID'],
            "numerical_id": str(numerical_id),
            "bands_data": {
                "C": {
                    "mjd": lc_data['mjd'].tolist(),
                    "target": lc_data['mag'].tolist(),
                    "past_feat_dynamic_real": lc_data['magunc'].tolist(),
                    "feat_dynamic_real": np.diff(lc_data['mjd'].tolist(), prepend=0),
                    "length": len(lc_data)
                }
            },
            "avg_mag_V": float(cat_row['magV']),
            "period": float(cat_row['P[d]']),
            "class_str": cat_row['class_str'],
            "class_int": int(cat_row['class']),
            "ra": cat_row['RA_deg_decimal'],
            "dec": cat_row['Dec_deg_decimal']
        }

        dataset_entries.append(entry)

    # Create HuggingFace dataset
    dataset = Dataset.from_list(dataset_entries, features=schema)

    print(f"Created dataset with {len(dataset_entries)} entries")
    print(f"No lightcurve data found for {len(no_lc_ids)} IDs")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert Catalina data to HuggingFace dataset format' +
        'with the standardized StarEmbed schema.'
    )
    parser.add_argument('--lcs_path', type=str, default='../../../data/Catalina_PVars_DR2.phot',
                        help='Path to lightcurve data file')
    parser.add_argument('--cat_path', type=str, default='../../../data/CSDR1_varstars_v2.txt',
                        help='Path to catalog data file')
    parser.add_argument('--output_dir', type=str, default='../../../data/catalina',
                        help='Directory to save the dataset')
    parser.add_argument('--num_proc', type=int, default=4,
                        help='Number of processes to use when writing the dataset')
    parser.add_argument('--max_shard_size', type=str, default='100MB',
                        help='Maximum size of each shard')

    args = parser.parse_args()

    dataset = create_dataset(args.lcs_path, args.cat_path)
    dataset.save_to_disk(
        args.output_dir,
        num_proc=args.num_proc,
        max_shard_size=args.max_shard_size
    )
