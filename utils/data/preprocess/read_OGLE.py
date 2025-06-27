import pandas as pd
import numpy as np
import glob
import re


def get_period_feature_columns(num_periods):
    """
    Return the column names for the period features in the catalog repeated num_periods times.
    """
    feature_names = [
        'period', 'period_unc', 'time_of_peak[HJD]', 'amp_I', 'fourier_R21',
        'fourier_phi21', 'fourier_R31', 'fourier_phi31'
    ]

    for i in range(2, num_periods + 1):
        feature_names.extend([
            f'period{i}', f'period{i}_unc', f'time_of_peak{i}[HJD]', f'amp{i}_I',
            f'fourier{i}_R21', f'fourier{i}_phi21', f'fourier{i}_R31', f'fourier{i}_phi31'
        ])

    return feature_names


def load_catalog(region, parent_type, sub_type):
    """
    Read in an OGLE catalog for a specific region, parent-type, and sub-type and
    return a dataframe with the catalog data.

    Also add the following columns to the dataframe:
    - remarks: string
    - region: string
    - parent_type: string
    - sub_type: string
    - class: string

    Parameters
    ----------
    region : str
        The OGLE region to search for stars in.
        Valid inputs are: blg, gal, gd, lmc, smc
    parent_type : str
        The parent type of the OGLE catalog to read in.
        Valid inputs are:
            TODO: Remove invalid types
            acep, cep, dsct, ecl, hb, rrlyr, t2cep, lpv, dpv, transits, rot,
            short_period_ecl
    sub_type : str
        The sub-type of the OGLE catalog to read in.
        Valid inputs are:
            TODO: Remove invalid types
            For cep: cep1O, cep1O2O, cep1O2O3O, cep2O3O, cepF, cepF1O
            For dsct: dsct, dsctspecconf
            For dpv: DPV
            For lpv: Miras
            For ecl: ecl, ell
            For hb: hb
            For rot: rot
            For rrlyr: RRab, RRc, RRd, aRRd
            For t2cep: t2cep
            For transits: transits

    Returns
    -------
    catalog : pd.DataFrame
        A dataframe with the catalog information on the stars in the specified
        region of the specified parent-type and sub-type.
    """
    region = region.lower()
    parent_type = parent_type.lower()
    region_class_dir = f"../../../data/ogle4_raw/OCVS/{region}/{parent_type}/"

    if parent_type == "cep":
        if sub_type in ["cep1O", "cepF"]:
            num_periods = 1
        elif sub_type in ["cepF1O", "cep1O2O", "cep2O3O"]:
            num_periods = 2
        elif sub_type in ["cep1O2O3O", "cepF1O2O"]:
            num_periods = 3

        # Catalog files have "-" in place of missing values, so pd.read_csv with
        # the whitespace delimiter is appropriate
        catalog = pd.read_csv(
            region_class_dir + f"{sub_type}.dat", delimiter=r'\s+',
            names=[
                'sourceid', 'avg_mag_I', 'avg_mag_V',
                *get_period_feature_columns(num_periods)
            ]
        )

    # Add empty columns to catalog for extra periods
    extra_features = set(get_period_feature_columns(3)) - \
        set(get_period_feature_columns(num_periods))
    for feature in extra_features:
        catalog[feature] = np.nan

    # Add class column which is combination of parent_type and sub_type
    catalog['remarks'] = ""
    catalog['region'] = region

    # Replace any "-" in any column with NaN
    catalog = catalog.replace("-", np.nan)

    # Add class column which is combination of parent_type and sub_type
    # TODO: Formatting depends on type
    if parent_type == "cep":
        catalog['parent_type'] = parent_type
        catalog['sub_type'] = sub_type[3:]
        catalog['class_str'] = sub_type

    return catalog


def merge_remarks(region, parent_type, sub_type, subtype_df):
    """
    Merge the remarks from the remarks.txt file into the corresponding entry in
    the subtype_df dataframe.

    Parameters
    ----------
    region : str
    parent_type : str
    sub_type : str
        Same as load_catalog
    subtype_df : pd.DataFrame
        A dataframe with the catalog information on the stars in the specified
        region of the specified parent-type and sub-type.

    Returns
    -------
    subtype_df : pd.DataFrame
        The input dataframe with a new 'remarks' column
        Example remarks:
            "OGLE-BLG-CEP-067 double Cepheid P1 = 2.610678 d, P2 = 1.692387 d
            "OGLE-BLG-CEP-097 variable period"
    """
    region = region.upper()
    parent_type = parent_type.upper()
    region_class_dir = f"../../../data/ogle4_raw/OCVS/{region.lower()}/{parent_type.lower()}/"

    # Open the remarks.txt file and loop over its lines
    with open(region_class_dir + "remarks.txt", 'r') as f:
        for remark in f:
            remark = remark[:-1]  # Remove newline character at the end

            # Find each OGLE star mentioned in this remark, and iterate over them
            OGLE_IDs = re.findall(rf'\S*OGLE-{region.upper()}-{parent_type.upper()}\S*', remark)
            for OGLE_ID in OGLE_IDs:
                # Skip if remark is for a star in a different catalog
                if OGLE_ID not in subtype_df['sourceid'].values:
                    continue

                # Get remarks for this OGLE_ID, starts with empty string
                existing_remarks = subtype_df.loc[
                    subtype_df['sourceid'] == OGLE_ID, 'remarks'
                ].values

                # There shouldn't be multiple entries for the same ID
                if len(existing_remarks) > 1:
                    print("Multiple entries with same sourceids:", OGLE_ID)
                    continue

                # If there's already a remark present, add a spacer
                if existing_remarks[0] != "":
                    existing_remarks += " | "

                # Add the new (concatenated) remark into the dataframe
                subtype_df.loc[
                    subtype_df['sourceid'] == OGLE_ID, 'remarks'
                ] = existing_remarks + remark

    # Return the updated dataframe
    return subtype_df


def merge_ident(region, parent_type, sub_type, subtype_df):
    """
    Merge the ident.dat file into the corresponding entry in the subtype_df
    dataframe.

    Parameters
    ----------
    region : str
    parent_type : str
    sub_type : str
    subtype_df : pd.DataFrame
        Same as load_catalog and merge_remarks

    Returns
    -------
    subtype_df : pd.DataFrame
        The input dataframe with a new 'RA', 'Dec', 'OGLE-IV', 'OGLE-III',
        'OGLE-II', 'OtherID' columns
    """
    region = region.upper()
    parent_type = parent_type.upper()
    region_class_dir = f"../../../data/ogle4_raw/OCVS/{region.lower()}/{parent_type.lower()}/"

    # Define the column widths based on the data format
    colspecs = [
        (0, 16),    # Star ID
        (17, 25),   # Type
        (27, 38),   # Right Ascension
        (39, 50),   # Declination
        (52, 68),   # OGLE-IV
        (69, 84),   # OGLE-III
        (85, 100),   # OGLE-II
        (101, 120)   # Additional identifiers
    ]

    # Missing values are represented by whitespace, so read_fwf must be used in place of pd.read_csv
    ident = pd.read_fwf(
        region_class_dir + "ident.dat", colspecs=colspecs,
        names=['sourceid', 'type', 'ra', 'dec',
               'OGLE_IV_id', 'OGLE_III_id', 'OGLE_II_id', 'other_id']
    )
    # ['ID', 'type', 'RAh', 'RAm', 'RAs', 'DE-', 'DEd', 'DEm', 'DEs',
    #  'OGLE-IV', 'OGLE-III', 'OGLE-II', 'OtherID']

    # Clean up any whitespace
    ident = ident.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Create a mapping from ID to the columns we want to copy
    cols_to_copy = ['ra', 'dec', 'OGLE_IV_id', 'OGLE_III_id', 'OGLE_II_id', 'other_id']
    id_to_cols = ident.set_index('sourceid')[cols_to_copy]
    subtype_df[cols_to_copy] = ""

    # For each row in subtype_df, look up the corresponding columns from ident using the ID
    for idx, row in subtype_df.iterrows():
        if row['sourceid'] in id_to_cols.index:
            # Add the new columns into the appropriate row in the dataframe
            subtype_df.loc[idx, cols_to_copy] = id_to_cols.loc[row['sourceid']]

    # Return the updated dataframe
    return subtype_df


def read_light_curve(region, parent_type, sub_type, star_ID):
    """
    Read in the light curve for a given star ID.

    Parameters
    ----------
    region : str
    parent_type : str
    sub_type : str
        Same as load_catalog
    star_ID : str
        The OGLE ID of the star to fetch the light curve for
        e.g., "OGLE-BLG-ECL-000007"

    Returns
    -------
    lc : pd.DataFrame
        A dataframe with the light curve data for the given star
        Columns:
            - mjd: Modified Julian Date
            - mag: Magnitude
            - mag_err: Magnitude error
    """
    region = region.upper()
    parent_type = parent_type.upper()
    region_class_dir = f"../../../data/ogle4_raw/OCVS/{region.lower()}/{parent_type.lower()}/"

    # Need to select all files matching this star ID across I and V band and
    # different formats of phot directories (sometimes phot/ sometimes phot_ogle4/ etc.)
    bands = ["I", "V"]
    multiband_lc = {}

    for band in bands:
        lc_files = glob.glob(region_class_dir + f"*phot*/{band}/{star_ID}.dat")
        if len(lc_files) == 0:
            # print(f"No light curve found for {star_ID}")
            multiband_lc[band] = None
            continue

        lc = pd.read_csv(lc_files[0], delimiter=r'\s+', names=['time', 'mag', 'magunc'], dtype=str)

        # The time column in light curve files are inconsistent and unpredictable
        # Check the length of one time entry to determine its format
        if len(lc['time'][0]) == 13:  # time is HJD, shift to MJD
            lc['mjd'] = lc['time'].astype(np.float64) - 2400000.5
        elif len(lc['time'][0]) == 10:  # time is shifted HJD, shift to MJD
            lc['mjd'] = lc['time'].astype(np.float64) + 2450000 - 2400000.5
        else:
            print(f"Unexpected time format for {star_ID} in band {band}. Expected 10 or 13 digits.")
            print(f"Found {len(str(lc['time'][0]))} digits in {lc['time'][0]}")
            exit(1)

        # Format as expected by bands_data entry in standardized StarEmbed schema
        multiband_lc[band] = {
            "mjd": lc['mjd'].tolist(),
            "target": lc['mag'].astype(np.float64).tolist(),
            "past_feat_dynamic_real": lc['magunc'].astype(np.float64).tolist(),
            "feat_dynamic_real": np.diff(lc['mjd'].tolist(), prepend=0),
            "length": len(lc)
        }

    # If no light curve found for any band, return None
    if all(multiband_lc[band] is None for band in bands):
        return None

    return multiband_lc
