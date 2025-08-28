import pandas as pd
import numpy as np
import glob
import re
import os


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


def format_coordinate(coord_str, is_dec=False):
    """
    Reformat coordinates from space-separated to colon-separated format.
    
    Parameters
    ----------
    coord_str : str
        The coordinate string in space-separated format
    is_dec : bool, default False
        Whether the coordinate is declination (True) or right ascension (False)
    
    Returns
    -------
    str
        The formatted coordinate string in colon-separated format
    """
    if pd.isna(coord_str) or coord_str == "":
        return coord_str
    
    parts = coord_str.split()
    if len(parts) != 3:
        return coord_str
    
    if is_dec:
        # Format: ±dd:mm:ss.s with leading zeros for declination
        sign = "-" if parts[0].startswith("-") else "+"
        abs_deg = parts[0].lstrip("+-")
        formatted_parts = [
            f"{sign}{abs_deg.zfill(2)}",
            parts[1].zfill(2),
            parts[2].zfill(4)
        ]
    else:
        # Format: hh:mm:ss.s with leading zeros for right ascension
        formatted_parts = [
            parts[0].zfill(2),
            parts[1].zfill(2),
            parts[2].zfill(5)
        ]
    
    return ':'.join(formatted_parts)


def add_nonstandard_feats_to_remarks(catalog, nonstandard_feat_names):
    """
    For each source, add "feat_name=feat_value" for each nonstandard feature
    to the remarks column. Separate features with " | ".

    Parameters
    ----------
    catalog : pd.DataFrame
        The catalog dataframe containing the sources and their features.
    nonstandard_feat_names : list of str
        List of nonstandard feature column names to be added to remarks.

    Returns
    -------
    catalog : pd.DataFrame
        The catalog dataframe with the nonstandard features added to the remarks column.
    """
    nonstandard_feats = catalog.loc[:, ['sourceid'] + nonstandard_feat_names]
    catalog = catalog.drop(columns=nonstandard_feat_names)

    feat_cols = [col for col in nonstandard_feats.columns if col != 'sourceid']
    # Create a DataFrame of "colname=val" strings
    remarks_df = nonstandard_feats[feat_cols].apply(
        lambda col: f"{col.name}=" + col.astype(str), axis=0
    )

    # Join all features for each row with " | "
    remarks_series = remarks_df.agg(" | ".join, axis=1)

    # Assign to catalog['remarks'] by matching sourceid
    catalog = catalog.merge(
        nonstandard_feats[['sourceid']].assign(remarks=remarks_series),
        on='sourceid', how='left', suffixes=('', '_new')
    )

    # Handle the case where remarks_new might not exist due to merge suffixes
    if 'remarks_new' in catalog.columns:
        catalog['remarks'] = catalog['remarks_new']
        catalog = catalog.drop(columns=['remarks_new'])
    # If no existing remarks column, the merge created 'remarks' directly
    return catalog


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
            acep, cep, dsct, ecl, hb, rrlyr, t2cep, lpv, transits, rot
    sub_type : str
        The sub-type of the OGLE catalog to read in.
        Valid inputs are:
            For cep: cep1O, cep1O2O, cep1O2O3O, cep2O3O, cepF, cepF1O
            For dsct: dsct
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

    if parent_type in ["cep", "rrlyr", "t2cep", "acep"]:
        if sub_type in ["cepF", "cep1O", "cep2O", "RRab", "RRc", "t2cep", "acepF", "acep1O"]:
            num_periods = 1
        elif sub_type in ["cepF1O", "cep1O2O", "cep1O3O", "cep2O3O", "RRd", "aRRd"]:
            num_periods = 2
        elif sub_type in ["cepF1O2O", "cep1O2O3O"]:
            num_periods = 3
        else:
            raise NotImplementedError(f"Subtype {sub_type} period count not implemented")

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
    elif parent_type == "dsct":
        # sh is a constant shift in the column positions for different regions relative to blg
        if region in ["blg", "lmc"]:
            sh = 0
        elif region == "smc":
            sh = -1
        elif region == "gd":
            sh = -2
        else:
            raise NotImplementedError(f"DSCT {region} not implemented")

        colspecs = [
            (0, 19 + sh), (21 + sh, 27 + sh), (28 + sh, 34 + sh),
            # Period 1 (period, period_unc, time_of_peak, I-band amplitude)
            (36 + sh, 46 + sh), (47 + sh, 57 + sh), (59 + sh, 69 + sh), (71 + sh, 76 + sh),
            # Period 1 Fourier coefficients
            (78 + sh, 83 + sh), (84 + sh, 89 + sh), (91 + sh, 96 + sh), (97 + sh, 102 + sh),
            # Period 2
            (104 + sh, 114 + sh), (115 + sh, 125 + sh), (127 + sh, 137 + sh), (139 + sh, 144 + sh),
            (146 + sh, 151 + sh), (152 + sh, 157 + sh), (159 + sh, 164 + sh), (165 + sh, 170 + sh),
            # Period 3
            (172 + sh, 182 + sh), (183 + sh, 193 + sh), (195 + sh, 205 + sh), (207 + sh, 212 + sh),
            (214 + sh, 219 + sh), (220 + sh, 225 + sh), (227 + sh, 232 + sh), (234 + sh, 238 + sh)
        ]
        catalog = pd.read_fwf(
            region_class_dir + f"{sub_type}.dat", colspecs=colspecs,
            names=[
                'sourceid', 'avg_mag_I', 'avg_mag_V', *get_period_feature_columns(3)
            ]
        )
    elif parent_type == "lpv":
        catalog = pd.read_csv(
            # Catalog file named "Miras.dat", need to add "s"
            region_class_dir + f"{sub_type}s.dat", delimiter=r'\s+',
            names=[
                'sourceid', 'avg_mag_I', 'avg_mag_V', 'period', 'amp_I'
            ]
        )

        # Add empty columns to catalog for extra periods
        extra_features = set(get_period_feature_columns(3)) - \
            set(get_period_feature_columns(1))
        extra_features = list(extra_features) + [
            'period_unc', 'time_of_peak[HJD]',
            'fourier_R21', 'fourier_phi21', 'fourier_R31', 'fourier_phi31'
        ]
        for feature in extra_features:
            catalog[feature] = np.nan
    elif parent_type == "hb":
        if region in ["blg", "lmc", "smc"]:
            sh = 0
        else:
            raise NotImplementedError(f"HB {region} not implemented")
        
        colspecs = [
            ( 0     , 16 + sh), (17 + sh, 23 + sh), (24 + sh, 30 + sh),
            (31 + sh, 44 + sh), (45 + sh, 55 + sh), (56 + sh, 61 + sh),
            (62 + sh, 68 + sh), (69 + sh, 74 + sh), (75 + sh, 81 + sh),
            (82 + sh, 91 + sh), (92 + sh, 94 + sh),
        ]
        
        catalog = pd.read_fwf(
            region_class_dir + f"{sub_type}.dat", colspecs=colspecs,
            names=[
                'sourceid', 'avg_mag_I', 'avg_mag_V', 'period', 't_peri_passage', 'amp_I',
                'orbit_ecc', 'orbit_incl', 'arg_peri', 'variability', 'model_flag'
            ]
        )

        # Add non-standard features into remarks column
        nonstandard_feat_names = [
            't_peri_passage', 'orbit_ecc', 'orbit_incl', 'arg_peri', 'variability', 'model_flag'
        ]
        catalog = add_nonstandard_feats_to_remarks(catalog, nonstandard_feat_names)

        # Add empty columns to catalog for extra periods
        extra_features = set(get_period_feature_columns(3)) - set(catalog.columns)
        for feature in extra_features:
            catalog[feature] = np.nan
    elif parent_type == "ecl":
        if region in ["blg"]:
            colspecs = [
                (0, 19), (21, 27), (28, 34), (35, 47), (49, 58), (60, 65), (66, 71)
            ]
        elif region in ["lmc", "smc"]:
            colspecs = [
                (0, 18), (20, 26), (27, 33), (34, 46), (48, 60), (62, 67), (68, 73)
            ]
        else:
            raise NotImplementedError(f"ECL {region} not implemented")
        
        catalog = pd.read_fwf(
            region_class_dir + f"{sub_type}.dat", colspecs=colspecs,
            names=[
                'sourceid', 'max_mag_I', 'max_mag_V', 'period', 't_p_ecl',
                'depth_p_ecl', 'depth_s_ecl'
            ]
        )
        
        # Add non-standard features into remarks column
        nonstandard_feat_names = [
            'max_mag_I', 'max_mag_V', 't_p_ecl', 'depth_p_ecl', 'depth_s_ecl'
        ]
        catalog = add_nonstandard_feats_to_remarks(catalog, nonstandard_feat_names)

        # Add empty columns to catalog for extra periods
        extra_features = set(get_period_feature_columns(3)) - set(catalog.columns)
        extra_features = list(extra_features) + ['avg_mag_I', 'avg_mag_V']
        for feature in extra_features:
            catalog[feature] = np.nan
    elif parent_type == "rot":
        catalog = pd.read_csv(
            region_class_dir + f"{sub_type}.dat", delimiter=r'\s+',
            names=[
                'sourceid', 'avg_mag_V', 'p_amp_V', 'avg_mag_I', 'p_amp_I', 'period'
            ]
        )
        
        # Add non-standard features into remarks column
        nonstandard_feat_names = ['p_amp_V', 'p_amp_I']
        catalog = add_nonstandard_feats_to_remarks(catalog, nonstandard_feat_names)

        # Add empty columns to catalog for extra periods
        extra_features = set(get_period_feature_columns(3)) - set(catalog.columns)
        for feature in extra_features:
            catalog[feature] = np.nan
    elif parent_type == "transits":
        catalog = pd.read_csv(
            region_class_dir + f"{sub_type}.dat", delimiter=r'\s+',
            names=[
                'sourceid', 'avg_mag_I', 'avg_mag_V', 'period', 't_inferior_conj',
                't_14', 'depth', 'prob_planet', 'SNR'
            ]
        )
        # Add non-standard features into remarks column
        nonstandard_feat_names = ['t_inferior_conj', 't_14', 'depth', 'prob_planet', 'SNR']
        catalog = add_nonstandard_feats_to_remarks(catalog, nonstandard_feat_names)

        # Add empty columns to catalog for extra periods
        extra_features = set(get_period_feature_columns(3)) - set(catalog.columns)
        for feature in extra_features:
            catalog[feature] = np.nan
    else:
        raise NotImplementedError(f"Parent type {parent_type} not implemented")

    # Replace any "-" in any column with NaN
    catalog = catalog.mask((catalog == "-") | (catalog == ""), np.nan)

    # Certain classes create the remarks column earlier, but create it now for the rest 
    if 'remarks' not in catalog.columns:
        catalog['remarks'] = ""
    catalog['region'] = region

    # Add class column which is combination of parent_type and sub_type
    if parent_type in ["cep", "rrlyr", "lpv", "rot", "transits"]:
        catalog['parent_type'] = parent_type
        catalog['sub_type'] = sub_type
        catalog['class_str'] = sub_type
    elif parent_type in ["dsct", "t2cep", "acep", "hb", "ecl"]:
        catalog['parent_type'] = parent_type
        # Populated in merge_ident()
        catalog['sub_type'] = ""
        catalog['class_str'] = ""
    else:
        raise NotImplementedError(f"Parent type {parent_type} not implemented")

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

    # If remarks file does not exist, return the dataframe with empty remarks
    remarks_file = region_class_dir + "remarks.txt"
    if not os.path.exists(remarks_file):
        subtype_df['remarks'] = ""
        print(f"  No remarks file found for {region} {parent_type} {sub_type}")
        return subtype_df

    # Open the remarks.txt file and loop over its lines
    with open(remarks_file, 'r') as f:
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
    # Differences come from regions names and ID numbers having different numbers of digits
    # Not always a constant shift in positions, so each needs to be defined separately
    if parent_type == "CEP":
        if region == "BLG":
            sh = 0
        elif region in ["GD", "LMC", "SMC"]:
            sh = 1

        colspecs = [
            # Star ID, Type, RA, Dec
            (0, 16 + sh), (17 + sh, 25 + sh), (27 + sh, 38 + sh), (39 + sh, 50 + sh),
            # OGLE-II, OGLE-III, OGLE-IV, Additional identifiers
            (52 + sh, 68 + sh), (69 + sh, 84 + sh), (85 + sh, 100 + sh), (101 + sh, 120 + sh)
        ]
    elif parent_type == "RRLYR":
        if region in ["BLG", "LMC"]:
            sh = 0
        elif region in ["GD", "SMC"]:
            sh = -1

        colspecs = [
            (0, 20 + sh), (22 + sh, 26 + sh), (28 + sh, 39 + sh), (40 + sh, 51 + sh),
            (53 + sh, 69 + sh), (70 + sh, 85 + sh), (86 + sh, 101 + sh), (102 + sh, 121 + sh)
        ]
    elif parent_type == "DSCT":
        if region in ["BLG", "LMC"]:
            sh = 0
        elif region == "SMC":
            sh = -1
        elif region == "GD":
            sh = -2

        colspecs = [
            (0, 19 + sh), (21 + sh, 31 + sh), (33 + sh, 44 + sh), (45 + sh, 56 + sh),
            (58 + sh, 74 + sh), (75 + sh, 90 + sh), (91 + sh, 107 + sh), (108 + sh, 130 + sh)
        ]
    elif parent_type == "LPV":
        if region == "BLG":
            sh = 0
        elif region == "GD":
            sh = -1

        colspecs = [
            (0, 19 + sh), (21 + sh, 25 + sh), (27 + sh, 38 + sh), (39 + sh, 50 + sh),
            (52 + sh, 68 + sh), (69 + sh, 84 + sh), (85 + sh, 101 + sh), (102 + sh, 130 + sh)
        ]
    elif parent_type == "T2CEP":
        if region == "BLG":
            sh = 0
        elif region in ["GD", "LMC"]:
            sh = -1
        elif region == "SMC":
            sh = -2

        colspecs = [
            (0, 19 + sh), (21 + sh, 26 + sh), (28 + sh, 39 + sh), (40 + sh, 51 + sh),
            (53 + sh, 69 + sh), (70 + sh, 85 + sh), (86 + sh, 101 + sh), (102 + sh, 130 + sh)
        ]
    elif parent_type == "ACEP":
        if region in ["GAL", "LMC", "SMC"]:
            sh = 0

        colspecs = [
            (0, 17 + sh), (18 + sh, 21 + sh), (23 + sh, 34 + sh), (35 + sh, 46 + sh),
            (48 + sh, 64 + sh), (65 + sh, 80 + sh), (81 + sh, 96 + sh), (97 + sh, 130 + sh)
        ]
    elif parent_type == "HB":
        if region in ["BLG", "LMC", "SMC"]:
            sh = 0

        colspecs = [
            (0      , 16 + sh), (17 + sh, 19 + sh), (20 + sh, 31 + sh), (32 + sh,  43 + sh),
            (44 + sh, 60 + sh), (61 + sh, 76 + sh), (77 + sh, 92 + sh), (93 + sh, 130 + sh)
        ]
    elif parent_type == "ECL":
        if region in ["BLG"]:
            sh = 0
        elif region in ["LMC", "SMC"]:
            sh = -1

        colspecs = [
            (0      , 19 + sh), (21 + sh, 24 + sh), (25 + sh, 36 + sh), (37 + sh,  48 + sh),
            (50 + sh, 66 + sh), (67 + sh, 82 + sh), (83 + sh, 98 + sh), (99 + sh, 130 + sh)
        ]
    elif parent_type == "ROT":
        colspecs = [
            (0, 19), (19, 30), (31, 43), (44, 60), (61, 76), (77, 93), (94, 130)
        ]
    elif parent_type == "TRANSITS":
        colspecs = [
            (0, 12), (13, 24), (25, 36), (37, 53), (54, 69), (70, 85), (86, 131)
        ]
    else:
        raise NotImplementedError(f"Region {region} and parent type {parent_type} not implemented")

    # Missing values are represented by whitespace, so read_fwf must be used in place of pd.read_csv
    # Only ROT and TRANSITS have no type column
    if parent_type not in ["ROT", "TRANSITS"]:
        ident = pd.read_fwf(
            region_class_dir + "ident.dat", colspecs=colspecs,
            names=['sourceid', 'type', 'ra', 'dec',
                   'OGLE_IV_id', 'OGLE_III_id', 'OGLE_II_id', 'other_id']
        )
    else:
        ident = pd.read_fwf(
            region_class_dir + "ident.dat", colspecs=colspecs,
            names=['sourceid', 'ra', 'dec',
                   'OGLE_IV_id', 'OGLE_III_id', 'OGLE_II_id', 'other_id']
        )
    
    # HB stars have RA and Dec terms separated by spaces instead of colons
    if parent_type in ["HB"]:
        ident['ra'] = ident['ra'].apply(format_coordinate)
        ident['dec'] = ident['dec'].apply(lambda x: format_coordinate(x, is_dec=True))

    # Clean up any whitespace
    ident = ident.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Create a mapping from ID to the columns we want to copy
    cols_to_copy = ['ra', 'dec', 'OGLE_IV_id', 'OGLE_III_id', 'OGLE_II_id', 'other_id']
    if parent_type in ["DSCT", "T2CEP", "ACEP", "HB", "ECL"]:
        cols_to_copy.append('type')
    id_to_cols = ident.set_index('sourceid')[cols_to_copy]

    # Set index for fast join
    subtype_df_indexed = subtype_df.set_index('sourceid', drop=False)
    merged = subtype_df_indexed.join(id_to_cols, how='left', rsuffix='_ident')

    # Fill missing columns with empty string
    merged = merged.fillna("")

    # Restore the original index order
    subtype_df[cols_to_copy] = merged[cols_to_copy].values

    if parent_type in ["DSCT", "T2CEP", "ACEP", "HB", "ECL"]:
        subtype_df['sub_type'] = subtype_df['type']
        subtype_df['class_str'] = parent_type.lower() + "_" + subtype_df['type']
        subtype_df.drop(columns=['type'], inplace=True)

    # Return the updated dataframe
    return subtype_df


def read_light_curve(template_lc_glob_path):
    """
    Read in the light curve for a single star ID.
    Slightly clunky implementation to allow support for multiprocessing

    Parameters
    ----------
    template_lc_glob_path : str
        A template light curve path that will be used to find the light curve
        files for a single star ID.
        Example:
            "../../../data/ogle4_raw/OCVS/{region}/{parent_type}/*phot*/BAND/{star_ID}.dat"

    Returns
    -------
    lc : pd.DataFrame
        A dataframe with the light curve data for the given star
        Columns:
            - mjd: Modified Julian Date
            - mag: Magnitude
            - mag_err: Magnitude error
    """
    # Extract OGLE star ID from template lc path
    star_ID = template_lc_glob_path.split("/")[-1].split(".")[0]

    # Need to select all files matching this star ID across I and V band and
    # different formats of phot directories (sometimes phot/ sometimes phot_ogle4/ etc.)
    bands = ["I", "V"]
    multiband_lc = {}

    for band in bands:
        lc_files = glob.glob(template_lc_glob_path.replace("BAND", band))
        if len(lc_files) == 0:
            # print(f"No light curve found for {star_ID}")
            multiband_lc[band] = None
            continue

        # read as str first because we can tell the unit by the length of the time value
        lc = pd.concat([
            pd.read_csv(lc_file, delimiter=r'\s+', names=['time', 'mag', 'magunc'], dtype=str)
            for lc_file in lc_files
        ])
        lc.reset_index(drop=True, inplace=True)

        # The time column in light curve files are HJD but sometimes shifted by a constant
        # Check the length of one time entry to determine its format
        if len(lc['time'][0]) == 13:  # time is HJD, shift to MJD
            lc['mjd'] = lc['time'].astype(np.float64) - 2400000.5
        elif len(lc['time'][0]) in [9, 10]:  # time is shifted HJD, shift to MJD
            lc['mjd'] = lc['time'].astype(np.float64) + 2450000 - 2400000.5
        else:
            print(f"Unexpected time format for {star_ID} {band}band. Expected 9, 10, or 13 digits.")
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
