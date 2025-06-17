import pandas as pd
import re


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
            For cep: 1O, 1O2O, 1O2O3O, 2O3O, F, F1O
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

    if sub_type in ["cep1O"]:
        catalog = pd.read_csv(
            region_class_dir + f"{sub_type}.dat", delimiter=r'\s+',
            names=[
                'ID', 'avg_mag_I', 'avg_mag_V', 'period[d]', 'period_unc[d]',
                'time_of_peak[HJD]', 'amp_I', 'fourier_R21', 'fourier_phi21',
                'fourier_R31', 'fourier_phi31'
            ]
        )

    catalog['remarks'] = ""
    catalog['region'] = region

    # Add class column which is combination of parent_type and sub_type
    # TODO: Formatting depends on type
    if parent_type == "cep":
        catalog['parent_type'] = parent_type
        catalog['sub_type'] = sub_type[3:]
        catalog['class'] = sub_type

    return catalog


def merge_remarks(region, parent_type, sub_type, subtype_df):
    region = region.upper()
    parent_type = parent_type.upper()
    region_class_dir = f"../../../data/ogle4_raw/OCVS/{region.lower()}/{parent_type.lower()}/"

    with open(region_class_dir + "remarks.txt", 'r') as f:
        for remark in f:
            remark = remark[:-1]  # Remove newline character at the end
            OGLE_IDs = re.findall(r'\S*OGLE-BLG-CEP\S*', remark)
            for OGLE_ID in OGLE_IDs:

                # Remark is for a star in a different catalog
                if OGLE_ID not in subtype_df['ID'].values:
                    # print("OGLE_ID not in df:", OGLE_ID)
                    continue

                # Get remarks for this OGLE_ID, starts with empty string
                existing_remarks = subtype_df.loc[subtype_df['ID'] == OGLE_ID, 'remarks'].values

                # There shouldn't be multiple entries for the same ID
                if len(existing_remarks) > 1:
                    print("Multiple entries with same ID:", OGLE_ID)
                    continue

                # If there's already a remark present, add a spacer
                if existing_remarks[0] != "":
                    existing_remarks += " | "

                # Add the new remark into the dataframe
                subtype_df.loc[subtype_df['ID'] == OGLE_ID, 'remarks'] = existing_remarks + remark

    return subtype_df


def merge_ident(region, parent_type, sub_type, subtype_df):
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
        (101, 110)   # Additional identifiers
    ]

    # Read the data
    ident = pd.read_fwf(
        region_class_dir + "ident.dat", colspecs=colspecs,
        names=['ID', 'type', 'RA', 'Dec', 'OGLE-IV', 'OGLE-III', 'OGLE-II', 'OtherID']
    )
    # ['ID', 'type', 'RAh', 'RAm', 'RAs', 'DE-', 'DEd', 'DEm', 'DEs',
    #  'OGLE-IV', 'OGLE-III', 'OGLE-II', 'OtherID']

    # Clean up the data
    ident = ident.apply(lambda x: x.str.strip() if x.dtype == "object" else x)  # Strip whitespace

    # Create a mapping from ID to the columns we want to copy
    cols_to_copy = ['RA', 'Dec', 'OGLE-IV', 'OGLE-III', 'OGLE-II', 'OtherID']
    id_to_cols = ident.set_index('ID')[cols_to_copy]

    # For each row in subtype_df, look up the corresponding columns from ident using the ID
    for idx, row in subtype_df.iterrows():
        if row['ID'] in id_to_cols.index:
            # Convert columns to string type before assignment to avoid dtype incompatibility
            subtype_df.loc[idx, cols_to_copy] = id_to_cols.loc[row['ID']]

    return subtype_df
