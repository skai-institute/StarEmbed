import pandas as pd
import numpy as np
import argparse
import tqdm
import glob
import os

# TODO: make this general or enforce the SCoPe directory exists at a particular place
path_to_SCoPe = "../../../../SCoPe/"


def get_cut_str(cuts):
    """
    Produce a string in the form {model}{col}{threshold} for each cut, separated by "_"
    e.g., "xgbpnp095_xgbvnv095"
    """
    return "_".join([cut[1]+cut[0]+str(cut[2]).replace(".", "") for cut in cuts])


def write_cut_catalog(field_file, cuts):
    """
    Read in a field file, apply cuts, and write out the cut catalog.
    Cut fields are written to a new directory formatted as {chunk}_{cut_str}.
        chunk is the RA range which SCoPe organizes fields into, e.g., "14_13"
        see get_cut_str for cut_str formatting.

    Parameters
    ----------
    field_file : str
        Path to the field file to be processed.
    cuts : list of tuples
        List of cuts to be applied, where each cut is a tuple of the form (model, col, threshold).
    """
    file_chunk = field_file[-45:-40]
    field_num = field_file[-7:-4]

    field = pd.read_csv(field_file)
    cut_field = field.copy()  # make a copy of the field dataframe to apply cuts to
    print(f"Stars in chunk {file_chunk}, field {field_num}:", len(field))

    # Apply each cut iteratively
    for cut in cuts:
        # parse the cut tuple
        model, col, threshold = cut

        # for given classification and model, get the column name as it appears in SCoPe files
        col_name = col + "_" + model

        print(f"  {model} {col} stats:")
        print(f"    Min, Max {col}:", np.min(field[col_name]), np.max(field[col_name]))
        print(f"    Mean, Std {col}: {np.mean(field[col_name]):.5f}, {np.std(field[col_name]):.5f}")
        print(f"    Median {col}: {np.median(field[col_name]):.5f}")

        # Apply the cut
        cut_field = cut_field.loc[cut_field[col_name] > threshold]
        print(f"  Remaining stars with {col} {model} > {threshold}: {len(cut_field)}" +
              f"({len(cut_field)/len(field)*100:.2f}%) of original")

    # Prepare the file path to write the cut field file to and create the directory if necessary
    full_cut_str = get_cut_str(cuts)
    fields_dir = os.path.join(
        os.path.dirname(field_file),
        f"{file_chunk}_{full_cut_str}",
    )
    os.makedirs(fields_dir, exist_ok=True)

    # Write the cut field file to disk as a CSV
    cut_field.to_csv(os.path.join(fields_dir, f"field_{field_num}_cut.csv"))
    print()


def process_all_fields(cuts):
    """
    Collects all field files from the SCoPe directory and calls write_cut_catalog
    for each field.
    As of May 2025, SCoPe has 229 fields completed

    Parameters
    ----------
    cuts : list of tuples
        List of cuts to be applied, where each cut is a tuple of the form (model, col, threshold).
    """
    # Gather all field files based on SCoPe directory structure
    field_files = glob.glob(path_to_SCoPe+"*_*_prediction_xgb_dnn_fields/field_*.csv")
    print(len(field_files), "fields")

    # Apply cuts to each field file and write the cut file to disk
    for idx, field_file in enumerate(field_files[150:]):
        write_cut_catalog(field_file, cuts)
        if (idx % 50 == 0) and (idx > 0):
            print(f"------- {idx+1} fields done of {len(field_files)} -------\n")


def merge_catalogs(cuts):
    """
    Convinience function which reads all fields with cuts applied by write_cut_catalog
    and merges them into a single csv titled all_fields_{cut_str}.csv
        see get_cut_str for cut_str formatting.

    Parameters
    ----------
    cuts : list of tuples
        List of cuts to be applied, where each cut is a tuple of the form (model, col, threshold).
    """
    # Find all field files with the specified set of cuts applied
    full_cut_str = get_cut_str(cuts)
    var_field_catalogs = glob.glob(
        path_to_SCoPe+f"*_*_prediction_xgb_dnn_fields/*_*_{full_cut_str}/field_*_cut.csv"
    )
    print(len(var_field_catalogs), "fields processed")

    # Gather relevant columns to keep in the merged catalog
    all_fields = pd.DataFrame()
    cut_cols = [cut[1] + "_" + cut[0] for cut in cuts]

    for var_field_catalog in tqdm.tqdm(var_field_catalogs):
        field = pd.read_csv(var_field_catalog)
        field = field[[
            "_id", "Gaia_EDR3___id", "AllWISE___id", "PS1_DR1___id", "ra", "dec",
            "period", "field", "ccd", "quad", "filter", "vnv_xgb", "vnv_dnn",
            "pnp_xgb", "pnp_dnn"
        ] + cut_cols]

        all_fields = pd.concat((all_fields, field))

    # Write the merged catalog to disk
    all_fields.to_csv(path_to_SCoPe + f"all_fields_{full_cut_str}.csv", index=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a catalog of ZTF stars using SCoPe classifications and specified cuts."
    )
    parser.add_argument(
        '--cuts',
        nargs='+',
        required=True,
        help="Cuts to apply, formatted as model,col,threshold (e.g., xgb,pnp,0.95)." +
             "model can be xgb or dnn;" +
             "col can be any SCoPe classification (see Healy+2024; DOI 10.3847/1538-4365/ad33c6)" +
             "threshold can be a float [0,1]." +
             "Multiple cuts can be provided and should be separated by a +."
    )
    args = parser.parse_args()

    # Parse the cuts into a list of tuples (model, col, threshold) as expected by all functions here
    cuts = []
    cut_arg = args.cuts[0].split('+')
    for cut in cut_arg:
        model, col, threshold = cut.split(',')
        cuts.append((model, col, float(threshold)))

    # Apply the cuts to the SCoPe catalogs
    process_all_fields(cuts)

    # Merge the catalogs into a single file
    merge_catalogs(cuts)
