import pandas as pd
import numpy as np
import tqdm
import glob
import os

path_to_SCoPe = "../SCoPe/"
# Note that filepaths won't work here as is - they're written to run within Nabeel's personal directory


def get_cut_str(cuts):
    return "_".join([cut[1]+cut[0]+str(cut[2]).replace(".", "") for cut in cuts])


def write_cut_catalog(field_file, cuts):
    file_chunk = field_file[-45:-40]
    field_num = field_file[-7:-4]

    field = pd.read_csv(field_file)
    cut_field = field.copy()
    print(f"Stars in chunk {file_chunk}, field {field_num}:", len(field))

    for cut in cuts:
        model, col, threshold = cut
        cut_str = col + "_" + model

        print(f"  {model} {col} stats:")
        print(f"    Min, Max {col}:", np.min(field[cut_str]), np.max(field[cut_str]))
        print(f"    Mean, Std {col}: {np.mean(field[cut_str]):.5f}, {np.std(field[cut_str]):.5f}")
        print(f"    Median {col}: {np.median(field[cut_str]):.5f}")

        cut_field = cut_field.loc[cut_field[cut_str] > threshold]
        print(f"  Remaining stars with {col} {model} > {threshold}: {len(cut_field)} ({len(cut_field)/len(field)*100:.2f}%) of original")

    full_cut_str = get_cut_str(cuts)
    field_vs_dir = os.path.join(
        path_to_SCoPe,
        field_file.split('/')[2],
        f"{file_chunk}_{full_cut_str}",
    )
    # print(field_vs_dir)
    os.makedirs(field_vs_dir, exist_ok=True)

    cut_field.to_csv(os.path.join(path_to_SCoPe, field_vs_dir, f"field_{field_num}_cut.csv"))

    print()


def process_all_fields(cuts):
    field_files = glob.glob(path_to_SCoPe+"*_*_prediction_xgb_dnn_fields/field_*.csv")
    print(len(field_files), "fields")

    for idx, field_file in enumerate(field_files[150:]):
        write_cut_catalog(field_file, cuts)
        if (idx % 50 == 0) and (idx > 0):
            print(f"------- {idx+1} fields done of {len(field_files)} -------\n")


def merge_catalogs(cuts):
    full_cut_str = get_cut_str(cuts)
    var_field_catalogs = glob.glob(path_to_SCoPe+f"*_*_prediction_xgb_dnn_fields/*_*_{full_cut_str}/field_*_cut.csv")
    print(len(var_field_catalogs), "fields processed")

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

    all_fields.to_csv(path_to_SCoPe + f"all_fields_{full_cut_str}.csv", index=None)


if __name__ == "__main__":
    cuts = [
        ("xgb", "pnp", 0.95)
    ]

    process_all_fields(cuts)
    merge_catalogs(cuts)
