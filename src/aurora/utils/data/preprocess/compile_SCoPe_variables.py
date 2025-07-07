import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm
import glob
import os
from typing import List

# Note that filepaths won't work here as is - they're written to run within Nabeel's personal directory

"""
This file includes code to extract variables stars from the scope data and save them as csv files.

The default threshold is 0.95 based on the scores from an XGBoost model.
"""

def write_vs_csv(field_file: List[str]):

    """
    
    
    """


    file_chunk = field_file[0:5]
    field_num = field_file[-7:-4]

    field = pd.read_csv(field_file)
    print(f"Stars in chunk {file_chunk}, field {field_num}:", len(field))
    
    print("  xgb vnv stats:")
    print("    Min, Max vnv:", np.min(field['vnv_xgb']), np.max(field['vnv_xgb']))
    print(f"    Mean, Std vnv: {np.mean(field['vnv_xgb']):.5f}, {np.std(field['vnv_xgb']):.5f}")
    print(f"    Median vnv: {np.median(field['vnv_xgb']):.5f}")
    
    field_vs = field.loc[field['vnv_xgb'] > 0.95]
    print(f"  Stars with vnv xgb > 0.95: {len(field_vs)} ({len(field_vs)/len(field)*100:.2f}%)")
    
    field_vs_dir = os.path.join(
        field_file.split('/')[0],
        f"{file_chunk}_xgb_095_vnv",
    )
    os.makedirs(field_vs_dir, exist_ok=True)
        
    field_vs.to_csv(os.path.join(field_vs_dir, f"field_{field_num}_vs.csv"))
    
    print()


def process_all_fields():
    field_files = glob.glob("*_*_prediction_xgb_dnn_fields/*")
    print(len(field_files), "fields")

    for field_file in field_files:
        write_vs_csv(field_file)


def merge_catalogs():
    var_field_catalogs = glob.glob("*_*_prediction_xgb_dnn_fields/*_*_xgb_095_vnv/field_*_vs.csv")
    print(len(var_field_catalogs), "fields processed")

    all_fields = pd.DataFrame()

    for var_field_catalog in tqdm.tqdm(var_field_catalogs):
        field = pd.read_csv(var_field_catalog)
        field = field[["_id", "Gaia_EDR3___id", "AllWISE___id",
                    "PS1_DR1___id", "ra", "dec", "period",
                    "field", "ccd", "quad", "filter",
                    "vnv_xgb", "vnv_dnn", "pnp_xgb", "pnp_dnn"]]

        all_fields = pd.concat((all_fields, field))

    all_fields.to_csv("all_fields_xgb_095_vnv.csv", index=None)


if __name__ == "__main__":
    process_all_fields()
    merge_catalogs()