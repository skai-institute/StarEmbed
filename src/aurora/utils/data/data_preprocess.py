from data_loading import load_csdr1_raw, load_csdr1_meta
from data_output import output_hf_dataset, save_dataset_with_script
import pandas as pd
import numpy as np
import os
from pathlib import Path

def mjd_to_datetime(mjd):
    # MJD epoch is November 17, 1858
    mjd_epoch = pd.Timestamp('1858-11-17')
    # Convert MJD to timedelta days and add to epoch
    return mjd_epoch + pd.TimedeltaIndex(mjd, unit='D')

def filter_short_observations(df: pd.DataFrame, threshold: int = 32):
    return df.query('(nobs_g > 32 and band == "g") or (nobs_r > 32 and band == "r") or (nobs_i > 32 and band == "i")')

def filter_bad_datapoints(df: pd.DataFrame):
    return df.query('catflags == 0')

def create_mjd_to_timestamp(df: pd.DataFrame):
    return df.assign(timestamp=lambda x: mjd_to_datetime(x['mjd']))

def create_mjd_to_delta_t(df: pd.DataFrame):
    delta_t = df.groupby('item_id')['timestamp'].diff().dt.days
    
    result = df.copy()
    result['delta_t'] = delta_t
    
    return result

def create_item_id_with_band(df: pd.DataFrame):
    return df.assign(item_id=lambda x: x['ps1_objid'].astype(str) + '_' + x['band'].astype(str))

def rename_cols(df: pd.DataFrame):
    return df.rename(columns={'mag': 'target', 'magerr': 'target_unc'})

def drop_cols(df: pd.DataFrame):
    return df.drop(columns=['ps1_objid', 'band', 'nobs_g', 'nobs_r', 'nobs_i'])

def sort_by_timestamp(df: pd.DataFrame):
    return df.sort_values(by=['timestamp'])




def process_csdr1_raw(path: str = None):
    all_objects = load_csdr1_raw(path)
    all_objects_meta = load_csdr1_meta()
    obj_count = 0
    dataset_records = []
    for i, obj in enumerate(all_objects):
        if obj is not None and not obj.empty:                
            cols = ['ps1_objid', 'mjd', 'band', 'mag', 'magerr', 'nobs_g', 'nobs_r', 'nobs_i', 'catflags']
            df_for_forecast = (
                obj
                [cols]
                .pipe(filter_bad_datapoints)
                .pipe(filter_short_observations)
                .reset_index(drop=True)
                .pipe(create_mjd_to_timestamp)
                .pipe(create_item_id_with_band)
                .pipe(drop_cols)
                .pipe(rename_cols)
                .pipe(sort_by_timestamp)
                .pipe(create_mjd_to_delta_t)
                .fillna(0)
            )

            if df_for_forecast.empty:
                continue

            # print(df_for_forecast.columns)

            star_id = df_for_forecast['item_id'].iloc[0]
            ps1_objid = df_for_forecast['item_id'].iloc[0].split('_')[0]
            start_date = df_for_forecast['timestamp'].iloc[0]
            lc_class = all_objects_meta['class'].iloc[i]
            lc_period = all_objects_meta['P'].iloc[i]
            csdr1_id = all_objects_meta['ID'].iloc[i]
            freq_str = f"1D"
            mag_float32 = df_for_forecast['target'].values.astype(np.float32)
            mag_err_float32 = df_for_forecast['target_unc'].values.astype(np.float32)
            delta_t = df_for_forecast['delta_t'].values.astype(np.float32)
            mjd_float32 = df_for_forecast['mjd'].values.astype(np.float32)

            # Create a record for this star
            record = {
                "item_id": star_id,
                "start": pd.Timestamp(start_date),
                "freq": freq_str,
                "target": mag_float32,
                "past_feat_dynamic_real": mag_err_float32,
                "feat_dynamic_real": delta_t,
                "period": np.float32(lc_period),
                "ps1_objid": star_id,
                "mjd": mjd_float32,
                "class": lc_class,
                "csdr1_id": csdr1_id
            }
            dataset_records.append(record)
            obj_count += 1
            print(f"Processed {obj_count} objects")
            

    # print(f"Processed {obj_count} objects")
    return output_hf_dataset(dataset_records)

if __name__ == "__main__":
    hf_csdr1_raw = process_csdr1_raw()
    save_dataset_with_script(
        dataset=hf_csdr1_raw,
        dataset_path=Path("hf_csdr1_raw4_catflags_filtered_with_labels"),
        num_shards=20,
        num_proc=os.cpu_count()-2
    )