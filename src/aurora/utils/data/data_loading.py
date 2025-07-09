import pickle
import pandas as pd
import datasets
CSDR1_RAW = f"/projects/p32795/weijian/cache/all_objects_47054_None.pkl"
CSDR1_META = f"/projects/p32015/git/moirai_supsup/data_download/CSDR1_varstars.txt"
CSDR1_TRAIN_TEST_SPLIT = f"/projects/p32795/hongyu/hf_csdr1_multiband_raw_lc_subclass_class_str"

def load_csdr1_raw(path=None):
    """
    
    return: a list of pandas dataframes, each containing the light curve for a star
    columns:
    ['index', 'ps1_objid', 'ra', 'dec', 'ps1_gMeanPSFMag', 'ps1_rMeanPSFMag',
       'ps1_iMeanPSFMag', 'nobs_g', 'nobs_r', 'nobs_i', 'mean_mag_g',
       'mean_mag_r', 'mean_mag_i', 'catflags', 'fieldID', 'mag', 'magerr',
       'mjd', 'rcID', 'band', 'Norder', 'Dir', 'Npix']
    """
    if not path:
        path = CSDR1_RAW
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_csdr1_train_test_split(path=None):
    if not path:
        path = CSDR1_TRAIN_TEST_SPLIT
    ds = datasets.load_from_disk(path)
    return ds

def load_csdr1_meta(csv_path=None):
    """
    Load periods from the variable stars catalog
    """
    if not csv_path:
        csv_path = CSDR1_META
    column_names = ['ID', 'RAh', 'RAm', 'RAs', 'Decsign', 'DEm', 'DEs', 
                   'magV', 'P', "Amp", "class", 'flag']
    
    varstars = pd.read_csv(
        csv_path,
        header=33,
        sep=r'\s+',
        names=column_names
    )
    
    return varstars
    
def load_periods(csv_path):
    """
    Load periods from the variable stars catalog
    """
    if not csv_path:
        csv_path = CSDR1_META
    column_names = ['ID', 'RAh', 'RAm', 'RAs', 'Decsign', 'DEm', 'DEs', 
                   'magV', 'P', "Amp", "class", 'flag']
    
    varstars = pd.read_csv(
        csv_path,
        header=33,
        sep=r'\s+',
        names=column_names
    )
    
    # Extract periods as a list
    periods = varstars['P'].values
    return periods

import os, glob
from datasets import load_from_disk, concatenate_datasets, DatasetDict, IterableDataset


def load_sharded_dataset(parent_dir: str, streaming: bool = False):
    """
    Parameters
    ----------
    parent_dir : str
        Path that contains `shard_1/`, `shard_2/`, … (each a HF dataset saved with `save_to_disk()`).
    streaming : bool, default = False
        • False  ➜ return a (map-style) Dataset / DatasetDict built with `concatenate_datasets`.  
        • True   ➜ return an IterableDataset that loads one shard at a time and yields rows
                   sequentially (≈ constant RAM).

    Returns
    -------
    datasets.Dataset | datasets.DatasetDict | datasets.IterableDataset
    """
    shard_dirs = sorted(
        p for p in glob.glob(os.path.join(parent_dir, "shard_*")) if os.path.isdir(p)
    )
    if not shard_dirs:
        raise FileNotFoundError(f"No shard_* folders inside {parent_dir}")

    if not streaming:
        # ---------- 1) Simple path: load every shard → concatenate ----------
        datasets_or_dicts = [load_from_disk(p) for p in shard_dirs]              # :contentReference[oaicite:0]{index=0}

        # shards may be plain Dataset objects or DatasetDicts with several splits
        if isinstance(datasets_or_dicts[0], DatasetDict):
            merged = DatasetDict()
            for split in datasets_or_dicts[0].keys():                            # keep train/valid/test structure
                merged[split] = concatenate_datasets(
                    [ds[split] for ds in datasets_or_dicts]
                )                                                                # :contentReference[oaicite:1]{index=1}
            return merged
        else:
            return concatenate_datasets(datasets_or_dicts)                       # :contentReference[oaicite:2]{index=2}

    # ---------- 2) Streaming path: load only one shard at a time ----------
    def _generator():
        for p in shard_dirs:
            ds = load_from_disk(p, keep_in_memory=False)                         # memory maps Arrow; no RAM blow-up
            for row in ds:
                yield row

    return IterableDataset.from_generator(_generator)                            # behaves like any streamed split

def filter_single_band(datapoint, band='r'):
    if band == 'gr':
        if datapoint['bands_data']['g'] is None or datapoint['bands_data']['r'] is None:
            return False
    else:
        if datapoint['bands_data'][band] is None:
            return False
    return True

def filter_single_length_datapoint(datapoint):
    if len(datapoint['bands_data']['g']['target']) ==1 or len(datapoint['bands_data']['r']['target']) == 1:
        return False
    return True

if __name__ == "__main__":
    dataset = load_sharded_dataset("/scratch/wlk5936/ztf/ztf_bucketed_dataset_sharded/")
    dataset.save_to_disk("/scratch/wlk5936/ztf/ztf_bucketed_dataset")
