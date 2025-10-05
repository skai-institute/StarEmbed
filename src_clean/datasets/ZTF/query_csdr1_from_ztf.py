from typing import List, Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from dask.distributed import Client
from lsdb import Catalog
import numpy.typing as npt
from dataclasses import dataclass
import os
from jaxtyping import Float, Bool
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from pytorch_lightning import LightningModule as L

# have to use lsdb==0.4.2, hats==0.4.3
from lsdb import read_hats
import lsdb
from astropy.coordinates import SkyCoord
from astropy import units as u

from sklearn.preprocessing import MinMaxScaler
import pickle


def analyze_multiple_stars_batched(
    raw_catalog: Catalog,
    varstars: pd.DataFrame,
    n_stars: int = 20,
    seed: int = 42,
    use_cache: bool = True
) -> None:
    """
    Analyze multiple stars using batched processing for efficient forecasting.
    
    Args:
        raw_catalog: ZTF catalog
        varstars: DataFrame containing variable stars information
        n_stars: Number of stars to analyze
        seed: Random seed for reproducibility
        
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    random_indices = list(range(len(varstars)))
    print(f"random_indices: {random_indices}")
    selected_stars = varstars.iloc[random_indices]
    print(f"selected_stars: {selected_stars[['ID', 'RAdeg', 'Decdeg']]}")
    cache_path = f"all_objects_47054_None2.pkl"
    with Client(n_workers=20, memory_limit="64GB") as client:
        all_objects = [
            (str(i),
            raw_catalog.cone_search(
                star['RAdeg'],
                star['Decdeg'],
                2,  # 2 arcsecond radius
            ).compute()
            )
            for i, star in selected_stars.iterrows()
        ]

    # cache the all_objects if not exists
    if not os.path.exists(cache_path):
        with open(cache_path, 'wb') as f:
            pickle.dump(all_objects, f)
    
    return None

if __name__ == "__main__":
    ZTF_SOURCES = "https://data.lsdb.io/hats/ztf_dr14/ztf_source"
    
    # Load ZTF catalog
    raw_catalog = read_hats(ZTF_SOURCES)
    
    # Load and process variable stars data
    column_names = ['ID', 'RAh', 'RAm', 'RAs', 'Decsign', 'DEm', 'DEs', 
                   'magV', 'P', "Amp", "class", 'flag']
    """
            ID  RAh  RAm    RAs  Decsign  DEm   DEs   magV         P   Amp class flag
    0  CSS_J000031.5-084652    0    0  31.50       -8   46  52.3  14.14  0.404185  0.12     1  NaN
    1  CSS_J000036.9+412805    0    0  36.94       41   28   5.7  17.39  0.274627  0.73     1  NaN
    """
    varstars = pd.read_csv(
        "CSDR1_varstars.txt",
        header=33,
        sep='\s+',
        names=column_names
    )
    
    # Convert coordinates
    RAhms = [f"{row.RAh}h{row.RAm}m{row.RAs}s" for _, row in varstars.iterrows()]
    Decdms = [f"{row.Decsign}d{row.DEm}m{row.DEs}s" for _, row in varstars.iterrows()]
    coords = SkyCoord(ra=RAhms, dec=Decdms, frame='icrs')
    varstars['RAdeg'] = coords.ra.deg
    varstars['Decdeg'] = coords.dec.deg
    
    # Run analysis
    metrics = analyze_multiple_stars_batched(raw_catalog, varstars, n_stars=20, use_cache=True, seed=50)