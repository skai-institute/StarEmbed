from datasets import Dataset, Features, Value, Sequence
from tqdm import tqdm
import datasets
import numpy as np
import pandas as pd

from utils.data.preprocess.read_OGLE_catalogs import read_OGLE_catalogs, merge_remarks, merge_IDs

if __name__ == "__main__":

    catalogs_to_process = [
        # region, parent_type, sub_type
        ("blg", "cep", "1O"),
    ]

    for catalog_to_process in catalogs_to_process:
        cat = read_OGLE_catalogs(*catalog_to_process)
        cat = merge_remarks(*catalog_to_process, cat)
        cat = merge_IDs(*catalog_to_process, cat)

    # Read in lightcurves
    # Set up schema
    # Assosciate light curves with catalogs (by ID) and create HF entries
    # Create HF dataset
    # Write HF dataset to disk

    # When to concatenate catalogs?
