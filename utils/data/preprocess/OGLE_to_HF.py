from datasets import Dataset, Features, Value, Sequence
from tqdm import tqdm
import datasets

from utils.data.preprocess.read_OGLE import load_catalog, merge_remarks, merge_ident

if __name__ == "__main__":

    catalogs_to_process = [
        # region, parent_type, sub_type
        ("blg", "cep", "cep1O"),
    ]

    for catalog_to_process in catalogs_to_process:
        cat = load_catalog(*catalog_to_process)
        cat = merge_remarks(*catalog_to_process, cat)
        cat = merge_ident(*catalog_to_process, cat)

    # Read in lightcurves
    # Set up schema
    # Assosciate light curves with catalogs (by ID) and create HF entries
    # Create HF dataset
    # Write HF dataset to disk

    # When to concatenate catalogs?
