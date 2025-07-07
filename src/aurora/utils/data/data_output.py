from datasets import Dataset, Features, Value, Sequence
from typing import List, Dict
import os
import shutil
import sys
import inspect
from pathlib import Path

def output_hf_dataset(dataset_records: List[Dict]):
    """
    Convert the processed records to a Hugging Face dataset

    Used by: gaussian process code to generate gp_light_curves_dataset_full_None that is used to generate embeddings
    """
    features = Features({
        "item_id": Value("string"),
        "start": Value("timestamp[ns]"),
        "freq": Value("string"),
        "target": Sequence(Value("float32")),
        "past_feat_dynamic_real": Sequence(Value("float32")),
        "feat_dynamic_real": Sequence(Value("float32")),
        "period": Value("float32"),
        "ps1_objid": Value("string"),
        "mjd": Sequence(Value("float32")),
        "class": Value("string"),
        "csdr1_id": Value("string")
    })
    
    dataset = Dataset.from_list(dataset_records, features=features)
    return dataset


def save_dataset_with_script(
    dataset, 
    dataset_path, 
    num_shards=None, 
    num_proc=None, 
    caller_file=None
):
    """
    Save a dataset to disk along with a copy of the calling script.
    
    Parameters:
    -----------
    dataset : Dataset
        The dataset to save (must have save_to_disk method).
    dataset_path : str or Path
        Path where to save the dataset.
    num_shards : int, optional
        Number of shards to split the dataset into.
    num_proc : int, optional
        Number of processes to use for saving.
    caller_file : str, optional
        Path to the calling script. If None, it will be automatically determined.
    
    Returns:
    --------
    Path
        The path where the dataset was saved.
    """
    # Convert to Path object
    dataset_path = Path(dataset_path)
    
    # Create the directory if it doesn't exist
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Save the dataset
    save_kwargs = {}
    if num_shards is not None:
        save_kwargs['num_shards'] = num_shards
    if num_proc is not None:
        save_kwargs['num_proc'] = num_proc
    
    dataset.save_to_disk(dataset_path=dataset_path, **save_kwargs)
    
    # Determine the caller script if not provided
    if caller_file is None:
        # Get the frame of the caller
        frame = inspect.stack()[1]
        caller_file = frame.filename
    
    # Get the path of the calling script
    caller_script_path = Path(caller_file)
    script_backup_path = dataset_path / caller_script_path.name
    
    # Copy the script
    shutil.copy2(caller_script_path, script_backup_path)
    
    print(f"Dataset saved to {dataset_path}")
    print(f"Script backup saved to {script_backup_path}")
    
    return dataset_path