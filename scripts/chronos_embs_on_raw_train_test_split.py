import pandas as pd
import torch
from chronos import BaseChronosPipeline
import datasets
from tqdm import tqdm
import numpy as np
from functools import partial
import argparse




def length_filter(example, min_length=160):
    # Get the length of the specified field
    field_length = len(example['target'])
    
    # Check minimum length requirement
    if field_length < min_length:
        return False
        
    return True

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

# Create a function to process each example
def add_embeddings(example, ctx=64, band='r'):
    if band == 'gr':
        bands = ['g', 'r']
    else:
        bands = [band]
    for band in bands:
        # Normalize the data
        target = np.array(example['bands_data'][band]['target'])
        mu = np.mean(target)
        sigma = np.std(target)
        normalized_target = (target - mu) / sigma


        seq_len  = len(target)
        pad_len  = max(0, ctx - seq_len)

        # -------- left‑pad target --------
        if pad_len:
            normalized_target = np.concatenate([np.zeros(pad_len, dtype=float), normalized_target])
        normalized_target = normalized_target[-ctx:]                     # keep exactly ctx
        
        # Convert to tensor for model input
        context = torch.tensor(normalized_target)
        
        # Generate embeddings
        embeddings, tokenizer_state = pipeline.embed(context)
        
        # Add embeddings to the example (convert tensor to numpy for storage)
        example[f"embeddings_{band}"] = embeddings.to(torch.float32).cpu().numpy()
    
    return example

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx", type=int, default=64)
    parser.add_argument("--model", type=str, default='t5')
    args = parser.parse_args()
    # d_model: tiny models: 256, mini models: 384, bolt_small: 512, bolt_base: 768, small: 512, base: 768, large: 1024
    if args.model == 't5':
        model_name = "amazon/chronos-t5-tiny"
    elif args.model == 'bolt':
        model_name = "amazon/chronos-bolt-tiny"
    else:
        raise ValueError(f"Invalid model: {args.model}")

    # Load model
    pipeline = BaseChronosPipeline.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )

    # Load dataset
    dataset = datasets.load_from_disk("/projects/p32795/hongyu/hf_csdr1_multiband_raw_lc_subclass_class_str_v2")
    print(f"total number of training lc: {len(dataset['train'])}")
    print(f"total number of validation lc: {len(dataset['validation'])}")
    print(f"total number of test lc: {len(dataset['test'])}")

    dataset_dict = {}
    for split in ['train', 'validation', 'test']:
        dataset_split = dataset[split]
        # dataset_split = dataset_split.select(range(5)) # just for testing
        # dataset = dataset.filter(length_filter)


        # Process the dataset with progress bar
        print(f"Generating embeddings for {split}...")
        ctx = args.ctx

        band = 'gr'
        dataset_split = dataset_split.filter(lambda x: filter_single_band(x, band))
        updated_dataset = dataset_split.map(partial(add_embeddings, ctx=ctx, band=band), desc="Generating embeddings")
        dataset_dict[split] = updated_dataset

    # Save the updated dataset
    dataset_to_save = datasets.DatasetDict(dataset_dict)
    output_path = f"/projects/p32795/weijian/embs/csdr1_raw4_catflags_filtered_embs_chronos_{args.model}_tiny_trn_val_tst_ctx{ctx}_band{band}"
    print(f"Saving dataset with embeddings to {output_path}")
    dataset_to_save.save_to_disk(output_path)

    print("Done!")