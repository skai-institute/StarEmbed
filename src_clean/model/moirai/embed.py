import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import datasets
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule, MoiraiFinetune
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
import matplotlib.pyplot as plt
from jaxtyping import Float, Bool
# from astropy.coordinates import SkyCoord
import datasets
from functools import partial
import argparse


class HFTimeSeriesDataset(Dataset):
    """Dataset adapter for HuggingFace time series data"""
    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        ctx: int = 64,
        horizon: int = 16
    ):
        self.dataset = hf_dataset
        self.ctx = ctx
        self.horizon = horizon
        self.item_ids = [item['item_id'] for item in self.dataset]

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        return {
            'item_id': np.array(item['source_id']),
            'target_series': torch.FloatTensor(item['bands_data'][-self.ctx-self.horizon:-self.horizon]),
            'covariates': torch.FloatTensor(item['past_feat_dynamic_real'][-self.ctx-self.horizon:-self.horizon]),
            'timestamps': pd.date_range(item['start'], periods=len(item['target']), freq='D').values[-self.ctx-self.horizon:]
        }
    
class HFTimeSeriesDatasetOutOfSample(Dataset):
    """Dataset adapter for HuggingFace time series data"""
    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        ctx: int = 64,
        horizon: int = 16,
        band: str = 'r'
    ):
        self.dataset = hf_dataset
        self.ctx = ctx
        self.horizon = horizon
        self.item_ids = [item['item_id'] for item in self.dataset]
        self.band = band

    def __len__(self) -> int:
        return len(self.dataset)


    def __getitem__(self, idx: int):
        item = self.dataset[idx]['bands_data'][self.band]
        item_id = self.dataset[idx]['item_id']
        start = self.dataset[idx]['start']
        target    = np.asarray(item["target"], dtype=float)
        target = (target - np.mean(target)) / np.std(target)
        covariate = np.asarray(item["past_feat_dynamic_real"], dtype=float)  # (T, F)
        # normalize covariate, which is the magnitude error
        covariate = (covariate - np.mean(covariate)) / np.std(covariate)

        seq_len  = len(target)
        pad_len  = max(0, self.ctx - seq_len)

        # -------- left‑pad target --------
        if pad_len:
            target = np.concatenate([np.zeros(pad_len, dtype=float), target])
        target = target[-self.ctx:]                     # keep exactly ctx

        # -------- left‑pad covariates --------
        if pad_len:
            pad_rows = np.zeros(pad_len, dtype=float)
            covariate = np.concatenate([pad_rows, covariate])
        covariate = covariate[-self.ctx:]               # (ctx, F)

        # -------- timestamps (extend left if padded) --------
        start_date = pd.to_datetime(start) - pd.Timedelta(days=pad_len)
        total_len  = self.ctx + self.horizon
        timestamps = pd.date_range(start_date, periods=total_len, freq="D").values

        return {
            "item_id"      : np.array(item_id),
            "target_series": torch.from_numpy(target.astype(np.float32)),
            "covariates"   : torch.from_numpy(covariate.astype(np.float32)),
            "timestamps"   : timestamps
        }

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack individual time series into batched tensors"""
    target_series = [item['target_series'] for item in batch]
    timestamps = [item['timestamps'] for item in batch]
    covariates = [item['covariates'] for item in batch]

    return {
        'item_ids': [item['item_id'] for item in batch],
        'target_series': torch.stack(target_series),
        'timestamps': timestamps,
        'covariates': torch.stack(covariates),
    }

from typing import List, Dict, Any
import pandas as pd                     # only needed to coerce the timestamp
from datasets import Dataset, Features, Sequence, Value

# -------------------------------------------------------------------------
# 1)  Describe the *target* layout once, so we can enforce it later
# -------------------------------------------------------------------------
target_features = Features({
    "item_id": Value("string"),
    "start":  Value("timestamp[ns]"),
    "freq":   Value("string"),
    "target": Sequence(Value("float32")),
    "past_feat_dynamic_real": Sequence(Value("float32")),
    "feat_dynamic_real":      Sequence(Value("float32")),
    "period": Value("float32"),
    "ps1_objid": Value("string"),
    "mjd":   Sequence(Value("float32")),
    "class": Value("string"),
    "csdr1_id": Value("string")
})


# -------------------------------------------------------------------------
# 2)  Conversion helper
# -------------------------------------------------------------------------
def convert_to_target_layout(src_ds: Dataset) -> Dataset:
    """
    Take an *arrow*‑backed HuggingFace `Dataset` (one row == one datapoint)
    and rebuild it as a list‑of‑dicts that exactly matches `target_features`.
    """
    records: List[Dict[str, Any]] = []

    for row in src_ds:                          # the dataset is iterable
        rec = {
            "item_id": str(row["item_id"]),
            # Cast to pandas.Timestamp to guarantee ns resolution
            "start":  pd.Timestamp(row["start"]),
            "freq":   row["freq"],              # already str
            "target": list(row["target"]),      # Arrow -> Python list
            "past_feat_dynamic_real": list(row["past_feat_dynamic_real"]),
            "feat_dynamic_real":      list(row["feat_dynamic_real"]),
            "period": float(row["period"]),
            "ps1_objid": str(row["ps1_objid"]),
            "mjd":   list(row["mjd"]),
            "class": str(row["class"]),
            "csdr1_id": str(row["csdr1_id"])
        }
        records.append(rec)

    # build the new HF dataset with the desired schema
    return Dataset.from_list(records, features=target_features)

def length_filter(example, min_length=160, band='r'):
    # Get the length of the specified field
    field_length = example['bands_data'][band]['length']
    
    # Check minimum length requirement
    if field_length < min_length:
        return False
        
    return True

def load_and_inference(
    hf_dataset: datasets.Dataset,
    checkpoint_path: str,
    ctx: int = 64,
    pdt: int = 64,
    psz: int = 32,
    batch_size: int = 128,
    num_samples: int = 100,
    out_of_sample: bool = False,
    band: str = 'r',
    model_name: str = 'moirai'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load HuggingFace dataset and run inference with Moirai model
    
    Args:
        dataset_path: Path to saved HuggingFace dataset
        checkpoint_path: Path to model checkpoint
        ctx: Context length
        pdt: Prediction length
        psz: Patch size
        batch_size: Batch size for inference
        num_samples: Number of samples for prediction
        
    Returns:
        Tuple of (predictions_mean, predictions_90th, predictions_10th) as DataFrames
    """
    
    if out_of_sample:
        dataset = HFTimeSeriesDatasetOutOfSample(
            hf_dataset=hf_dataset,
            ctx=ctx,
            horizon=pdt,
            band=band
        )
    else:
        dataset = HFTimeSeriesDataset(
            hf_dataset=hf_dataset,
            ctx=ctx,
            horizon=pdt
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        drop_last=False
    )
    
    print(f"Loading model from {checkpoint_path}")
    # Load model
    if checkpoint_path:
        model = MoiraiFinetune.load_from_checkpoint(checkpoint_path)
        model = MoiraiForecast(
            module=model.module,
            prediction_length=pdt,
            context_length=ctx,
            patch_size=psz,
            num_samples=num_samples,
            target_dim=1,
            past_feat_dynamic_real_dim=None,
            feat_dynamic_real_dim=None,
        )
    elif 'moe' in model_name:
        model = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-large"),
            prediction_length=pdt,
            context_length=ctx,
            patch_size=psz,
            num_samples=num_samples,
            target_dim=1,
            feat_dynamic_real_dim=None,
            past_feat_dynamic_real_dim=None,
        )
    else:
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(model_name),
            prediction_length=pdt,
            context_length=ctx,
            patch_size=psz,
            num_samples=num_samples,
            target_dim=1,
            past_feat_dynamic_real_dim=None,
            feat_dynamic_real_dim=None,
        )

    print(f"Model loaded successfully")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    
    # Run inference
    series_ids = []
    batched_predictions = []
    batched_predictions_90 = []
    batched_predictions_10 = []
    trn_timestamps = []
    batched_embeds = []
    
    with torch.no_grad():
        for batch in dataloader:
            item_ids = batch['item_ids']
            series_ids.extend(item_ids)

            target_series: Float[torch.Tensor, "batch past_time tgt"] = batch['target_series'].unsqueeze(-1).to(device)
            past_observed_target: Bool[torch.Tensor, "batch past_time tgt"] = torch.ones_like(target_series, dtype=torch.bool)
            past_is_pad: Bool[torch.Tensor, "batch past_time"] = torch.zeros_like(target_series, dtype=torch.bool).squeeze(-1)

            past_feat_dynamic_real: Float[torch.Tensor, "batch past_time past_feat"] = batch['covariates'].unsqueeze(-1).to(device)
            past_observed_feat_dynamic_real: Float[torch.Tensor, "batch past_time past_feat"] = torch.ones_like(past_feat_dynamic_real, dtype=torch.float32)
            if past_feat_dynamic_real.shape[2] == 0:
                past_feat_dynamic_real = None
                past_observed_feat_dynamic_real = None

            feat_dynamic_real = None
            observed_feat_dynamic_real = None
            embeds = model.embed(
                past_target=target_series,
                past_observed_target=past_observed_target,
                past_is_pad=past_is_pad,
                past_feat_dynamic_real=past_feat_dynamic_real,
                past_observed_feat_dynamic_real=past_observed_feat_dynamic_real,
                feat_dynamic_real=feat_dynamic_real,
                observed_feat_dynamic_real=observed_feat_dynamic_real,
            )
            
            batched_embeds.append(embeds)

    batched_embeds = torch.cat(batched_embeds, dim=0)
    
    return batched_embeds, series_ids

# Create a function to process each example
def add_embeddings(batched_embeds_g, series_ids_g, batched_embeds_r, series_ids_r, example, idx):
    
    # Get embeddings
    embeddings_g = batched_embeds_g[idx]
    embeddings_r = batched_embeds_r[idx]
    series_id_g = series_ids_g[idx]
    series_id_r = series_ids_r[idx]

    assert example["item_id"] == series_id_g == series_id_r
    # Add embeddings to the example (convert tensor to numpy for storage)
    example["embeddings_g"] = embeddings_g.to(torch.float32).cpu().numpy()
    example["embeddings_r"] = embeddings_r.to(torch.float32).cpu().numpy()
    
    return example

def filter_single_band(datapoint, band='r'):
    if band == 'gr':
        if datapoint['bands_data']['g'] is None or datapoint['bands_data']['r'] is None:
            return False
    else:
        if datapoint['bands_data'][band] is None:
            return False
    return True



# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx", type=int, default=64)
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--model", type=str, default='moirai')
    parser.add_argument("--dataset_path", type=str, required=True, default='/projects/p32795/hongyu/hf_csdr1_multiband_raw_lc_minority_class_str_v2')
    args = parser.parse_args()
    ctx = args.ctx
    pdt = 64
    psz = 16
    # d_model: small: 384, base: 768, large: 1024
    if args.model == 'moirai':
        model_name = "Salesforce/moirai-1.1-R-small"
    elif args.model == 'moe':
        model_name = "Salesforce/moirai-moe-1.0-R-small"
    else:
        raise ValueError(f"Invalid model: {args.model}")


    dataset_path = args.dataset_path
    print(f"Loading dataset from {dataset_path}")   
    # Load dataset
    hf_dataset = datasets.load_from_disk(dataset_path)
    # hf_dataset = convert_to_target_layout(hf_dataset)
    print(f"total number of training lc: {len(hf_dataset['train'])}")
    print(f"total number of validation lc: {len(hf_dataset['validation'])}")
    print(f"total number of test lc: {len(hf_dataset['test'])}")

    dataset_dict = {}

    for split in ['train', 'validation', 'test']:
        hf_dataset_split = hf_dataset[split]
        band = 'gr'
        hf_dataset_split = hf_dataset_split.filter(lambda x: filter_single_band(x, band))
        batched_embeds_g, series_ids_g = load_and_inference(
            hf_dataset=hf_dataset_split,
            checkpoint_path=None,
            ctx=ctx,
            pdt=pdt,
            psz=psz,
            out_of_sample=True,
            band='g',
            model_name=model_name
        )
        batched_embeds_r, series_ids_r = load_and_inference(
            hf_dataset=hf_dataset_split,
            checkpoint_path=None,
            ctx=ctx,
            pdt=pdt,
            psz=psz,
            out_of_sample=True,
            band='r',
            model_name=model_name
        )


        dataset = hf_dataset_split

        # Process the dataset with progress bar
        print("Generating embeddings for all items...")
        updated_dataset = dataset.map(partial(add_embeddings, batched_embeds_g, series_ids_g, batched_embeds_r, series_ids_r), desc="Generating embeddings", with_indices=True)
        dataset_dict[split] = updated_dataset
    # Save the updated dataset
    # output_path = f"csdr1_raw_embs_moiral_small_trn_val_tst_ctx{ctx}_pdt{pdt}_psz{psz}_band{band}"
    output_path = f"/projects/p32795/hongyu/csdr1_minority_raw_embs_moiral_small_trn_val_tst_ctx{ctx}_pdt{pdt}_psz{psz}_band{band}"
    print(f"Saving dataset with embeddings to {output_path}")
    dataset_to_save = datasets.DatasetDict(dataset_dict)
    dataset_to_save.save_to_disk(output_path)

    print("Done!")
