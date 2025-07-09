"""
This script is to generate the re-ordered indices for data loading order on the cross-matched ztf dataset based on the metadata for the curriculum training to fine-tune the universal forecaster.

There are two types of metadata:
1. The average magnitude of the time series. It is used to decide the curriculum phases. We start from brighter light curves and gradually include fainter ones.
2. The average measurement error of the light curve. The inverse of the measurement error is used as the sampling weight in data loader.

The metadata is pre-calculated and saved together with the cross-matched ztf dataset.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import pytorch_lightning as pl
import datasets
from datasets import load_dataset
from typing import Dict, List, Optional, Tuple
from pytorch_lightning.callbacks import Callback
import os
import json
import time
import glob 
from tqdm import tqdm
from aurora.common.env import env

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



class CurriculumSamplerPreprocessor:
    """Preprocesses dataset to calculate curriculum binning and weighting once and save to disk"""
    
    def __init__(self, 
                 dataset: datasets.Dataset,
                 magnitude_bins: List[Tuple[float, float]] = [(13, 15.7), (15.7, 18.3), (18.3, 21)],
                 error_weight_factor: float = 2.0,
                 cache_dir: str = env.CURRICULUM_CACHE_PATH):
        """
        Args:
            dataset: HuggingFace dataset with pre-computed avg_mag and avg_magerr
            magnitude_bins: Magnitude ranges for curriculum phases
            error_weight_factor: How much to weight measurement error in sampling
            cache_dir: Directory to store preprocessed binning data
        """
        self.dataset = dataset
        self.magnitude_bins = magnitude_bins
        self.error_weight_factor = error_weight_factor
        self.num_samples = len(dataset)
        self.num_bins = len(magnitude_bins)
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_cache_filename(self):
        """Generate a unique cache filename based on dataset and parameters"""
        # Create a hash of dataset name and parameters
        dataset_name = getattr(self.dataset, "_name", "dataset")
        dataset_config = getattr(self.dataset, "_config_name", "")
        
        # Format magnitude bins and weight factor for the filename
        bins_str = "_".join([f"{min_mag:.1f}-{max_mag:.1f}" for min_mag, max_mag in self.magnitude_bins])
        
        filename = f"{dataset_name}_{dataset_config}_bins_{bins_str}_weight_{self.error_weight_factor:.1f}.json"
        return os.path.join(self.cache_dir, filename)
    
    def preprocess(self, force_recompute=False):
        """Compute bin assignments and weights, with caching"""
        cache_file = self.get_cache_filename()
        
        # Check if cache file exists and load if it does
        if os.path.exists(cache_file) and not force_recompute:
            print(f"Loading pre-computed curriculum binning from {cache_file}")
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                
                # Convert string keys back to integers for bin_indices
                bin_indices = []
                for bin_idx in range(self.num_bins):
                    bin_indices.append([int(idx) for idx in cached_data['bin_indices'][bin_idx]])
                
                bin_weights = cached_data['bin_weights']
                return bin_indices, bin_weights
        
        # If no cache or forced recompute, calculate from scratch
        print("Computing curriculum binning from scratch...")
        start_time = time.time()
        
        # Initialize data structures
        bin_indices = [[] for _ in range(self.num_bins)]
        bin_weights = [[] for _ in range(self.num_bins)]
        
        # Temporary storage for errors by bin
        bin_errors = [[] for _ in range(self.num_bins)]
        
        # First pass: assign samples to bins based on magnitude
        print("First pass")
        for idx in tqdm(range(self.num_samples)):
            # Get pre-computed metrics from the dataset
            avg_mag = self.dataset[idx]['avg_mag']
            avg_magerr = self.dataset[idx]['avg_magerr']
            
            # Assign to bin based on magnitude
            for bin_idx, (min_mag, max_mag) in enumerate(self.magnitude_bins):
                if min_mag <= avg_mag < max_mag:
                    bin_indices[bin_idx].append(idx)
                    bin_errors[bin_idx].append(avg_magerr)
                    break

        print("First pass done")
        
        # Second pass: compute weights based on measurement error for each bin
        for bin_idx in range(self.num_bins):
            errors = bin_errors[bin_idx]
            if not errors:
                continue
                
            # Convert errors to weights (lower error = higher weight)
            max_error = max(errors) if errors else 1.0
            min_error = min(errors) if errors else 0.0
            error_range = max_error - min_error
            
            if error_range > 0:
                # Normalize errors to [0, 1] and invert so lower errors get higher weights
                normalized_weights = [1 - ((err - min_error) / error_range) for err in errors]
                
                # Apply error weight factor (higher factor = more bias toward low error samples)
                weights = [w ** self.error_weight_factor for w in normalized_weights]
            else:
                weights = [1.0] * len(errors)
            
            bin_weights[bin_idx] = weights
        
        elapsed_time = time.time() - start_time
        print(f"Binning complete in {elapsed_time:.2f} seconds. Samples per bin: {[len(indices) for indices in bin_indices]}")
        
        # Save to cache
        cached_data = {
            'bin_indices': bin_indices,
            'bin_weights': bin_weights,
            'metadata': {
                'dataset_size': self.num_samples,
                'magnitude_bins': self.magnitude_bins,
                'error_weight_factor': self.error_weight_factor,
                'computation_time': elapsed_time
            }
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cached_data, f)
        
        print(f"Saved curriculum binning to {cache_file}")
        return bin_indices, bin_weights


class CurriculumSampler(Sampler):
    """Custom sampler that implements curriculum learning with pre-computed bins and weights"""
    
    def __init__(self, 
                 bin_indices: List[List[int]], 
                 bin_weights: List[List[float]],
                 num_samples: int,
                 curriculum_phase: int = 0):
        """
        Args:
            bin_indices: Pre-calculated indices for each magnitude bin
            bin_weights: Pre-calculated weights for each sample in each bin
            num_samples: Total number of samples in dataset
            curriculum_phase: Starting curriculum phase (0, 1, or 2)
        """
        self.bin_indices = bin_indices
        self.bin_weights = bin_weights
        self.num_samples = num_samples
        self.num_bins = len(bin_indices)
        self.curriculum_phase = curriculum_phase
    
    def set_curriculum_phase(self, phase):
        """Update the curriculum phase"""
        assert 0 <= phase < self.num_bins, f"Phase must be between 0 and {self.num_bins-1}"
        print(f"Setting curriculum phase from {self.curriculum_phase} to {phase}")
        self.curriculum_phase = phase
    
    def __iter__(self):
        """Return iterator over indices based on current curriculum phase"""
        # Get indices and weights for current phase
        indices = self.bin_indices[self.curriculum_phase]
        weights = self.bin_weights[self.curriculum_phase]
        
        if not indices:
            # Fallback if bin is empty
            print(f"Warning: Bin {self.curriculum_phase} is empty. Using all samples.")
            indices = list(range(self.num_samples))
            weights = None
        
        # Convert to numpy array for efficient sampling
        indices_array = np.array(indices)
        
        # Create a generator to yield indices
        total_samples = len(indices)
        
        if weights and sum(weights) > 0:
            # Normalize weights
            normalized_weights = np.array(weights) / sum(weights)
            
            sampled_indices = np.random.choice(
                indices_array, 
                size=total_samples, 
                replace=True, 
                p=normalized_weights
            )
        else:
            sampled_indices = np.random.choice(
                indices_array, 
                size=total_samples, 
                replace=True
            )
        
        for idx in sampled_indices:
            yield int(idx)
    
    def __len__(self):
        """Return the number of samples in the current phase"""
        return len(self.bin_indices[self.curriculum_phase]) or self.num_samples


class CurriculumCallback(Callback):
    """PyTorch Lightning callback to manage curriculum phases"""
    
    def __init__(self, 
                 sampler, 
                 phase_epochs=[5, 10, 15]):
        """
        Args:
            sampler: The CurriculumSampler instance
            phase_epochs: List of epoch counts for each phase
        """
        self.sampler = sampler
        self.phase_epochs = phase_epochs
        self.current_phase = 0
        self.phase_boundaries = self._calculate_phase_boundaries()
    
    def _calculate_phase_boundaries(self):
        """Calculate cumulative epoch counts for phase transitions"""
        boundaries = []
        total = 0
        for epochs in self.phase_epochs:
            total += epochs
            boundaries.append(total)
        return boundaries
    
    def on_epoch_start(self, trainer, pl_module):
        """Check if we need to transition to next phase at start of epoch"""
        current_epoch = trainer.current_epoch
        
        # Determine which phase we should be in
        new_phase = 0
        for i, boundary in enumerate(self.phase_boundaries[:-1]):
            if current_epoch >= boundary:
                new_phase = i + 1
        
        # Update phase if needed
        if new_phase != self.current_phase:
            self.current_phase = new_phase
            self.sampler.set_curriculum_phase(new_phase)
            print(f"Transitioning to curriculum phase {new_phase}")


class TimeSeriesDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule with curriculum learning capability"""
    
    def __init__(self, 
                 dataset_name: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 magnitude_bins: List[Tuple[float, float]] = [(13, 15.7), (15.7, 18.3), (18.3, 21)],
                 error_weight_factor: float = 2.0,
                 cache_dir: str = "/projects/p32795/weijian/exp_cache/curriculum_cache",
                 force_recompute: bool = False):
        """
        Args:
            dataset_name: Name of HuggingFace dataset
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes for dataloaders
            magnitude_bins: Magnitude ranges for curriculum phases
            error_weight_factor: How much to weight measurement error in sampling
            cache_dir: Directory to store preprocessed binning data
            force_recompute: Force recomputation of curriculum binning even if cache exists
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.magnitude_bins = magnitude_bins
        self.error_weight_factor = error_weight_factor
        self.cache_dir = cache_dir
        self.force_recompute = force_recompute
        
        self.sampler = None
    
    def setup(self, stage=None):
        """Load dataset and create sampler with cached pre-computation"""
        # Load your HuggingFace dataset
        self.dataset = load_dataset(self.dataset_name, split="train")
        
        # Optional: create separate validation set if needed
        # self.val_dataset = load_dataset(self.dataset_name, split="validation")
        
        # Create the preprocessor and get cached or computed binning
        preprocessor = CurriculumSamplerPreprocessor(
            self.dataset,
            magnitude_bins=self.magnitude_bins,
            error_weight_factor=self.error_weight_factor,
            cache_dir=self.cache_dir
        )
        
        # Get bin indices and weights (from cache if available, or compute and cache)
        bin_indices, bin_weights = preprocessor.preprocess(force_recompute=self.force_recompute)
        
        # Create the sampler with pre-computed bins and weights
        self.sampler = CurriculumSampler(
            bin_indices,
            bin_weights,
            len(self.dataset),
            curriculum_phase=0  # Start with phase 0
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,  # Use our custom sampler
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        # For validation, we don't need curriculum sampling
        return DataLoader(
            self.dataset,  # or self.val_dataset if you have a separate validation set
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_curriculum_callback(self, phase_epochs=[5, 10, 15]):
        """Get a callback to manage curriculum phases"""
        return CurriculumCallback(self.sampler, phase_epochs)


# Example usage with PyTorch Lightning
def main():
    # Create data module with caching support
    data_module = TimeSeriesDataModule(
        dataset_name="your_dataset_name",
        batch_size=32,
        num_workers=4,
        magnitude_bins=[(13, 15.7), (15.7, 18.3), (18.3, 21)],
        error_weight_factor=2.0,
        cache_dir="./curriculum_cache",  # Directory to store/load cached binning
        force_recompute=False  # Set to True to force recomputation of binning
    )
    
    # Set up data module (will use cached binning if available)
    data_module.setup()
    
    # Create your Lightning model
    model = YourLightningModel()
    
    # Create curriculum callback
    curriculum_callback = data_module.get_curriculum_callback(
        phase_epochs=[5, 10, 15]  # Epochs per phase
    )
    
    # Create trainer with curriculum callback
    trainer = pl.Trainer(
        max_epochs=30,  # Total epochs across all phases
        callbacks=[curriculum_callback],
        accelerator='gpu',  # Use 'cpu' if no GPU available
        devices=1
    )
    
    # Train the model
    trainer.fit(model, data_module)

# Direct usage without Lightning DataModule
def standalone_usage():
    # Load dataset
    

    dataset_parent_path = "/scratch/wlk5936/ztf/ztf_bucketed_dataset/"
    # dataset_paths = glob.glob(os.path.join(dataset_parent_path, "shard_*"))

    dataset = load_sharded_dataset(dataset_parent_path)

    
    # Create preprocessor
    preprocessor = CurriculumSamplerPreprocessor(
        dataset,
        magnitude_bins=[(12, 15.7), (15.7, 18.3), (18.3, 22)],
        error_weight_factor=1.0,
        cache_dir=env.CURRICULUM_CACHE_PATH
    )
    
    # Get bin indices and weights (from cache if available)
    bin_indices, bin_weights = preprocessor.preprocess(force_recompute=False)
    
    # Create sampler
    sampler = CurriculumSampler(
        bin_indices,
        bin_weights,
        len(dataset),
        curriculum_phase=0
    )
    
    # Create callback
    curriculum_callback = CurriculumCallback(
        sampler, 
        phase_epochs=[5, 10, 15]
    )
    
    # Use in your own training loop or with Lightning
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # ... rest of your training code

if __name__ == "__main__":
    # main()
    standalone_usage()
