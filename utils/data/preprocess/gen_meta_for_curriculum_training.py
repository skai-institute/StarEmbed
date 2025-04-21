import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import pytorch_lightning as pl
from datasets import load_dataset
from typing import Dict, List, Optional, Tuple
from pytorch_lightning.callbacks import Callback
import datasets

"""
This script is to generate the metadata for the curriculum training to fine-tune the universal forecaster.

There are two types of metadata:
1. The average magnitude of the time series. It is used to decide the curriculum phases. We start from brighter light curves and gradually include fainter ones.
2. The average measurement error of the light curve. The inverse of the measurement error is used as the sampling weight in data loader.

The metadata is calculated for all the time series in the dataset.
"""


class TimeSeriesMetadataCalculator:
    """Calculate metadata for time series data once and store it"""
    
    def __init__(self, dataset: datasets.Dataset, 
                 magnitude_bins: List[Tuple[float, float]], 
                 error_weight_factor: float):
        self.dataset = dataset
        self.magnitude_bins = magnitude_bins
        self.error_weight_factor = error_weight_factor
        self.num_bins = len(magnitude_bins)
        
        # Initialize data structures
        self.metadata = {}
        self.bin_indices = [[] for _ in range(self.num_bins)]
        self.bin_weights = [[] for _ in range(self.num_bins)]
    
    def calculate_metadata(self):
        """Calculate all metadata at once"""
        print("Calculating dataset metadata...")
        
        # Step 1: Extract metadata for all samples
        for idx in range(len(self.dataset)):
            # Get the time series data and its error estimates
            ts_data = self.dataset[idx]['mag']  # Adjust field name as needed
            ts_errors = self.dataset[idx]['magerr']  # Adjust field name as needed
            
            # Calculate average magnitude
            avg_magnitude = np.mean(np.abs(ts_data))
            
            # Calculate average measurement error
            avg_error = np.mean(ts_errors)
            
            # Store metadata
            self.metadata[idx] = {
                'avg_magnitude': avg_magnitude,
                'avg_error': avg_error,
                'idx': idx
            }
            
            # Determine which bin this sample belongs to
            for bin_idx, (min_mag, max_mag) in enumerate(self.magnitude_bins):
                if min_mag <= avg_magnitude < max_mag:
                    self.bin_indices[bin_idx].append(idx)
                    break
        
        # Step 2: Calculate weights based on measurement error for each bin
        for bin_idx in range(self.num_bins):
            indices = self.bin_indices[bin_idx]
            if not indices:
                continue
                
            # Get average errors for each sample in this bin
            errors = [self.metadata[idx]['avg_error'] for idx in indices]
            
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
                weights = [1.0] * len(indices)
            
            self.bin_weights[bin_idx] = weights
        
        print("Metadata calculation complete.")
        return self.metadata, self.bin_indices, self.bin_weights


class CurriculumSampler(Sampler):
    """Custom sampler that implements curriculum learning with pre-calculated bins and weights"""
    
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
        self.curriculum_phase = phase
    
    def __iter__(self):
        """Return iterator over indices based on current curriculum phase"""
        # Get indices and weights for current phase
        indices = self.bin_indices[self.curriculum_phase]
        weights = self.bin_weights[self.curriculum_phase]
        
        if not indices:
            # Fallback if bin is empty
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
                 error_weight_factor: float = 2.0):
        """
        Args:
            dataset_name: Name of HuggingFace dataset
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes for dataloaders
            magnitude_bins: Magnitude ranges for curriculum phases
            error_weight_factor: How much to weight measurement error in sampling
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.magnitude_bins = magnitude_bins
        self.error_weight_factor = error_weight_factor
        
        self.sampler = None
    
    def setup(self, stage=None):
        """Load dataset and calculate metadata"""
        # Load your HuggingFace dataset
        self.dataset = load_dataset(self.dataset_name, split="train")
        
        # Optional: create separate validation set if needed
        # self.val_dataset = load_dataset(self.dataset_name, split="validation")
        
        # Calculate metadata, bin indices, and weights
        calculator = TimeSeriesMetadataCalculator(
            self.dataset, 
            self.magnitude_bins, 
            self.error_weight_factor
        )
        _, self.bin_indices, self.bin_weights = calculator.calculate_metadata()
        
        # Create the sampler
        self.sampler = CurriculumSampler(
            self.bin_indices,
            self.bin_weights,
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
        # You might want to use a separate validation dataset
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


class TimeSeriesModel(pl.LightningModule):
    """Example PyTorch Lightning model for time series data"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        
        # Example model architecture (adjust based on your needs)
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take the output from the last time step
        output = self.fc(lstm_out[:, -1, :])
        return output
    
    def training_step(self, batch, batch_idx):
        # Process batch according to your model's requirements
        x = batch['values'].float()  # Adjust field name as needed
        y = batch['targets'].float()  # Adjust field name as needed
        
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['values'].float()
        y = batch['targets'].float()
        
        y_hat = self(x)
        val_loss = torch.nn.functional.mse_loss(y_hat, y)
        
        self.log('val_loss', val_loss)
        return val_loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# Example usage
def main():
    # Create data module
    data_module = TimeSeriesDataModule(
        dataset_name="your_dataset_name",
        batch_size=32,
        num_workers=4,
        magnitude_bins=[(13, 15.7), (15.7, 18.3), (18.3, 21)],
        error_weight_factor=2.0
    )
    
    # Set up data module (calculates metadata and creates sampler)
    data_module.setup()
    
    # Create model
    model = TimeSeriesModel(
        input_dim=1,  # Adjust based on your data
        hidden_dim=64,
        output_dim=1  # Adjust based on your task
    )
    
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