import torch
from torch.utils.data import Sampler, DistributedSampler
import numpy as np
import math
from typing import List, Optional, Iterator

class DistributedCurriculumSampler(Sampler):
    """
    Combines functionality of DistributedSampler with curriculum sampling.
    
    This sampler ensures:
    1. Data is distributed across different processes/GPUs
    2. Each process gets a unique subset of the data
    3. Within each process, data is sampled according to curriculum phases
    """
    
    def __init__(self,
                 bin_indices: List[List[int]],
                 bin_weights: List[List[float]],
                 num_samples: int,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 0,
                 curriculum_phase: int = 0,
                 drop_last: bool = False):
        """
        Args:
            bin_indices: Pre-calculated indices for each magnitude bin
            bin_weights: Pre-calculated weights for each sample in each bin
            num_samples: Total number of samples in dataset
            num_replicas: Number of distributed processes
            rank: Rank of the current process
            shuffle: Whether to shuffle the indices
            seed: Random seed for reproducibility
            curriculum_phase: Starting curriculum phase (0, 1, or 2)
            drop_last: Whether to drop the last batch if it's incomplete
        """
        # Initialize distributed parameters
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Distributed package is not available, use regular sampler instead")
            num_replicas = torch.distributed.get_world_size()
            
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Distributed package is not available, use regular sampler instead")
            rank = torch.distributed.get_rank()
            
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas-1}]")
            
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        
        # Initialize curriculum parameters
        self.bin_indices = bin_indices
        self.bin_weights = bin_weights
        self.num_samples = num_samples
        self.num_bins = len(bin_indices)
        self.curriculum_phase = curriculum_phase
        
        # Validate curriculum phase
        if curriculum_phase < 0 or curriculum_phase >= self.num_bins:
            raise ValueError(f"Invalid curriculum phase {curriculum_phase}, must be between 0 and {self.num_bins-1}")
            
        # Calculate the number of samples per process
        self.num_samples_per_replica = self._get_num_samples_per_replica()
            
    def _get_num_samples_per_replica(self):
        """Calculate how many samples each process should handle"""
        indices = self.bin_indices[self.curriculum_phase]
        
        if not indices:
            # If bin is empty, use all samples
            indices = list(range(self.num_samples))
        
        total_size = len(indices)
        
        if self.drop_last:
            # Divide total size by num_replicas and drop remainder
            return math.floor(total_size / self.num_replicas)
        else:
            # Divide total size by num_replicas and round up
            return math.ceil(total_size / self.num_replicas)
    
    def set_curriculum_phase(self, phase):
        """Update the curriculum phase"""
        if phase < 0 or phase >= self.num_bins:
            raise ValueError(f"Invalid curriculum phase {phase}, must be between 0 and {self.num_bins-1}")
        
        print(f"Setting curriculum phase from {self.curriculum_phase} to {phase}")
        self.curriculum_phase = phase
        
        # Recalculate samples per replica since bin size may have changed
        self.num_samples_per_replica = self._get_num_samples_per_replica()
        
    def set_epoch(self, epoch):
        """Set the epoch for this sampler (for reproducibility)"""
        self.epoch = epoch
    
    def __iter__(self) -> Iterator[int]:
        """Return iterator over indices for this process with curriculum sampling"""
        # Get indices for current curriculum phase
        indices = self.bin_indices[self.curriculum_phase]
        weights = self.bin_weights[self.curriculum_phase]
        
        if not indices:
            # If bin is empty, use all samples
            indices = list(range(self.num_samples))
            weights = None
        
        # Get total number of samples in this bin
        total_size = len(indices)
        
        # Set up random state for this epoch
        if self.shuffle:
            # Use a different seed for each epoch, process, and curriculum phase
            # to ensure different shuffling across epochs and processes
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch + self.rank + self.curriculum_phase * 100)
        
        # Step 1: Optional weighted sampling within the curriculum bin
        if weights and sum(weights) > 0:
            # Convert to numpy array for efficient sampling
            indices_array = np.array(indices)
            normalized_weights = np.array(weights) / sum(weights)
            
            # Create a larger number of samples to ensure we have enough after partition
            # We'll trim this later
            oversampling_factor = 2  # Sample 2x more than needed
            num_samples_to_generate = total_size * oversampling_factor
            
            # Sample with weights
            generated_indices = np.random.choice(
                indices_array, 
                size=num_samples_to_generate,
                replace=True, 
                p=normalized_weights
            )
            
            # Convert back to list
            indices = generated_indices.tolist()
            total_size = len(indices)
        
        # Step 2: Optional shuffling
        if self.shuffle:
            # Convert to tensor for shuffling
            indices_tensor = torch.tensor(indices, dtype=torch.int64)
            indices_tensor = indices_tensor[torch.randperm(total_size, generator=g)]
            indices = indices_tensor.tolist()
        
        # Step 3: Distribute samples across processes
        # Calculate how many samples each process should get
        if self.drop_last and total_size % self.num_replicas != 0:
            # Remove samples to make it evenly divisible
            total_size_kept = total_size - (total_size % self.num_replicas)
            indices = indices[:total_size_kept]
        
        # Divide indices among processes
        indices_per_replica = [
            indices[i:i + self.num_samples_per_replica]
            for i in range(0, total_size, self.num_samples_per_replica)
        ]
        
        # If we have fewer slices than processes, pad with more samples
        if len(indices_per_replica) < self.num_replicas:
            # We need to generate more samples to cover all processes
            if weights and sum(weights) > 0:
                # Use weighted sampling to generate more samples
                extra_samples_needed = (self.num_replicas - len(indices_per_replica)) * self.num_samples_per_replica
                extra_indices = np.random.choice(
                    indices_array, 
                    size=extra_samples_needed,
                    replace=True, 
                    p=normalized_weights
                ).tolist()
                
                # Create slices for the remaining processes
                extra_indices_per_replica = [
                    extra_indices[i:i + self.num_samples_per_replica]
                    for i in range(0, len(extra_indices), self.num_samples_per_replica)
                ]
                
                indices_per_replica.extend(extra_indices_per_replica)
            else:
                # Just cycle through existing indices
                for i in range(len(indices_per_replica), self.num_replicas):
                    process_indices = indices[:(i % len(indices)) + 1] * self.num_samples_per_replica
                    process_indices = process_indices[:self.num_samples_per_replica]
                    indices_per_replica.append(process_indices)
        
        # Take the slice for this rank
        rank_indices = indices_per_replica[self.rank]
        
        # Ensure we have exactly num_samples_per_replica
        if len(rank_indices) > self.num_samples_per_replica:
            rank_indices = rank_indices[:self.num_samples_per_replica]
        elif len(rank_indices) < self.num_samples_per_replica:
            # Pad if we don't have enough
            additional_needed = self.num_samples_per_replica - len(rank_indices)
            if weights and sum(weights) > 0:
                # Generate more with weighted sampling
                additional_indices = np.random.choice(
                    indices_array, 
                    size=additional_needed,
                    replace=True, 
                    p=normalized_weights
                ).tolist()
            else:
                # Just repeat existing indices
                additional_indices = rank_indices[:additional_needed]
            
            rank_indices.extend(additional_indices)
        
        return iter(rank_indices)
    
    def __len__(self) -> int:
        """Return the number of samples for this process"""
        return self.num_samples_per_replica


# Example usage for creating data loaders with distributed sampling
def create_distributed_data_loaders(
    time_series_dataset,
    bin_indices,
    bin_weights,
    batch_size=32,
    num_workers=4,
    curriculum_phase=0,
    rank=0,
    world_size=1,
    seed=42
):
    """
    Create PyTorch DataLoaders with distributed curriculum sampling
    
    Args:
        time_series_dataset: Dataset to sample from
        bin_indices: Curriculum bin indices
        bin_weights: Curriculum bin weights
        batch_size: Batch size
        num_workers: Number of worker processes
        curriculum_phase: Starting curriculum phase
        rank: Process rank
        world_size: Total number of processes
        seed: Random seed
        
    Returns:
        train_loader: DataLoader with distributed curriculum sampling
        val_loader: DataLoader with distributed sequential sampling
        curriculum_callback: Callback for managing curriculum phases
    """
    from torch.utils.data import DataLoader, SequentialSampler
    
    # Create custom distributed curriculum sampler
    train_sampler = DistributedCurriculumSampler(
        bin_indices=bin_indices,
        bin_weights=bin_weights,
        num_samples=len(time_series_dataset),
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        curriculum_phase=curriculum_phase
    )
    
    # Create distributed sampler for validation
    val_sampler = DistributedSampler(
        time_series_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create curriculum callback
    class DistributedCurriculumCallback:
        def __init__(self, sampler, phase_epochs=[5, 10, 15]):
            self.sampler = sampler
            self.phase_epochs = phase_epochs
            self.current_phase = 0
            self.phase_boundaries = self._calculate_phase_boundaries()
        
        def _calculate_phase_boundaries(self):
            boundaries = []
            total = 0
            for epochs in self.phase_epochs:
                total += epochs
                boundaries.append(total)
            return boundaries
        
        def on_epoch_start(self, epoch):
            # Set epoch for reproducibility
            self.sampler.set_epoch(epoch)
            
            # Determine which phase we should be in
            new_phase = 0
            for i, boundary in enumerate(self.phase_boundaries[:-1]):
                if epoch >= boundary:
                    new_phase = i + 1
            
            # Update phase if needed
            if new_phase != self.current_phase:
                self.current_phase = new_phase
                self.sampler.set_curriculum_phase(new_phase)
                print(f"Transitioning to curriculum phase {new_phase}")
    
    # Create curriculum callback
    curriculum_callback = DistributedCurriculumCallback(
        train_sampler,
        phase_epochs=[5, 10, 15]  # Default phase durations
    )
    
    # Create train and validation loaders
    train_loader = DataLoader(
        time_series_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        time_series_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, curriculum_callback

if __name__ == "__main__":
    # test the sampler

    pass