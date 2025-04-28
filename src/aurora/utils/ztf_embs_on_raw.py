import pandas as pd
import torch
from chronos import BaseChronosPipeline
import datasets
from tqdm import tqdm
import numpy as np


# Load model
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-large",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

def length_filter(example, min_length=160):
    # Get the length of the specified field
    field_length = len(example['target'])
    
    # Check minimum length requirement
    if field_length < min_length:
        return False
        
    return True

# Load dataset
dataset = datasets.load_from_disk("/projects/p32795/weijian/hf_csdr1_raw3")
dataset = dataset.filter(length_filter)
# Create a function to process each example
def add_embeddings(example):
    # Normalize the data
    target = np.array(example["target"])
    mu = np.mean(target)
    sigma = np.std(target)
    normalized_target = (target - mu) / sigma
    
    # Convert to tensor for model input
    context = torch.tensor(normalized_target)
    
    # Generate embeddings
    embeddings, tokenizer_state = pipeline.embed(context)
    
    # Add embeddings to the example (convert tensor to numpy for storage)
    example["embeddings"] = embeddings.to(torch.float32).cpu().numpy()
    
    return example

# Process the dataset with progress bar
print("Generating embeddings for all items...")
updated_dataset = dataset.map(add_embeddings, desc="Generating embeddings")

# Save the updated dataset
output_path = ""
print(f"Saving dataset with embeddings to {output_path}")
updated_dataset.save_to_disk(output_path)

print("Done!")