#!/usr/bin/env python3
"""
Test script to demonstrate the updated embedding processing functions.
Shows how they handle different band configurations automatically.
"""

import numpy as np
import sys
import os

# Add src_clean to path for importing benchmark.utils
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from benchmark.utils import get_available_bands, process_embeddings, process_embeddings_batch, cal_avg_embedding

def test_band_detection():
    """Test automatic band detection from different data formats."""
    print("=== Testing Band Detection ===")
    
    # Test case 1: Pre-computed embeddings (g, r bands)
    example1 = {
        'g_embedding': np.random.randn(128).tolist(),
        'r_embedding': np.random.randn(128).tolist(),
        'class_str': 'EA'
    }
    bands1 = get_available_bands(example1)
    print(f"Pre-computed g/r embeddings: {bands1}")
    
    # Test case 2: Raw time series embeddings (g, r, i bands)
    example2 = {
        'embeddings_g': np.random.randn(50, 128).tolist(),
        'embeddings_r': np.random.randn(50, 128).tolist(),
        'embeddings_i': np.random.randn(50, 128).tolist(),
        'class_str': 'RRab'
    }
    bands2 = get_available_bands(example2)
    print(f"Raw time series g/r/i embeddings: {bands2}")
    
    # Test case 3: bands_data structure
    example3 = {
        'bands_data': {
            'g': {'target': [1, 2, 3]},
            'r': {'target': [2, 3, 4]},
            'i': {'target': [3, 4, 5]}
        },
        'class_str': 'EW'
    }
    bands3 = get_available_bands(example3)
    print(f"bands_data structure: {bands3}")
    
def test_process_embeddings():
    """Test the updated process_embeddings function."""
    print("\n=== Testing process_embeddings ===")
    
    # Create example with g, r, i bands
    example = {
        'embeddings_g': np.random.randn(50, 128).tolist(),
        'embeddings_r': np.random.randn(50, 128).tolist(), 
        'embeddings_i': np.random.randn(50, 128).tolist(),
        'class_str': 'RRab'
    }
    
    # Test different scenarios
    result1 = process_embeddings(example.copy(), scenario="concat", return_separate=True)
    print(f"Concat scenario - combined shape: {result1['combined_embedding'].shape}")
    print(f"Separate embeddings created: {[k for k in result1.keys() if k.endswith('_embedding')]}")
    
    result2 = process_embeddings(example.copy(), scenario="avg", return_separate=True)
    print(f"Average scenario - combined shape: {result2['combined_embedding'].shape}")
    
    result3 = process_embeddings(example.copy(), scenario="i", return_separate=True)
    print(f"I-band only scenario - combined shape: {result3['combined_embedding'].shape}")
    
def test_batch_processing():
    """Test the updated batch processing function."""
    print("\n=== Testing process_embeddings_batch ===")
    
    # Create batch with 3 examples, each with g, r, i bands
    batch = {
        'embeddings_g': [np.random.randn(50, 128).tolist() for _ in range(3)],
        'embeddings_r': [np.random.randn(50, 128).tolist() for _ in range(3)],
        'embeddings_i': [np.random.randn(50, 128).tolist() for _ in range(3)],
        'class_str': ['RRab', 'EA', 'EW']
    }
    
    result = process_embeddings_batch(batch, hand_crafted=False)
    print(f"Batch processing results:")
    for key in result.keys():
        if key.endswith('_embedding'):
            print(f"  {key}: {len(result[key])} examples, each shape {np.array(result[key][0]).shape}")

def test_legacy_function():
    """Test the updated legacy cal_avg_embedding function."""
    print("\n=== Testing cal_avg_embedding (legacy) ===")
    
    # Test with multiple bands
    example = {
        'embeddings_g': np.random.randn(50, 128).tolist(),
        'embeddings_r': np.random.randn(50, 128).tolist(),
        'embeddings_i': np.random.randn(50, 128).tolist(),
        'class_str': 'RRab'
    }
    
    result1 = cal_avg_embedding(example.copy(), concat=True)
    print(f"Concatenated embedding shape: {result1['avg_embedding'].shape}")
    
    result2 = cal_avg_embedding(example.copy(), concat=False)
    print(f"Averaged embedding shape: {result2['avg_embedding'].shape}")

if __name__ == "__main__":
    test_band_detection()
    test_process_embeddings()
    test_batch_processing()
    test_legacy_function()
    print("\nAll tests completed successfully!")
