#!/usr/bin/env python3
"""
Analyze clustering results across multiple seeds.
Computes mean and standard deviation for ARI, NMI, and F1-score
for each embedding and clustering algorithm combination.
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

def extract_clustering_metrics(log_file_path):
    """Extract clustering metrics from a log file."""
    metrics = {}
    
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
            
        # Extract metrics for K-Means (handles Test, All, Train, etc.)
        kmeans_pattern = r'(Test|All|Train|Valid).*K-Means.*ARI=(-?[0-9.]+),.*NMI=([0-9.]+),.*F1=([0-9.]+)'
        kmeans_match = re.search(kmeans_pattern, content)
        if kmeans_match:
            metrics['kmeans_ari'] = float(kmeans_match.group(2))
            metrics['kmeans_nmi'] = float(kmeans_match.group(3))
            metrics['kmeans_f1'] = float(kmeans_match.group(4))
        
        # Extract metrics for Ward (handles Test, All, Train, etc.)
        ward_pattern = r'(Test|All|Train|Valid).*Ward.*ARI=(-?[0-9.]+),.*NMI=([0-9.]+),.*F1=([0-9.]+)'
        ward_match = re.search(ward_pattern, content)
        if ward_match:
            metrics['ward_ari'] = float(ward_match.group(2))
            metrics['ward_nmi'] = float(ward_match.group(3))
            metrics['ward_f1'] = float(ward_match.group(4))
            
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        
    return metrics

def extract_experiment_info(dir_name):
    """Extract embedding name, seed, and other parameters from directory name."""
    # Pattern: {embedding}_{mode}_c{concat}_std{standardize}_p{perplexity}_seed{seed}
    # where mode can be 'test', 'all', 'train', 'validation', etc.
    pattern = r'(.+)_(test|all|train|validation|both)_c(\d+)_std(\d+)_p([0-9.]+)_seed(\d+)$'
    match = re.search(pattern, dir_name)
    
    if match:
        return {
            'embedding': match.group(1),
            'mode': match.group(2),
            'concat': int(match.group(3)),
            'standardize': int(match.group(4)),
            'perplexity': float(match.group(5)),
            'seed': int(match.group(6))
        }
    return None

def load_clustering_data(base_dir):
    """Load all clustering data from the results."""
    base_path = Path(base_dir)
    results = []
    
    for result_dir in base_path.iterdir():
        if not result_dir.is_dir():
            continue
            
        # Extract experiment info
        exp_info = extract_experiment_info(result_dir.name)
        if not exp_info:
            continue
            
        # Load metrics from log file
        log_file = result_dir / 'log.txt'
        if not log_file.exists():
            continue
            
        metrics = extract_clustering_metrics(log_file)
        if not metrics:
            continue
            
        # Combine experiment info and metrics
        result = {**exp_info, **metrics}
        results.append(result)
    
    return pd.DataFrame(results)

def compute_clustering_summary(df):
    """Compute mean and std for each embedding-algorithm combination."""
    summary_results = []
    
    # Group by embedding, mode, concat, and standardize
    grouped = df.groupby(['embedding', 'mode', 'concat', 'standardize'])
    
    for (embedding, mode, concat, standardize), group in grouped:
        if len(group) == 0:
            continue
            
        # Create base info
        base_info = {
            'embedding': embedding,
            'mode': mode,
            'concat': concat,
            'standardize': standardize,
            'n_seeds': len(group),
            'seeds': sorted(group['seed'].tolist())
        }
        
        # Compute stats for K-Means
        if 'kmeans_ari' in group.columns:
            kmeans_metrics = ['kmeans_ari', 'kmeans_nmi', 'kmeans_f1']
            for metric in kmeans_metrics:
                if metric in group.columns:
                    values = group[metric].dropna().values
                    if len(values) > 0:
                        base_info[f'{metric}_mean'] = np.mean(values)
                        base_info[f'{metric}_std'] = np.std(values, ddof=1) if len(values) > 1 else 0.0
        
        # Compute stats for Ward
        if 'ward_ari' in group.columns:
            ward_metrics = ['ward_ari', 'ward_nmi', 'ward_f1']
            for metric in ward_metrics:
                if metric in group.columns:
                    values = group[metric].dropna().values
                    if len(values) > 0:
                        base_info[f'{metric}_mean'] = np.mean(values)
                        base_info[f'{metric}_std'] = np.std(values, ddof=1) if len(values) > 1 else 0.0
        
        summary_results.append(base_info)
    
    return pd.DataFrame(summary_results)

def format_clustering_results(summary_df):
    """Format results in a nice table with mean ± std format."""
    formatted_results = []
    
    for _, row in summary_df.iterrows():
        result = {
            'embedding': row['embedding'],
            'mode': row['mode'],
            'concat': row['concat'],
            'standardize': row['standardize'],
            'n_seeds': row['n_seeds'],
            'seeds': str(row['seeds'])
        }
        
        # Format K-Means metrics
        kmeans_metrics = ['kmeans_ari', 'kmeans_nmi', 'kmeans_f1']
        for metric in kmeans_metrics:
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            if mean_col in row and pd.notna(row[mean_col]):
                mean_val = row[mean_col]
                std_val = row[std_col] if std_col in row and pd.notna(row[std_col]) else 0.0
                result[metric] = f"{mean_val:.4f} ± {std_val:.4f}"
        
        # Format Ward metrics
        ward_metrics = ['ward_ari', 'ward_nmi', 'ward_f1']
        for metric in ward_metrics:
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            if mean_col in row and pd.notna(row[mean_col]):
                mean_val = row[mean_col]
                std_val = row[std_col] if std_col in row and pd.notna(row[std_col]) else 0.0
                result[metric] = f"{mean_val:.4f} ± {std_val:.4f}"
            
        formatted_results.append(result)
    
    return pd.DataFrame(formatted_results)

def main():
    base_dir = "/projects/b1094/StarEmbed/src/output/clustering/all_split_all_standardize"
    output_dir = "/projects/b1094/StarEmbed/src/output/clustering/all_split_all_standardize"
    
    print("Loading clustering data...")
    df = load_clustering_data(base_dir)
    
    if df.empty:
        print("No clustering data found!")
        return
    
    print(f"Loaded {len(df)} result entries")
    print(f"Found {df['embedding'].nunique()} embeddings")
    print(f"Seeds found: {sorted(df['seed'].unique())}")
    
    # Compute summary statistics
    print("\nComputing summary statistics...")
    summary_df = compute_clustering_summary(df)
    
    # Save detailed summary
    summary_output_file = os.path.join(output_dir, "clustering_summary_detailed.csv")
    summary_df.to_csv(summary_output_file, index=False)
    print(f"Detailed summary saved to: {summary_output_file}")
    
    # Create formatted table
    formatted_df = format_clustering_results(summary_df)
    formatted_output_file = os.path.join(output_dir, "clustering_summary_formatted.csv")
    formatted_df.to_csv(formatted_output_file, index=False)
    print(f"Formatted summary saved to: {formatted_output_file}")
    
    # Print results
    print("\n" + "="*100)
    print("CLUSTERING RESULTS SUMMARY")
    print("="*100)
    
    for _, row in formatted_df.iterrows():
        print(f"\nEmbedding: {row['embedding']}")
        print(f"Mode: {row['mode']}")
        print(f"Config: concat={row['concat']}, standardize={row['standardize']}")
        print(f"Seeds: {row['seeds']}")
        print("-" * 80)
        
        # K-Means results
        if 'kmeans_ari' in row and pd.notna(row['kmeans_ari']):
            print("K-Means Clustering:")
            print(f"  ARI: {row['kmeans_ari']}")
            print(f"  NMI: {row['kmeans_nmi']}")
            print(f"  F1:  {row['kmeans_f1']}")
        
        # Ward results
        if 'ward_ari' in row and pd.notna(row['ward_ari']):
            print("Ward Clustering:")
            print(f"  ARI: {row['ward_ari']}")
            print(f"  NMI: {row['ward_nmi']}")
            print(f"  F1:  {row['ward_f1']}")

if __name__ == "__main__":
    main()
