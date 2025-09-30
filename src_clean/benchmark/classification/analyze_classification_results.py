#!/usr/bin/env python3
"""
Analyze classification results across multiple seeds.
Computes mean and standard deviation for accuracy, precision, recall, and F1-score
for each embedding and classifier combination.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import re

def extract_embedding_name(dir_name):
    """Extract embedding name from directory name, removing seed info."""
    # Remove _concat_seed{number} pattern
    pattern = r'_concat_seed\d+$'
    return re.sub(pattern, '', dir_name)

def load_metrics_data(base_dir):
    """Load all metrics data from the linear classification results."""
    base_path = Path(base_dir)
    results = []
    
    for result_dir in base_path.iterdir():
        if not result_dir.is_dir():
            continue
            
        dir_name = result_dir.name
        embedding_name = extract_embedding_name(dir_name)
        
        # Extract seed from directory name
        seed_match = re.search(r'seed(\d+)$', dir_name)
        if not seed_match:
            continue
        seed = int(seed_match.group(1))
        
        # Load KNN metrics
        knn_file = result_dir / 'knn_metrics_report.csv'
        if knn_file.exists():
            knn_df = pd.read_csv(knn_file, index_col=0)
            # Extract key metrics
            accuracy = knn_df.loc['accuracy', 'precision']  # accuracy is stored in precision column
            macro_precision = knn_df.loc['macro avg', 'precision']
            macro_recall = knn_df.loc['macro avg', 'recall']
            macro_f1 = knn_df.loc['macro avg', 'f1-score']
            weighted_precision = knn_df.loc['weighted avg', 'precision']
            weighted_recall = knn_df.loc['weighted avg', 'recall']
            weighted_f1 = knn_df.loc['weighted avg', 'f1-score']
            
            results.append({
                'embedding': embedding_name,
                'classifier': 'knn',
                'seed': seed,
                'accuracy': accuracy,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1,
                'weighted_precision': weighted_precision,
                'weighted_recall': weighted_recall,
                'weighted_f1': weighted_f1
            })
        
        # Load Logistic Regression metrics
        logistic_file = result_dir / 'logistic_metrics_report.csv'
        if logistic_file.exists():
            logistic_df = pd.read_csv(logistic_file, index_col=0)
            # Extract key metrics
            accuracy = logistic_df.loc['accuracy', 'precision']  # accuracy is stored in precision column
            macro_precision = logistic_df.loc['macro avg', 'precision']
            macro_recall = logistic_df.loc['macro avg', 'recall']
            macro_f1 = logistic_df.loc['macro avg', 'f1-score']
            weighted_precision = logistic_df.loc['weighted avg', 'precision']
            weighted_recall = logistic_df.loc['weighted avg', 'recall']
            weighted_f1 = logistic_df.loc['weighted avg', 'f1-score']
            
            results.append({
                'embedding': embedding_name,
                'classifier': 'logistic',
                'seed': seed,
                'accuracy': accuracy,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1,
                'weighted_precision': weighted_precision,
                'weighted_recall': weighted_recall,
                'weighted_f1': weighted_f1
            })
    
    return pd.DataFrame(results)

def compute_summary_stats(df):
    """Compute mean and std for each embedding-classifier combination."""
    # Group by embedding and classifier
    grouped = df.groupby(['embedding', 'classifier'])
    
    summary_results = []
    
    for (embedding, classifier), group in grouped:
        if len(group) == 0:
            continue
            
        # Compute mean and std for each metric
        stats = {
            'embedding': embedding,
            'classifier': classifier,
            'n_seeds': len(group),
            'seeds': sorted(group['seed'].tolist())
        }
        
        metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 
                  'weighted_precision', 'weighted_recall', 'weighted_f1']
        
        for metric in metrics:
            values = group[metric].values
            stats[f'{metric}_mean'] = np.mean(values)
            stats[f'{metric}_std'] = np.std(values, ddof=1) if len(values) > 1 else 0.0
            
        summary_results.append(stats)
    
    return pd.DataFrame(summary_results)

def format_results_table(summary_df):
    """Format results in a nice table with mean ± std format."""
    formatted_results = []
    
    for _, row in summary_df.iterrows():
        result = {
            'embedding': row['embedding'],
            'classifier': row['classifier'],
            'n_seeds': row['n_seeds'],
            'seeds': str(row['seeds'])
        }
        
        # Format as mean ± std
        metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 
                  'weighted_precision', 'weighted_recall', 'weighted_f1']
        
        for metric in metrics:
            mean_val = row[f'{metric}_mean']
            std_val = row[f'{metric}_std']
            result[metric] = f"{mean_val:.4f} ± {std_val:.4f}"
            
        formatted_results.append(result)
    
    return pd.DataFrame(formatted_results)

def main():
    base_dir = "/projects/b1094/StarEmbed/src/output/linear_classification"
    output_dir = "/projects/b1094/StarEmbed/src/output/linear_classification"
    
    print("Loading metrics data...")
    df = load_metrics_data(base_dir)
    
    if df.empty:
        print("No data found!")
        return
    
    print(f"Loaded {len(df)} result entries")
    print(f"Found {df['embedding'].nunique()} embeddings and {df['classifier'].nunique()} classifiers")
    print(f"Seeds found: {sorted(df['seed'].unique())}")
    
    # Compute summary statistics
    print("\nComputing summary statistics...")
    summary_df = compute_summary_stats(df)
    
    # Save detailed summary with separate mean and std columns
    summary_output_file = os.path.join(output_dir, "classification_summary_detailed.csv")
    summary_df.to_csv(summary_output_file, index=False)
    print(f"Detailed summary saved to: {summary_output_file}")
    
    # Create formatted table with mean ± std
    formatted_df = format_results_table(summary_df)
    formatted_output_file = os.path.join(output_dir, "classification_summary_formatted.csv")
    formatted_df.to_csv(formatted_output_file, index=False)
    print(f"Formatted summary saved to: {formatted_output_file}")
    
    # Print a preview of the results
    print("\n" + "="*100)
    print("CLASSIFICATION RESULTS SUMMARY")
    print("="*100)
    
    # Sort by classifier and then by accuracy for better readability
    display_df = formatted_df.copy()
    display_df['accuracy_numeric'] = summary_df['accuracy_mean'].values
    display_df = display_df.sort_values(['classifier', 'accuracy_numeric'], ascending=[True, False])
    display_df = display_df.drop('accuracy_numeric', axis=1)
    
    for classifier in ['knn', 'logistic']:
        classifier_data = display_df[display_df['classifier'] == classifier]
        if len(classifier_data) > 0:
            print(f"\n{classifier.upper()} Results:")
            print("-" * 80)
            for _, row in classifier_data.iterrows():
                print(f"{row['embedding'][:60]:<60}")
                print(f"  Accuracy: {row['accuracy']:<20} Macro F1: {row['macro_f1']}")
                print(f"  Macro Precision: {row['macro_precision']:<15} Macro Recall: {row['macro_recall']}")
                print(f"  Seeds: {row['seeds']}")
                print()

if __name__ == "__main__":
    main()
