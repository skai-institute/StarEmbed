# 2 x 2
# class 1
# /home/magics/hdd/sky_ws/ebsim_ws/data/lsdb/1/120620174173493992_g.png
# /home/magics/hdd/sky_ws/ebsim_ws/data/lsdb/1/129030693624086743_g.png
# class 5
# /home/magics/hdd/sky_ws/ebsim_ws/data/lsdb/5/97211305511826371_g.png # 174
# /home/magics/hdd/sky_ws/ebsim_ws/data/lsdb/5/130603207444200557_g.png # 397
# /home/magics/hdd/sky_ws/ebsim_ws/data/lsdb/5/135293431680766623_g.png

import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from astropy.time import Time
from chronos.base import BaseChronosPipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def seed_everything(seed=42):
    """
    Seed everything for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_star_embeddings(processed_df, model_name="amazon/chronos-t5-small"):
    """
    Get embeddings for each star's magnitude data using Chronos model.
    
    Args:
        processed_df: DataFrame with two-level index (ps1_objid, phase)
        model_name: Name of the Chronos model to use
        
    Returns:
        np.ndarray: Array of shape (n_stars, embedding_size) containing averaged embeddings
        np.ndarray: Array of shape (n_stars) containing true labels
        list: List of star IDs
    """
    # Initialize Chronos pipeline
    pipeline = BaseChronosPipeline.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    
    star_embeddings = []
    true_labels = []
    star_ids_list = []

    # Process each star
    for star_id in processed_df.index.get_level_values('ps1_objid').unique():
        # Get magnitude data for this star
        star_data = processed_df.loc[star_id]
        star_ids_list.append(star_id)

        # Get magnitudes with NaN values preserved
        mags = star_data['mag'].values
        
        # Get true label (assuming it's in the DataFrame)
        true_label = star_data['class'].iloc[0]  # Get class label from first row
        true_labels.append(true_label)

        # Convert to tensor, keeping NaN values
        context = torch.tensor(mags, dtype=torch.float32)
        
        # Get embeddings
        embeddings, _ = pipeline.embed(context=context)
        
        # Average embeddings across context length dimension
        avg_embedding = embeddings.mean(dim=1)  # Shape: [1, embedding_size]
        avg_embedding = avg_embedding.to(torch.float32)

        # Convert to numpy and append
        star_embeddings.append(avg_embedding.cpu().numpy().squeeze())
    
    # Stack all embeddings into a single numpy array
    final_embeddings = np.stack(star_embeddings)
    true_labels = np.array(true_labels)

    return final_embeddings, true_labels, star_ids_list


def cluster_embeddings(embeddings, n_clusters=2, random_state=42, true_labels=None):
    """
    Perform K-means clustering on star embeddings.
    
    Args:
        embeddings: numpy array of shape (n_stars, embedding_size)
        n_clusters: number of clusters (default: 2)
        random_state: random seed for reproducibility
        true_labels: optional ground truth labels for evaluation metrics
        
    Returns:
        labels: numpy array of cluster labels
        kmeans: fitted KMeans model
        metrics: dictionary of evaluation metrics
    """
    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(embeddings)
    
    # Calculate evaluation metrics
    metrics = {}
    
    # Intrinsic metrics (don't require ground truth)
    if len(embeddings) > 1:  # Silhouette score requires at least 2 samples
        metrics['silhouette_score'] = silhouette_score(embeddings, labels)
    if len(embeddings) > n_clusters:  # Davies-Bouldin requires number of samples > number of clusters
        metrics['davies_bouldin_score'] = davies_bouldin_score(embeddings, labels)
    if len(embeddings) > n_clusters:  # Calinski-Harabasz requires number of samples > number of clusters
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, labels)
    
    # External metrics (require ground truth)
    if true_labels is not None:
        metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, labels)
        metrics['normalized_mutual_info_score'] = normalized_mutual_info_score(true_labels, labels)
        
        # Calculate purity
        unique_true_labels = np.unique(true_labels)
        label_to_idx = {label: idx for idx, label in enumerate(unique_true_labels)}
        
        contingency_matrix = np.zeros((n_clusters, len(unique_true_labels)))
        for i in range(len(labels)):
            true_label_idx = label_to_idx[true_labels[i]]
            contingency_matrix[labels[i], true_label_idx] += 1
        metrics['purity'] = np.sum(np.max(contingency_matrix, axis=1)) / len(labels)
    
    # Calculate inertia (sum of squared distances to nearest centroid)
    metrics['inertia'] = kmeans.inertia_
        
    return labels, kmeans, metrics


def predict_lightcurve(processed_df, model_name="amazon/chronos-t5-small", context_percent=0.8, split_point=None, fig_dir='./data/debug/eval'):
    """
    Predict magnitude values for each star in the processed DataFrame
    
    Args:
        processed_df: DataFrame with two-level index (ps1_objid, phase)
        model_name: Name of the Chronos model to use
        context_percent: Percentage of data to use as context (default: 0.8)
        split_point: Point to split the data into context and target, it will override context_percent (default: None)

    Returns:
        dict: Dictionary containing predictions for each star
            {star_id: {'context': array, 
                      'target': array,
                      'mean': array, 
                      'quantiles': array}}
    """
    os.makedirs(fig_dir, exist_ok=True)  # equivalent to mkdir -p

    # Initialize Chronos pipeline
    pipeline = BaseChronosPipeline.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    
    predictions = {}
    
    # Process each star
    for star_id in processed_df.index.get_level_values('ps1_objid').unique():
        # Get magnitude data for this star
        star_data = processed_df.loc[star_id]
        
        # Get magnitudes
        mags = star_data['mag'].values
        
        # Calculate split point
        n_points = len(mags)
        if split_point is None:
            split_point = int(context_percent * n_points)
        else:
            split_point = int(split_point)
        
        # Split into context and target
        context = mags[:split_point]
        target = mags[split_point:]
        
        # Convert to tensor
        context_tensor = torch.tensor(context, dtype=torch.float32)
        
        # Predict
        quantiles, mean = pipeline.predict_quantiles(
            context=context_tensor,
            prediction_length=len(target),
            quantile_levels=[0.1, 0.5, 0.9],
        )

        # Squeeze predictions to remove extra dimensions
        mean = mean.squeeze().cpu().numpy()
        quantiles = quantiles.squeeze().cpu().numpy()
        
        # Store results
        predictions[star_id] = {
            'context': context,
            'target': target,
            'mean': mean,
            'quantiles': quantiles
        }
        
        # Calculate MSE dropping NaN values
        valid_mask = ~np.isnan(target) & ~np.isnan(mean)
        target_clean = target[valid_mask]
        mean_pred_clean = mean[valid_mask]
        mse = np.mean((target_clean - mean_pred_clean)**2)

        # Plot prediction for this star
        plt.figure(figsize=(12, 6))
        phases = star_data.index.get_level_values('phase')
        context_phases = phases[:split_point]
        target_phases = phases[split_point:]
        
        # Plot context
        plt.plot(context_phases, context, 'b.', label='Context', alpha=0.5)
        # Plot target
        plt.plot(target_phases, target, 'g.', label='Target', alpha=0.5)
        # Plot prediction
        plt.plot(target_phases, mean, 'r-', label='Prediction', linewidth=2)
        # Plot quantiles
        plt.fill_between(target_phases, 
                        quantiles[:, 0], 
                        quantiles[:, 2], 
                        color='r', alpha=0.2, label='90% Confidence')

        
        plt.title(f'Star {star_id} (Class {star_data["class"].iloc[0]}) Prediction - MSE: {mse:.4f}')
        plt.xlabel('Phase')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.gca().invert_yaxis()  # Invert y-axis for magnitude
        plt.savefig(f'{fig_dir}/prediction_{star_id}.png')
        plt.close()
    
    return predictions


def main(args):
    """
    Main function to run forecast evaluation.
    
    Args:
        args: Command line arguments containing:
            model_path: Path to Chronos model to use
            data_path: Path to processed light curve CSV file
            output_dir: Directory to save prediction results and plots
    """    # initialize random seed
    seed_everything()

    # Load processed data
    print(f"Loading data from {args.data_path}...")
    processed_df = pd.read_csv(args.data_path)
    processed_df = processed_df.set_index(['ps1_objid', 'phase']).sort_index()
    star_ids_list = processed_df.index.get_level_values('ps1_objid').unique()
    print(f"Loaded data for {len(star_ids_list)} stars\n")

    # Print some information about the processed data
    for star_id in star_ids_list[:10]:
        star_data = processed_df.loc[star_id]
        print(f"Star {star_id}:")
        print(f"Phase range: {star_data.index.min():.3f} to {star_data.index.max():.3f}")
        print(f"Number of points: {len(star_data)}\n")
    print("...\n")

    # Get embeddings
    embeddings, true_labels, star_ids_list = get_star_embeddings(processed_df, model_name=args.model_path)
    # print(f"Star IDs: {star_ids_list}")
    # print(f"Embeddings shape: {embeddings.shape}")
    print(f"True labels: {true_labels}\n")

    # Perform clustering and Print clustering metrics
    labels, kmeans, metrics = cluster_embeddings(embeddings, n_clusters=2, true_labels=true_labels)
    print(f"Labels: {labels}\n")
    print("Clustering Performance Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    print()

    # Plot the data for clustering
    cluster_dir = args.output_dir
    star_to_cluster = dict(zip(star_ids_list, labels))
    star_to_embedding = dict(zip(star_ids_list, embeddings.tolist()))
    plt.figure(figsize=(10, 6))
    for star_id in star_ids_list:
        star_data = processed_df.loc[star_id]
        cluster_label = star_to_cluster[star_id]
        plt.plot(star_data.index, star_data['mag'], label=f'Star {star_id}, True class {star_data["class"].iloc[0]}, Predicted label {cluster_label}', linestyle='None', marker='o')
    plt.title(f'Kmeans Clusters for Lightcurves')
    plt.legend()
    plt.show()
    os.makedirs(cluster_dir, exist_ok=True)
    plt.savefig(f'{cluster_dir}/lightcurve.png')
    plt.close()


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Forecast light curves using Chronos model")
    parser.add_argument("--model-path", type=str, default="amazon/chronos-t5-small",
                        help="Path to Chronos model to use")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to processed light curve CSV file")
    parser.add_argument("--output-dir", type=str, default="./data/eval/kmeans",
                        help="Directory to save prediction results and plots")
    
    args = parser.parse_args()
    
    # Check if data path exists before proceeding
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Call main with args
    main(args)