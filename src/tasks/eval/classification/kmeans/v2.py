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
from pathlib import Path

from astropy.time import Time
from chronos.base import BaseChronosPipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def plot_light_curves_by_cluster(true_labels, predicted_labels, star_ids, processed_df, n_per_cluster=10, output_dir="./data/eval/kmeans_plots"):
    """
    Plot light curves organized by predicted clusters.
    
    Args:
        true_labels: Array of true class labels
        predicted_labels: Array of predicted cluster labels
        star_ids: List of star IDs
        processed_df: DataFrame with light curve data (multi-index with ps1_objid and phase/mjd)
        n_per_cluster: Number of stars to plot per cluster
        output_dir: Directory to save the plots
    """
    # Convert to numpy arrays if not already
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    
    # Create dictionary mapping star_id to its true label and predicted cluster
    star_info = {}
    for i, star_id in enumerate(star_ids):
        star_info[star_id] = {
            'true_label': true_labels[i],
            'predicted_cluster': predicted_labels[i]
        }
    
    # Get unique predicted clusters
    unique_clusters = np.unique(predicted_labels)
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for cluster in unique_clusters:
        # Create directory for this cluster
        cluster_dir = output_path / f"predicted_cluster_{cluster}"
        cluster_dir.mkdir(exist_ok=True)
        
        # Find stars in this cluster
        cluster_stars = [star_id for star_id, info in star_info.items() 
                         if info['predicted_cluster'] == cluster]
        
        # Limit to 10 stars per cluster for the overview plot if there are more
        if len(cluster_stars) > n_per_cluster:
            overview_stars = random.sample(cluster_stars, n_per_cluster)
        else:
            overview_stars = cluster_stars
        
        # Create overview plot for this cluster
        plt.figure(figsize=(15, 10))
        for star_id in overview_stars:
            star_data = processed_df.loc[star_id]
            true_label = star_info[star_id]['true_label']
            
            # Plot the light curve
            plt.scatter(star_data.index, star_data['mag'], alpha=0.7, s=20, 
                     label=f'Star {star_id} (True: Class {true_label})')
        
        # Calculate true label distribution for this cluster
        true_label_counts = {}
        for star_id in cluster_stars:
            true_label = star_info[star_id]['true_label']
            true_label_counts[true_label] = true_label_counts.get(true_label, 0) + 1
        
        # Title with cluster info and class distribution
        distribution_str = ", ".join([f"Class {label}: {count}" for label, count in true_label_counts.items()])
        plt.title(f'Light Curves for Predicted Cluster {cluster}\nClass Distribution: {distribution_str}',
                 fontsize=14)
        
        plt.xlabel('Phase/Time', fontsize=12)
        plt.ylabel('Magnitude', fontsize=12)
        plt.gca().invert_yaxis()  # Astronomical convention: brighter objects have lower magnitudes
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the overview plot
        plt.savefig(cluster_dir / f"cluster_{cluster}_overview.png")
        plt.close()
        
        # Create a second overview plot with folded light curves
        plt.figure(figsize=(15, 10))
        
        for star_id in overview_stars:
            star_data = processed_df.loc[star_id]
            true_label = star_info[star_id]['true_label']
            
            # If we already have phase in the index, just use it
            if 'phase' in str(star_data.index.name):
                phase = star_data.index
                plt.scatter(phase, star_data['mag'], alpha=0.7, s=20,
                          label=f'Star {star_id} (True: Class {true_label})')
            # Otherwise, try to fold it
            else:
                try:
                    # Get the period from the dataframe if it exists
                    if 'period' in star_data.columns:
                        period = star_data['period'].iloc[0]
                    else:
                        # Estimate period as 0.5 days if not available
                        print(f"No period found for star {star_id}")
                        period = 0.5
                    
                    # Calculate phase as mjd % period / period
                    mjd = star_data.index
                    phase = (mjd % period) / period
                    plt.scatter(phase, star_data['mag'], alpha=0.7, s=20,
                              label=f'Star {star_id} (True: Class {true_label})')
                except Exception as e:
                    print(f"Couldn't fold light curve for star {star_id}: {str(e)}")
        
        plt.title(f'Folded Light Curves for Predicted Cluster {cluster}\nClass Distribution: {distribution_str}',
                 fontsize=14)
        plt.xlabel('Phase', fontsize=12)
        plt.ylabel('Magnitude', fontsize=12)
        plt.gca().invert_yaxis()
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the folded overview plot
        plt.savefig(cluster_dir / f"cluster_{cluster}_folded_overview.png")
        plt.close()
        
        # Create individual plots for each star in this cluster
        for star_id in overview_stars:
            star_data = processed_df.loc[star_id]
            true_label = star_info[star_id]['true_label']
            
            # Create a figure with two subplots - original and folded
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
            
            # Plot 1: Original light curve (time series)
            ax1.scatter(star_data.index, star_data['mag'], color='blue', s=25)
            ax1.set_title(f'Original Light Curve', fontsize=14)
            ax1.set_xlabel('Time (MJD)' if 'mjd' in str(star_data.index.name) else 'Phase', fontsize=12)
            ax1.set_ylabel('Magnitude', fontsize=12)
            ax1.invert_yaxis()  # Astronomical convention
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Folded light curve
            # If we already have phase in the index, just use it
            if 'phase' in str(star_data.index.name):
                phase = star_data.index
                ax2.scatter(phase, star_data['mag'], color='red', s=25)
            # Otherwise, try to fold it
            else:
                try:
                    # Get the period from the dataframe if it exists
                    if 'period' in star_data.columns:
                        period = star_data['period'].iloc[0]
                    else:
                        # Estimate period as 0.5 days if not available (you can change this default)
                        print(f"No period found for star {star_id}")
                        period = 0.5
                    
                    # Calculate phase as mjd % period / period
                    mjd = star_data.index
                    phase = (mjd % period) / period
                    ax2.scatter(phase, star_data['mag'], color='red', s=25)
                except Exception as e:
                    # If folding fails, note it in the plot
                    ax2.text(0.5, 0.5, f"Couldn't fold: {str(e)}", 
                            ha='center', va='center', transform=ax2.transAxes)
            
            ax2.set_title(f'Folded Light Curve', fontsize=14)
            ax2.set_xlabel('Phase', fontsize=12)
            ax2.set_ylabel('Magnitude', fontsize=12)
            ax2.invert_yaxis()
            ax2.grid(True, alpha=0.3)
            
            # Add overall title for the figure
            plt.suptitle(f'Star {star_id} (True: Class {true_label}, Predicted: Cluster {cluster})',
                         fontsize=16)
            
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(cluster_dir / f"star_{star_id}.png")
            plt.close()


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
        true_label = star_data['star_class'].iloc[0]  # Get class label from first row
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
    if args.fold:
        processed_df = processed_df.set_index(['ps1_objid', 'phase']).sort_index()
    else:
        processed_df = processed_df.set_index(['ps1_objid', 'mjd']).sort_index()
    star_ids_list = processed_df.index.get_level_values('ps1_objid').unique()
    print(f"Loaded data for {len(star_ids_list)} stars\n")

    # Print some information about the processed data
    # for star_id in star_ids_list[:10]:
    #     star_data = processed_df.loc[star_id]
    #     print(f"Star {star_id}:")
    #     print(f"Phase range: {star_data.index.min():.3f} to {star_data.index.max():.3f}")
    #     print(f"Number of points: {len(star_data)}\n")
    # print("...\n")

    # Get embeddings
    embeddings, true_labels, star_ids_list = get_star_embeddings(processed_df, model_name=args.model_path)
    # print(f"Star IDs: {star_ids_list}")
    # print(f"Embeddings shape: {embeddings.shape}")
    print(f"True labels: {','.join([str(label) for label in true_labels])}\n")

    # Perform clustering and Print clustering metrics
    labels, kmeans, metrics = cluster_embeddings(embeddings, n_clusters=args.n_clusters, true_labels=true_labels)
    print(f"Labels: {','.join([str(label) for label in labels])}\n")
    
    print("Clustering Performance Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    print()

    # Use the new plotting function instead of the old plotting code
    cluster_dir = args.output_dir
    os.makedirs(cluster_dir, exist_ok=True)
    
    # Plot light curves organized by predicted clusters
    plot_light_curves_by_cluster(true_labels, labels, star_ids_list, processed_df, n_per_cluster=10,
                                output_dir=os.path.join(cluster_dir, "cluster_plots"))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Forecast light curves using Chronos model")
    parser.add_argument("--model-path", type=str, default="amazon/chronos-t5-small",
                        help="Path to Chronos model to use")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to processed light curve CSV file")
    parser.add_argument("--output-dir", type=str, default="./data/eval/kmeans",
                        help="Directory to save prediction results and plots")
    parser.add_argument("--fold", action="store_true",
                        help="Fold the light curves")
    parser.add_argument("--n-clusters", type=int, default=2,
                        help="Number of clusters to use for KMeans")
    
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