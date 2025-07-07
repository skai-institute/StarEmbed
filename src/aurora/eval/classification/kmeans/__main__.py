# 2 x 2
# class 1
# /home/magics/hdd/sky_ws/ebsim_ws/outputs/lsdb/1/120620174173493992_g.png
# /home/magics/hdd/sky_ws/ebsim_ws/outputs/lsdb/1/129030693624086743_g.png
# class 5
# /home/magics/hdd/sky_ws/ebsim_ws/outputs/lsdb/5/97211305511826371_g.png # 174
# /home/magics/hdd/sky_ws/ebsim_ws/outputs/lsdb/5/130603207444200557_g.png # 397
# /home/magics/hdd/sky_ws/ebsim_ws/outputs/lsdb/5/135293431680766623_g.png

import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from astropy.time import Time
from chronos.base import BaseChronosPipeline
from sklearn.cluster import KMeans


def seed_everything(seed=42):
    """
    Seed everything for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_star_ids(data_path='/home/magics/hdd/sky_ws/ebsim_ws/outputs/lsdb/combined_lightcurves.csv', 
                class_ids=None, stars_per_class=None, band='g'):
    """
    Get star IDs from the combined lightcurves CSV file.
    
    Args:
        data_path (str): Path to the CSV file containing star data
        class_ids (list, optional): List of class IDs to filter by. If None, all classes are used.
        stars_per_class (int, optional): Number of stars to return per class. If None, all stars are returned.
        band (str, optional): Band to filter by.
        random_seed (int, optional): Random seed for reproducibility when sampling stars.
        
    Returns:
        list: List of star IDs
    """    
    # Read the data
    print(f"Reading data from {data_path}...")
    star_data = pd.read_csv(data_path)
    
    # Get unique combination of star ID and class
    star_info = star_data[['ps1_objid', 'star_class']].drop_duplicates()
    
    # If no class filtering is requested, return all star IDs
    if class_ids is None:
        all_star_ids = star_info['ps1_objid'].unique()
        print(f"No class filtering requested. Returning all {len(all_star_ids)} stars.")
        return all_star_ids.tolist()
    
    # Filter by specified classes
    star_info = star_info[star_info['star_class'].isin(class_ids)]
    
    # If no stars_per_class limit is set, return all filtered stars
    if stars_per_class is None:
        all_star_ids = star_info['ps1_objid'].unique()
        print(f"No stars_per_class limit. Returning all {len(all_star_ids)} stars from classes {class_ids}.")
        return all_star_ids.tolist()
    
    # Get list of available classes after filtering
    available_classes = sorted(star_info['star_class'].unique())
    print(f"Available classes after filtering: {available_classes}")
    
    # Initialize result list
    selected_star_ids = []
    
    # Process each class
    for class_id in available_classes:
        # Get all star IDs for this class
        class_star_ids = star_info[star_info['star_class'] == class_id]['ps1_objid'].unique()
        print(f"Class {class_id}: {len(class_star_ids)} stars available")
        
        # Sample stars up to the requested number
        if stars_per_class < len(class_star_ids):
            # Randomly sample the required number of stars
            class_star_ids = np.random.choice(class_star_ids, size=stars_per_class, replace=False)
            print(f"Selected {len(class_star_ids)} stars from class {class_id}")
        else:
            print(f"Requested {stars_per_class} stars but only {len(class_star_ids)} available for class {class_id}. Using all available.")
        
        # Add to result list
        selected_star_ids.extend(class_star_ids)
    
    print(f"Total stars selected: {len(selected_star_ids)}")
    return selected_star_ids


def create_indexed_star_df(star_ids, band, data_path='/home/magics/hdd/sky_ws/ebsim_ws/outputs/lsdb/combined_lightcurves.csv'):
    """
    Create a DataFrame with two-level index (ps1_objid and datetime) for given star IDs.
    
    Args:
        star_ids (list): List of PS1 object IDs to process
        band (str): Band to process
        data_path (str): Path to the CSV file containing star data
        
    Returns:
        pd.DataFrame: DataFrame with two-level index (ps1_objid, datetime)
    """
    # Read the data
    star_data = pd.read_csv(data_path)
    
    # Filter and combine data for all star IDs
    star_dfs = []
    for star_id in star_ids:
        df = star_data[(star_data['ps1_objid']==star_id) & (star_data['band']==band)]
        star_dfs.append(df)
    star_df = pd.concat(star_dfs, ignore_index=True)
    
    # Convert MJD to datetime
    t = Time(star_df['mjd'], format='mjd')
    star_df['datetime'] = pd.to_datetime(t.datetime)
    
    # Set multi-level index and sort
    star_df = star_df.set_index(['ps1_objid', 'datetime']).sort_index()
    
    return star_df


def process_lightcurve_data(star_df, phase_interval=0.005, interpolation_method="nan"):
    """
    Process light curve data with phase folding and resampling.
    
    Args:
        star_df: DataFrame with two-level index (ps1_objid, datetime)
        phase_interval: Interval for phase bins
        interpolation_method: Method for interpolating missing values
                             "nan" - Use NaN for missing values
                             "LastValue" - Use the last valid value (None if first)
        
    Returns:
        pd.DataFrame: Processed DataFrame with two-level index (ps1_objid, phase)
    """
    # Reset index to work with ps1_objid as column
    df = star_df.reset_index()
    
    processed_dfs = []
    
    # Process each star separately
    for star_id in df['ps1_objid'].unique():
        star_data = df[df['ps1_objid'] == star_id].copy()
        period = star_data['period'].iloc[0]  # Get period for this star
        class_id = star_data['star_class'].iloc[0]
        
        # Calculate phase (0 to period)
        star_data['phase'] = star_data['mjd'] % period
        
        # Create regular phase bins from 0 to period with 0.005*period interval
        phase_interval = phase_interval #* period
        phase_bins = np.arange(0, period + phase_interval, phase_interval)
        
        # Assign each point to a bin
        star_data['phase_bin'] = pd.cut(star_data['phase'], 
                                      bins=phase_bins, 
                                      labels=phase_bins[:-1] + (phase_interval/2),
                                      include_lowest=True)
        
        # For each bin, select the point with lowest magerr
        resampled_data = []
        last_valid_mag = np.nan  # Initialize with NaN instead of None
        last_valid_magerr = np.nan

        for bin_label in phase_bins[:-1]:
            bin_data = star_data[star_data['phase_bin'] == bin_label + (phase_interval/2)]
            if not bin_data.empty:
                # Select row with minimum magerr
                best_point = bin_data.loc[bin_data['magerr'].idxmin()]
                # Update last valid values
                last_valid_mag = best_point['mag']
                last_valid_magerr = best_point['magerr']
                resampled_data.append({
                    'ps1_objid': star_id,
                    'phase': bin_label + (phase_interval/2),
                    'mag': best_point['mag'],
                    'magerr': best_point['magerr'],
                    'period': period,  # Keep period information
                    'class': class_id,
                })
            else:
                # Handle missing data based on interpolation method
                if interpolation_method.lower() == "lastvalue":
                    mag_value = last_valid_mag  # Could be None if this is the first bin
                    magerr_value = last_valid_magerr
                else:
                    # Default to NaN for any other method
                    mag_value = np.nan
                    magerr_value = np.nan
                
                resampled_data.append({
                    'ps1_objid': star_id,
                    'phase': bin_label + (phase_interval/2),
                    'mag': mag_value,
                    'magerr': magerr_value,
                    'period': period,
                    'class': class_id,
                })
        
        processed_dfs.append(pd.DataFrame(resampled_data))
    
    # Combine all processed data
    result_df = pd.concat(processed_dfs, ignore_index=True)
    
    # Set multi-level index and sort
    result_df = result_df.set_index(['ps1_objid', 'phase']).sort_index()
    
    return result_df


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


def cluster_embeddings(embeddings, n_clusters=2, random_state=42):
    """
    Perform K-means clustering on star embeddings.
    
    Args:
        embeddings: numpy array of shape (n_stars, embedding_size)
        n_clusters: number of clusters (default: 2)
        random_state: random seed for reproducibility
        
    Returns:
        labels: numpy array of cluster labels
        kmeans: fitted KMeans model
    """
    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(embeddings)
    
    return labels, kmeans


def predict_lightcurve(processed_df, model_name="amazon/chronos-t5-small", context_percent=0.8, split_point=None, fig_dir='./outputs/debug/eval'):
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
    print(f"Loaded data for {len(star_ids_list)} stars")

    # Get embeddings
    embeddings, true_labels, star_ids_list = get_star_embeddings(processed_df, model_name=args.model_path)
    # print(f"Star IDs: {star_ids_list}")
    # print(f"Embeddings shape: {embeddings.shape}")
    print(f"True labels: {true_labels}\n")

    # Perform clustering
    labels, kmeans = cluster_embeddings(embeddings, n_clusters=2)
    print(f"Labels: {labels}\n")

    # Print some information about the processed data
    for star_id in star_ids_list:
        star_data = processed_df.loc[star_id]
        print(f"Star {star_id}:")
        print(f"Phase range: {star_data.index.min():.3f} to {star_data.index.max():.3f}")
        print(f"Number of points: {len(star_data)}\n")

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
    parser.add_argument("--output-dir", type=str, default="./outputs/eval/kmeans",
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