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


def seed_everything(seed=42):
    """
    Seed everything for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_star_ids(data_path='/home/magics/hdd/sky_ws/ebsim_ws/data/lsdb/combined_lightcurves.csv', 
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

def create_indexed_star_df(star_ids, band, data_path='/home/magics/hdd/sky_ws/ebsim_ws/data/lsdb/combined_lightcurves.csv'):
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

def plot_raw_lightcurves(star_df, output_dir):
    """
    Plot raw light curves (magnitude vs. time) for each star,
    differentiating between good data (catflags=0) and bad data (catflags≠0).
    
    Args:
        star_df: DataFrame with two-level index (ps1_objid, datetime)
        output_dir: Directory to save plots
    """
    plot_dir = os.path.join(output_dir, "raw_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Get unique star IDs from the first level of the index
    star_ids = star_df.index.get_level_values(0).unique()
    
    print(f"Plotting raw light curves for {len(star_ids)} stars...")
    
    for star_id in star_ids:
        # Get data for this star and reset index to access catflags
        star_data = star_df.loc[star_id].reset_index()
        
        # Get class from the first row
        class_id = star_data['star_class'].iloc[0]
        
        # Separate good and bad data points
        good_data = star_data[star_data['catflags'] == 0]
        bad_data = star_data[star_data['catflags'] != 0]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot good data points (blue)
        if not good_data.empty:
            plt.errorbar(good_data['mjd'] % good_data['period'].iloc[0], good_data['mag'], 
                         yerr=good_data['magerr'], fmt='o', color='blue', alpha=0.7, 
                         markersize=4, capsize=2, label='Good data (catflags=0)')
        
        # Plot bad data points (red)
        if not bad_data.empty:
            plt.errorbar(bad_data['mjd'] % bad_data['period'].iloc[0], bad_data['mag'], 
                         yerr=bad_data['magerr'], fmt='x', color='red', alpha=0.7, 
                         markersize=4, capsize=2, label='Bad data (catflags≠0)')
        
        # Set labels and title
        plt.xlabel('Phase (days)')
        plt.ylabel('Magnitude')
        plt.title(f"Star {star_id} (Class {class_id}) - Raw Light Curve")
        
        # Add legend if both types of data are present
        if not good_data.empty or not bad_data.empty:
            plt.legend()
        
        # Save the plot
        class_dir = os.path.join(plot_dir, f"class_{class_id}")
        os.makedirs(class_dir, exist_ok=True)
        plt.savefig(os.path.join(class_dir, f"raw_star_{star_id}.png"), dpi=150)
        plt.close()
    
    print(f"Raw light curve plots saved to {plot_dir}")

def plot_processed_lightcurves(processed_df, output_dir):
    """
    Plot processed light curves for each star.
    
    Args:
        processed_df: DataFrame with processed light curve data, having multi-index (ps1_objid, phase)
        output_dir: Directory to save plots
    """
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Group by star ID (first level of the index)
    star_ids = processed_df.index.get_level_values(0).unique()
    
    print(f"Plotting light curves for {len(star_ids)} stars...")
    
    for star_id in star_ids:
        # Get data for this star
        star_data = processed_df.loc[star_id]
        
        # Get class from the first row
        class_id = star_data['class'].iloc[0]
        period = star_data['period'].iloc[0]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Phase on x-axis, magnitude on y-axis
        plt.errorbar(star_data.index, star_data['mag'], yerr=star_data['magerr'], 
                     fmt='o', color='blue', alpha=0.7, markersize=4, capsize=2)
        
        # Add a second cycle of points (phase + period) for better visualization
        # plt.errorbar(star_data.index + period, star_data['mag'], yerr=star_data['magerr'], 
                    #  fmt='o', color='blue', alpha=0.4, markersize=4, capsize=2)
        
        # Set labels and title
        plt.xlabel('Phase')
        plt.ylabel('Magnitude')
        plt.title(f"Star ID: {star_id}, Class: {class_id}, Period: {period:.4f} days")
        
        # Invert y-axis (astronomical convention: brighter stars have lower magnitudes)
        # plt.gca().invert_yaxis()
        
        # Save the plot
        class_dir = os.path.join(plot_dir, f"class_{class_id}")
        os.makedirs(class_dir, exist_ok=True)
        plt.savefig(os.path.join(class_dir, f"star_{star_id}.png"), dpi=150)
        plt.close()
    
    print(f"Light curve plots saved to {plot_dir}")

def filter_bad_flags(star_df):
    """
    Filter out rows with non-zero catflags (bad data flags).
    
    Args:
        star_df: DataFrame with two-level index (ps1_objid, datetime)
        
    Returns:
        pd.DataFrame: Filtered DataFrame with bad flag data removed
    """
    print("Filtering out rows with non-zero catflags...")
    
    # Reset index to access the catflags column
    star_df_reset = star_df.reset_index()
    
    # Find indices of good data (catflags == 0)
    good_idx = np.where(np.array(star_df_reset['catflags'])==0)[0]
    
    # Create filtered DataFrame
    star_df_filtered = star_df_reset.iloc[good_idx]
    
    # Set multi-level index and sort
    star_df_filtered = star_df_filtered.set_index(['ps1_objid', 'datetime']).sort_index()
    
    print(f"Removed {len(star_df_reset) - len(star_df_filtered)} rows with bad flags out of {len(star_df_reset)} total rows.")
    
    return star_df_filtered

def main(args):
    """
    Main function to preprocess light curve data.
    
    Args:
        args: Command line arguments containing:
            input_file: Path to the input CSV file with light curve data
            output_dir: Directory to save processed data
            output_filename: Name of output CSV file
            band: Band to process (g, r, or i)
            class_ids: List of class IDs to filter by
            stars_per_class: Number of stars to process per class
            phase_interval: Interval for phase bins
            interpolation_method: Method for handling missing values
    """
    # Initialize random seed
    seed_everything()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_filename)
    
    # Get star IDs
    if args.class_ids:
        print(f"Getting star IDs for classes {args.class_ids} with {args.stars_per_class} stars per class...")
        star_ids = get_star_ids(data_path=args.input_file, class_ids=args.class_ids, 
                               stars_per_class=args.stars_per_class, band=args.band)
        # star_ids = [120620174173493992, 129030693624086743,  130603207444200557, 122802196917572800]
    else:
        print("Getting star IDs for all classes...")
        star_ids = get_star_ids(data_path=args.input_file, band=args.band)

    # Load the data
    print(f"Creating indexed DataFrame for {len(star_ids)} stars...")
    star_df = create_indexed_star_df(star_ids, band=args.band, data_path=args.input_file)
    
    # Filter out bad flag data
    filtered_star_df = filter_bad_flags(star_df)

    # Process the data
    print(f"Processing light curve data...")
    processed_df = process_lightcurve_data(filtered_star_df, interpolation_method="LastValue", 
                                         phase_interval=0.0025)
    
    # Save the processed data
    processed_df.to_csv(output_path)
    print(f"Processed data saved to: {output_path}")
    
    # Plot the processed data if requested
    if args.plot:
        few_stars = star_df.loc[star_df.index.get_level_values(0).unique()[:10]]
        plot_raw_lightcurves(few_stars, args.output_dir)
        few_processed_stars = processed_df.loc[processed_df.index.get_level_values(0).unique()[:10]]
        plot_processed_lightcurves(few_processed_stars, args.output_dir)

    # Print some information about the processed data
    print("\nSummary of processed data:")
    for star_id in star_ids:  # Show info for first 5 stars only
        if star_id in processed_df.index:
            star_data = processed_df.loc[star_id]
            print(f"Star {star_id}:")
            print(f"  Phase range: {star_data.index.min():.3f} to {star_data.index.max():.3f}")
            print(f"  Number of points: {len(star_data)}")
    
    # Print overall stats
    print(f"\nTotal stars processed: {len(processed_df.index.get_level_values(0).unique())}")
    print(f"Total data points: {len(processed_df)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess variable star light curve data")
    parser.add_argument("--input-file", default="./data/download/lsdb/lightcurves.csv", 
                        help="Path to the input CSV file with light curve data")
    parser.add_argument("--output-dir", default="./data/processed", 
                        help="Directory to save processed data")
    parser.add_argument("--output-filename", default="processed_lightcurves.csv",
                        help="Name of output CSV file")
    parser.add_argument("--band", default="g", choices=["g", "r", "i"],
                        help="Photometric band to process (g, r, or i)")
    parser.add_argument("--class-ids", nargs='+', type=int, default=None,
                        help="List of class IDs to filter by (e.g., --class-ids 1 5)")
    parser.add_argument("--stars-per-class", type=int, default=100, 
                        help="Number of stars to process per class")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots of the processed light curves")

    args = parser.parse_args()
    
    # Call main with the args object directly
    main(args)

