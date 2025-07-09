# You'll need to install `lsdb` package - this is a service which hosts the ZTF data we're grabbing
import math
from dask.distributed import Client
import matplotlib.pyplot as plt
from lsdb import read_hats
import pandas as pd
import numpy as np
import lsdb
from astropy.coordinates import SkyCoord
import os


def load_variable_stars(filepath):
    """
    Load variable stars catalog from file and process coordinates.
    
    Args:
        filepath: Path to the variable stars catalog file.
    
    Returns:
        pd.DataFrame: Processed variable stars catalog with RA and Dec in degrees.
    """
    # Load the dataset of variable stars
    column_names = ['ID', 'RAh', 'RAm', 'RAs', 'Decsign', 'DEm', 'DEs', 'magV', 'P', "Amp", "class", 'flag']
    varstars = pd.read_csv(filepath, header=33, sep='\s+', names=column_names)
    
    # Reformat the coordinates to allow for conversion of units
    RAhms = [f"{row.RAh}h{row.RAm}m{row.RAs}s" for _, row in varstars.iterrows()]
    Decdms = [f"{row.Decsign}d{row.DEm}m{row.DEs}s" for _, row in varstars.iterrows()]
    
    # Convert RA, Dec units and store them back in the dataframe
    coords = SkyCoord(ra=RAhms, dec=Decdms, frame='icrs')
    varstars['RAdeg'] = coords.ra.deg
    varstars['Decdeg'] = coords.dec.deg
    
    return varstars

def select_random_stars(varstars, n=5, class_ids=None, seed=None):
    """
    Select n random stars from the variable stars catalog, optionally filtering by class.
    
    Args:
        varstars: DataFrame containing variable stars.
        n: Number of random stars to select per class (if class_ids provided) or total (if class_ids is None).
        class_ids: Optional list of class identifiers to filter stars (e.g., ['1', '2', '5']).
        seed: Random seed for reproducibility.
    
    Returns:
        pd.DataFrame: Subset of randomly selected stars.
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    selected_stars_list = []

    if class_ids is None:
        class_ids = varstars['class'].unique()
    
    if class_ids is not None:
        # Process each class_id in the list
        for class_id in class_ids:
            # Filter stars by class
            class_stars = varstars[varstars['class'] == str(class_id)]
            if len(class_stars) == 0:
                print(f"No stars found with class '{class_id}'")
                continue
            
            # Adjust n if there are fewer stars than requested
            n_select = min(n, len(class_stars))
            if n_select < n:
                print(f"Warning: Only {n_select} stars available for class '{class_id}' (requested {n})")
            
            # Select random stars from the filtered dataset
            random_indices = np.random.choice(len(class_stars), size=n_select, replace=False)
            selected_stars_list.append(class_stars.iloc[random_indices])
            
            print(f"Selected {n_select} stars from class '{class_id}'")
        
        # Combine all selected stars
        if selected_stars_list:
            selected_stars = pd.concat(selected_stars_list, ignore_index=True)
        else:
            selected_stars = pd.DataFrame()
    else:
        # Original random selection from all stars
        random_indices = np.random.choice(len(varstars), size=min(n, len(varstars)), replace=False)
        # random_indices = [120620174173493992, 129030693624086743, 130603207444200557, 122802196917572800]
        selected_stars = varstars.iloc[random_indices]
    
    print(f"Selected {len(selected_stars)} stars in total:")
    print(f"Classes: {selected_stars['class'].value_counts().to_dict()}")
    
    return selected_stars

def get_ztf_lightcurve(raw_catalog, star, search_radius=2):
    """
    Retrieve ZTF light curve data for a given star.
    
    Args:
        raw_catalog: ZTF catalog from lsdb.
        star: DataFrame row containing star information.
        search_radius: Search radius in arcseconds.
    
    Returns:
        DataFrame: ZTF light curve data for the star.
    """
    with Client(n_workers=20, memory_limit="12GB") as client:
        ndf_obj = raw_catalog.cone_search(
            star['RAdeg'],  # RA
            star['Decdeg'],  # DEC
            search_radius,  # radius around coordinates in arcseconds
        ).compute()
    
    # Calculate flux from magnitude
    if not ndf_obj.empty:
        # Flux in microJy
        ndf_obj['flux'] = 3.631e9 * 10**(ndf_obj['mag'] / -2.5)
        # Flux uncertainty in microJy
        ndf_obj['fluxerr'] = ndf_obj['flux'] * np.log(10) * ndf_obj['magerr'] / 2.5
        # Add star metadata
        ndf_obj['star_id'] = star['ID']
        ndf_obj['star_class'] = star['class']
        ndf_obj['period'] = star['P']
    
    return ndf_obj

def plot_lightcurve_by_band(ndf_obj, star, output_dir='outputs/lsdb'):
    """
    Plot the light curve by band and save to the appropriate class folder.
    
    Args:
        ndf_obj: DataFrame containing light curve data.
        star: Series containing star information.
        output_dir: Base directory for saving plots.
    """
    if ndf_obj.empty:
        print(f"No data found for star {star['ID']}")
        return
    
    # Color mapping for bands
    band_to_color = {
        "g": "mediumaquamarine",
        "r": "crimson",
        "i": "gold"
    }
    
    # Create output directory for class if it doesn't exist
    class_dir = os.path.join(output_dir, str(star['class']))
    os.makedirs(class_dir, exist_ok=True)
    
    # Get the period for phasing
    P = float(star['P'])
    print(f"Star {star['ID']} has period {P} days and class {star['class']}")
    
    # Get object ID for filename
    if 'ps1_objid' in ndf_obj.columns and not ndf_obj.empty:
        obj_id = ndf_obj['ps1_objid'].iloc[0]
    else:
        obj_id = f"unknown_{star['ID']}"
    
    # Plot separately for each band
    for band in ['r', 'g', 'i']:
        obs_in_band = ndf_obj.loc[ndf_obj['band'] == band]
        if len(obs_in_band) == 0:
            print(f"No data for band {band} for star {star['ID']}")
            continue
            
        # Calculate phase
        phase = (obs_in_band["mjd"].to_numpy() % P)
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(phase, obs_in_band["mag"], yerr=obs_in_band["magerr"], 
                     label=band, color=band_to_color[band], fmt='x')
        
        plt.title(f"{star['ID']} - Class: {star['class']} - objid: {obj_id}")
        plt.legend()
        plt.xlabel("Phase")
        plt.ylabel("Magnitude")
        # plt.gca().invert_yaxis()
        if not os.path.exists(f'{class_dir}'):
            os.makedirs(f'{class_dir}', exist_ok=True)
        plt.savefig(f'{class_dir}/{obj_id}_{band}.png')
        plt.close()

def process_stars(random_stars, raw_catalog, plot_dir=None):
    """
    Process a set of stars: get light curves and save plots.
    
    Args:
        random_stars: DataFrame containing selected stars.
        raw_catalog: ZTF catalog from lsdb.
    
    Returns:
        pd.DataFrame: Combined light curve data for all processed stars.
    """
    all_data = []
    
    for idx, star in random_stars.iterrows():
        print(f"Processing star {idx+1}/{len(random_stars)}: ID={star['ID']}, Class={star['class']}")
        
        # Get ZTF light curve
        ndf_obj = get_ztf_lightcurve(raw_catalog, star)
        
        if not ndf_obj.empty:
            # Plot and save light curves
            if plot_dir is not None:
              plot_lightcurve_by_band(ndf_obj, star, output_dir=plot_dir)
            
            # Add to collected data
            all_data.append(ndf_obj)
        else:
            print(f"No data found for star {star['ID']}")
    
    # Combine all light curves into one DataFrame if we have any data
    if all_data:
        combined_df = pd.concat([pd.DataFrame(d) for d in all_data], ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()

def save_merged_csv(df, output_file):
    """
    Save the DataFrame to a CSV file, merging with existing file if present.
    
    Args:
        df: DataFrame containing light curve data.
        output_file: Path to save the CSV file.
    """
    # Check if the file already exists
    if os.path.exists(output_file):
        print(f"Found existing CSV file at {output_file}, merging...")
        
        try:
            # Read existing file
            existing_df = pd.read_csv(output_file)
            
            # Identify potential duplicates
            # Assuming the combination of 'star_id', 'mjd', and 'band' uniquely identifies an observation
            if all(col in df.columns for col in ['star_id', 'mjd', 'band']):
                # Create a unique identifier for each row
                if all(col in existing_df.columns for col in ['star_id', 'mjd', 'band']):
                    existing_df['_merge_key'] = existing_df['star_id'].astype(str) + '_' + \
                                              existing_df['mjd'].astype(str) + '_' + \
                                              existing_df['band'].astype(str)
                    
                    df['_merge_key'] = df['star_id'].astype(str) + '_' + \
                                     df['mjd'].astype(str) + '_' + \
                                     df['band'].astype(str)
                    
                    # Filter out rows already in the existing file
                    new_rows = df[~df['_merge_key'].isin(existing_df['_merge_key'])]
                    new_rows = new_rows.drop('_merge_key', axis=1)
                    
                    # Remove temporary column from existing_df
                    existing_df = existing_df.drop('_merge_key', axis=1)
                    
                    # Merge the DataFrames
                    merged_df = pd.concat([existing_df, new_rows], ignore_index=True)
                else:
                    # If existing file doesn't have expected columns, simply append
                    merged_df = pd.concat([existing_df, df], ignore_index=True)
            else:
                # If the new data doesn't have expected columns, simply append
                merged_df = pd.concat([existing_df, df], ignore_index=True)
                
            # Save the merged DataFrame
            merged_df.to_csv(output_file, index=False)
            print(f"Merged data saved to {output_file}")
            
            # Return summary stats
            return {
                'existing_rows': len(existing_df),
                'new_rows': len(df),
                'merged_rows': len(merged_df),
                'added_rows': len(merged_df) - len(existing_df)
            }
        except Exception as e:
            print(f"Error merging CSV files: {e}")
            print("Saving as new file instead...")
            df.to_csv(output_file, index=False)
            return {'new_file': len(df)}
    else:
        # If file doesn't exist, just save the current DataFrame
        df.to_csv(output_file, index=False)
        print(f"New data saved to {output_file}")
        return {'new_file': len(df)}

def main(varstars_file="./src/tasks/download/catalog/CSDR1_varstars.txt",
         output_dir="./outputs/download/lsdb",
         output_file_name="combined_lightcurves.csv",
         ztf_source="https://data.lsdb.io/hats/ztf_dr14/ztf_source",
         n=20,
         class_ids=None,
         seed=42,
         plot=False):
    """Main function to run the variable star analysis.
    
    Args:
        varstars_file (str): Path to the variable stars catalog file
        output_dir (str): Directory to save output files
        output_file_name (str): Name of the output CSV file
        ztf_source (str): URL for the ZTF data source
        n (int): Number of stars to select per class
        class_ids (list): List of class IDs to filter by (None for all classes)
        seed (int): Random seed for star selection
    """
    # Path configurations
    plot_dir = f"{output_dir}/plots"
    output_file = f"{output_dir}/{output_file_name}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load ZTF catalog
    print("Loading ZTF catalog...")
    raw_catalog = read_hats(ztf_source)
    
    # Load and process variable stars catalog
    print("Loading variable stars catalog...")
    varstars = load_variable_stars(varstars_file)
    
    # Select random stars
    print("Selecting random stars...")
    random_stars = select_random_stars(varstars, n=n, class_ids=class_ids, seed=seed)

    # Process each star and collect data
    print("Processing selected stars...")
    combined_lightcurve_data = process_stars(random_stars, raw_catalog, plot_dir=plot_dir if plot else None)
    
    # Save combined data to CSV
    if not combined_lightcurve_data.empty:
        merge_stats = save_merged_csv(combined_lightcurve_data, output_file)
        
        print("\nCSV merge statistics:")
        for key, value in merge_stats.items():
            print(f"  {key}: {value}")
        
        # Basic summary of current data
        grouped = combined_lightcurve_data.groupby(['star_id', 'star_class'])
        summary = grouped.size().reset_index(name='data_points')
        print("\nSummary of newly collected data:")
        print(summary)
        
        # If we successfully merged, provide summary of all data
        if os.path.exists(output_file) and 'merged_rows' in merge_stats:
            all_data = pd.read_csv(output_file)
            print("\nSummary of all data in CSV:")
            all_grouped = all_data.groupby(['star_id', 'star_class'])
            all_summary = all_grouped.size().reset_index(name='data_points')
            print(all_summary)
    else:
        print("No data was collected for any of the selected stars.")

# Run the main function when executed as a script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and process variable star data")
    parser.add_argument("--varstars-file", default="./src/tasks/download/catalog/CSDR1_varstars.txt", 
                        help="Path to the variable stars catalog file")
    parser.add_argument("--output-dir", default="./outputs/download/lsdb", 
                        help="Directory to save output files")
    parser.add_argument("--output-file-name", default="lightcurves.csv",
                        help="Name of the output CSV file")
    parser.add_argument("--ztf-source", default="https://data.lsdb.io/hats/ztf_dr14/ztf_source", 
                        help="URL for the ZTF data source")
    parser.add_argument("--n", type=int, default=20, 
                        help="Number of stars to select per class")
    parser.add_argument("--class-ids", nargs='+', default=None,
                        help="List of class IDs to filter by (e.g., --class-ids 1 2 5)")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed for star selection")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Plot light curves")
    
    args = parser.parse_args()
    
    main(
        varstars_file=args.varstars_file,
        output_dir=args.output_dir,
        output_file_name=args.output_file_name,
        ztf_source=args.ztf_source,
        n=args.n,
        class_ids=args.class_ids,
        seed=args.seed,
        plot=args.plot
    )
