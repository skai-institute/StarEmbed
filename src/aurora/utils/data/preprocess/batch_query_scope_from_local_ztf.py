import os
import glob
import pandas as pd
import re
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import time
from datasets import Dataset, concatenate_datasets
from functools import partial
import multiprocessing
import traceback
from datasets import Dataset, Features, Value, Sequence
import gc
from multiprocessing import Pool

"""
This script is to batch query the variable light curves identified through automatic labeling by the scope project from the local full ZTF dataset. It takes about 5 minutes to process 10 fiedls of scope data (200 fields in total) to generate 1.4m light curves of all bands.

The queried light curves are saved as a huggingface dataset. They are the raw data for the later fine-tuning phase of the universal forecaster.
"""


class ZTFCatalogSearcher:
    def __init__(self, parquet_base_dir, verbose=True):
        """
        Initialize the ZTF catalog searcher
        """
        self.parquet_base_dir = parquet_base_dir
        self.verbose = verbose
        self.field_pattern_map = {}
        self._build_path_patterns()
        # 1) First define the schema for each per‐band “match_dict”
        match_schema = Features({
            # derived fields
            "avg_mag":        Value("float64"),
            "avg_magerr":     Value("float64"),
            "band":           Value("string"),
            # the original LST columns from parquet:
            "catflags":       Sequence(feature=Value("int32")),
            "clrcoeff":       Sequence(feature=Value("float64")),
            "fieldid":        Value("int16"),
            "filterid":       Value("int8"),
            "hmjd":           Sequence(feature=Value("float64")),
            "mag":            Sequence(feature=Value("float64")),
            "magerr":         Sequence(feature=Value("float64")),
            "nepochs":        Value("int64"),
            "objectid":       Value("int64"),
            "objra":          Value("float32"),
            "objdec":         Value("float32"),
            "rcid":           Value("int8"),
            # the extra columns from scope:
            "separation_arcsec": Value("float64"),
            "scope_id":          Value("int64"),
            "scope_period":         Value("float64")
        })

        # 2) Now build the top‐level schema, including a nested dict for your g/r/i bands
        self.schema = Features({
            "objectid":    Value("int64"),
            "bands_data":  {
                "g": match_schema,
                "r": match_schema,
                "i": match_schema,
            },
            "avg_magerr":  Value("float64"),
            "avg_mag":     Value("float64"),
        })
        
    
    def _build_path_patterns(self):
        """Build path patterns for efficiently finding parquet files"""
        if self.verbose:
            print("Building path patterns for parquet files...")
            start_time = time.time()
        
        # Create a mapping of field_id to pattern template
        fields = glob.glob(os.path.join(self.parquet_base_dir, "**/field*"), recursive=True)
        
        for field_dir in fields:
            # Extract field ID from the directory name
            match = re.search(r'field0*(\d+)', os.path.basename(field_dir))
            if match:
                field_id = match.group(1)
                # Store the directory path for this field
                self.field_pattern_map[field_id] = field_dir
        
        if self.verbose:
            print(f"Indexed {len(self.field_pattern_map)} fields")
            print(f"Path patterns built in {time.time() - start_time:.2f} seconds")
    
    def _extract_field_id_from_csv(self, csv_path):
        """Extract field ID from CSV filename"""
        match = re.search(r'field_(\d+)_vs\.csv', os.path.basename(csv_path))
        if match:
            return match.group(1)
        return None
    
    def _get_parquet_path(self, field_id, ccd_id, quad_id):
        """
        Get the parquet file path for specific field, CCD, and quad IDs
        """
        # Convert IDs to strings
        field_id_str = str(field_id)
        
        # Handle potential formats
        ccd_id_str = str(int(float(ccd_id)))  # Handle .0 suffix
            
        quad_id_str = str(int(float(quad_id)))  # Handle .0 suffix

        
        # Get the field directory
        field_dir = self.field_pattern_map[field_id_str]
        
        # Try all pattern variations
        all_matching_files = []
        # for the pattern, if ccd_id_str is single digit, add a leading 0
        if len(ccd_id_str) == 1:
            ccd_id_str = f"0{ccd_id_str}"
        pattern = os.path.join(field_dir, f"ztf_000{field_id_str}_z*_c{ccd_id_str}_q{quad_id_str}_*.parquet")
        matching_files = glob.glob(pattern)
        all_matching_files.extend(matching_files)
        
        # Remove duplicates
        unique_files = list(set(all_matching_files))

        assert len(unique_files) > 0, f"No files found for field={field_id_str}, ccd={ccd_id_str}, quad={quad_id_str}"
        
        return unique_files
    
    def _extract_band_from_filename(self, filename):
        """Extract band information from parquet filename"""
        match = re.search(r'_z([a-z])_', filename)
        if match:
            return match.group(1)
        return "unknown"
    
    def process_csv_file(self, args):
        """
        Process a single CSV file containing star coordinates
        """
        csv_path, radius_arcsec, csv_index, total_csvs, output_dir = args
        
        # Print start message
        print(f"[{csv_index + 1}/{total_csvs}] Starting {os.path.basename(csv_path)}")
        start_time = time.time()
        
        # Extract field ID from CSV filename
        csv_field_id = self._extract_field_id_from_csv(csv_path)
        if not csv_field_id:
            print(f"Could not extract field ID from {csv_path}")
            return None
        
        # Read star coordinates from CSV
        try:
            stars_df = pd.read_csv(csv_path)
            size_bytes = os.path.getsize(csv_path)
            size_mb = size_bytes / (1024 * 1024)
            print(f"[{csv_index + 1}/{total_csvs}] {os.path.basename(csv_path)}: Loaded {len(stars_df)} stars, {size_mb:.2f} MB")
        except Exception as e:
            print(f"[{csv_index + 1}/{total_csvs}] Error reading {csv_path}: {e}")
            return None
        
        # Check required columns
        required_cols = ['ra', 'dec', 'ccd', 'quad']  # Adjust based on your column names
        if not all(col in stars_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in stars_df.columns]
            print(f"[{csv_index + 1}/{total_csvs}] CSV {csv_path} missing required columns: {missing}")
            return None
        
        # Group stars by ccd and quad to process each group with the same parquet file
        all_matches = []
        
        # Group the stars
        groups = stars_df.groupby(['ccd', 'quad'])
        print(f"Grouped stars into {len(groups)} CCD/quad combinations")

        proper_motion_cnt = 0
        lc_cnt = 0
        
        # Process each group
        for (ccd_id, quad_id), group_df in groups:
            # print(f"Processing group with CCD={ccd_id}, quad={quad_id}, containing {len(group_df)} stars")
            
            # Get parquet files for this group (only once)
            parquet_files = self._get_parquet_path(csv_field_id, ccd_id, quad_id)
            
            assert len(parquet_files) > 0

            # Create SkyCoord for all stars in this group at once
            group_coords = SkyCoord(ra=group_df['ra'].values*u.degree, 
                                dec=group_df['dec'].values*u.degree)
            
            # Process each parquet file
            obj_curves = {} # key is the object id, value is a dict with band as key and data as dict as value
            for parquet_file in parquet_files:
                try:
                    # print(f"Reading parquet file: {os.path.basename(parquet_file)}")
                    # Extract band information
                    band = self._extract_band_from_filename(parquet_file)
                    
                    # Read parquet file (once for all stars in this group)
                    df = pd.read_parquet(parquet_file)
                    

                    if df.empty:
                        print(f"[WARN] parquet {parquet_file} is empty—skipping")
                        continue
                    
                    # Create catalog coordinates once
                    catalog_coords = SkyCoord(ra=df['objra'].values*u.degree, 
                                            dec=df['objdec'].values*u.degree)
                    
                    # Match all stars in this group to the catalog at once
                    idx, sep2d, _ = group_coords.match_to_catalog_sky(catalog_coords)
                    
                    # Find matches within the radius
                    within_radius = sep2d < (radius_arcsec * u.arcsec)

                    proper_motion_cnt += (len(group_df) - sum(within_radius))

                    lc_cnt += sum(within_radius)
                    
                    # Process matches
                    for i, (is_match, star_idx, separation) in enumerate(zip(within_radius, idx, sep2d)):
                        if is_match:
                            # Get original star data
                            star_row = group_df.iloc[i]
                            
                            # Get the matching catalog entry
                            match_dict = df.iloc[star_idx].to_dict()

                            # good_idx = np.where(np.array(match_dict['catflags'])==0)[0]
                            # match_dict['hmjd'] = match_dict['hmjd'][good_idx]
                            # match_dict['mag'] = match_dict['mag'][good_idx]
                            # match_dict['magerr'] = match_dict['magerr'][good_idx]
                            # match_dict['clrcoeff'] = match_dict['clrcoeff'][good_idx]
                            # # remove catflags
                            # del match_dict['catflags']

                            if len(match_dict['mag']) == 0:
                                print(f"[WARN] {i}/{len(group_df)} no good mags for objectid={match_dict['objectid']}, skipping")
                                continue

                            objectid = match_dict['objectid']
                            
                            # Add separation and source info
                            match_dict['separation_arcsec'] = separation.to(u.arcsec).value
                            # match_dict['catalog_file'] = os.path.basename(parquet_file)
                            match_dict['band'] = band

                            match_dict['avg_magerr'] = np.mean(match_dict['magerr'])
                            match_dict['avg_mag'] = np.mean(match_dict['mag'])
                            
                            # # Add original CSV data
                            # for col in star_row.index:
                            #     match_dict[f'scope_{col}'] = star_row[col]

                            match_dict['scope_id'] = star_row['_id']
                            match_dict['scope_period'] = star_row['period']

                            if objectid in obj_curves:
                                obj_curves[objectid][band] = match_dict
                            else:
                                obj_curves[objectid] = {band: match_dict}


                except Exception as e:
                    print(f"Error processing parquet file {parquet_file}: {e}")
                    traceback.print_exc()
                del df, catalog_coords, idx, sep2d  # drop references
                gc.collect()                        # force a GC pass

            for objectid, bands_data in obj_curves.items():
                avg_magerr = np.mean([bands_data[band]['avg_magerr'] for band in bands_data])
                avg_mag = np.mean([bands_data[band]['avg_mag'] for band in bands_data]) 
                bands_data_sorted = {}
                for k in ['g', 'r', 'i']:
                    if k in bands_data:
                        bands_data_sorted[k] = bands_data[k]
                    else:
                        bands_data_sorted[k] = None

                all_matches.append({'objectid': objectid, 'bands_data': bands_data_sorted, 'avg_magerr': avg_magerr, 'avg_mag': avg_mag})

        all_matches = pd.DataFrame(all_matches)
        
        elapsed_time = time.time() - start_time
        # the number of matches might be less than the number of stars in the query because some stars have "proper motion" between two catalogue snapshots
        motion_percentage = proper_motion_cnt / len(stars_df)
        print(f"[{csv_index + 1}/{total_csvs}] {os.path.basename(csv_path)}: Finished processing {len(stars_df)} stars in {elapsed_time:.2f}s, found {len(all_matches)} matches, {motion_percentage:.2f}% of light curves have proper motion")
        
        if len(all_matches) > 0:
            shard_path = os.path.join(output_dir, f"shard_{csv_field_id}")
            ds = Dataset.from_pandas(all_matches, features=self.schema)
            ds.save_to_disk(shard_path)
            # cleanup
            del ds, all_matches
            gc.collect()
            return shard_path, lc_cnt
        else:
            print(f"[{csv_index + 1}/{total_csvs}] {os.path.basename(csv_path)}: No matches found")
            return None, lc_cnt
    
    def batch_search_from_csvs(self, csv_pattern, radius_arcsec=2.0, output_dir=None, 
                              num_workers=4):
        """
        Search for sources using coordinates from multiple CSV files
        """
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            # Path for the final dataset
            dataset_path = os.path.join(output_dir, "ztf_matches_dataset_full3")
        
        # Find all CSV files
        csv_files = glob.glob(csv_pattern)
        if not csv_files:
            print(f"No CSV files found matching pattern: {csv_pattern}")
            return None
        
        print(f"Found {len(csv_files)} CSV files to process")
        
        # Prepare arguments for CSV processing
        total_csvs = len(csv_files)
        process_args = [(csv_file, radius_arcsec, i, total_csvs, output_dir) 
                        for i, csv_file in enumerate(csv_files)]
        
        # Track datasets for each chunk
        overall_start_time = time.time()
        total_csvs = len(process_args)
        completed = 0
        tot_lc_cnt = 0
        chunk_datasets = []
        shard_paths = []


        
        # Process CSV files in parallel
        # Use Pool with maxtasksperchild=1
        with Pool(processes=num_workers, maxtasksperchild=1) as pool:
            # imap_unordered yields results as soon as they arrive, in any order
            for shard_path, lc_cnt in pool.imap_unordered(self.process_csv_file, process_args):
                completed += 1
                tot_lc_cnt += lc_cnt

                if shard_path:
                    shard_paths.append(shard_path)
                    print(f"[{completed}/{total_csvs}] saved shard: {shard_path}")
                else:
                    print(f"[{completed}/{total_csvs}] no matches")

        overall_elapsed_time = time.time() - overall_start_time
        print(f"Total processing time: {overall_elapsed_time:.2f} seconds")
        print(f"Total light curves found: {tot_lc_cnt}")
        
        # # Combine all chunks into a single dataset
        # if chunk_datasets:
        #     print(f"Combining {len(chunk_datasets)} dataset chunks...")
        #     final_dataset = concatenate_datasets(chunk_datasets)
            
        #     # Save the final dataset with multiple shards
        #     if output_dir:
        #         print(f"Saving dataset with {len(final_dataset)} examples to {dataset_path}...")
        #         # Save with multiple shards
        #         final_dataset.save_to_disk(
        #             dataset_path,
        #             num_proc=min(num_workers, 4),  # Use multiple processes for saving
        #             max_shard_size="100MB"  # Adjust shard size as needed
        #         )
            
            
            
        #     return final_dataset
        # else:
        #     print("No matches found for any stars")
        #     return None

# Example usage
if __name__ == "__main__":
    # Set parameters
    parquet_dir = "/projects/b1094/stroh/software/catalogs/ztf/lc_dr23/"
    # csv_pattern = "/projects/b1094/rehemtulla/SkAI/SCoPe/*/*/field_*_vs.csv"
    csv_pattern = "/projects/b1094/rehemtulla/SkAI/SCoPe/*/*/field_806_vs.csv"
    output_dir = "/projects/p32795/weijian/queried_scope_from_ztf/matched_data_full2"
    
    # Initialize the searcher
    searcher = ZTFCatalogSearcher(parquet_dir, verbose=True)
    
    # Determine optimal parallelism strategy
    total_cpus = multiprocessing.cpu_count()
    num_workers = max(2, total_cpus - 2)
    num_workers = 1
    
    print(f"Using {num_workers} workers")
    
    # Run the batch search
    searcher.batch_search_from_csvs(
        csv_pattern=csv_pattern,
        radius_arcsec=2.0,
        output_dir=output_dir,
        num_workers=num_workers,
    )