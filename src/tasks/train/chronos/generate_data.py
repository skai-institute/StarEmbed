from pathlib import Path
from typing import List, Union

import numpy as np
from gluonts.dataset.arrow import ArrowWriter
import pandas as pd


def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    compression: str = "lz4",
):
    """
    Store a given set of series into Arrow format at the specified path.

    Input data can be either a list of 1D numpy arrays, or a single 2D
    numpy array of shape (num_series, time_length).
    """
    assert isinstance(time_series, list) or (
        isinstance(time_series, np.ndarray) and
        time_series.ndim == 2
    )

    # Set an arbitrary start time
    start = np.datetime64("2000-01-01 00:00", "s")

    dataset = [
        {"start": start, "target": ts} for ts in time_series
    ]

    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=path,
    )


def scale_time_series(data, scale_factor=2.0):
    """
    Scale time series data while preserving the maximum value and shape.
    
    Parameters:
    -----------
    data : array-like
        The input time series data
    scale_factor : float
        Factor by which to scale the "gaps" (differences from maximum)
        Default is 2.0 (double the gaps)
    
    Returns:
    --------
    scaled_data : numpy.ndarray
        Scaled time series with same shape and maximum value
    """
    # Convert to numpy array if not already
    data = np.array(data)
    
    # Find the maximum value
    max_val = np.max(data)
    
    # Calculate gaps (differences from maximum)
    gaps = max_val - data
    
    # Scale the gaps
    scaled_gaps = gaps * scale_factor
    
    # Calculate new values by subtracting scaled gaps from maximum
    scaled_data = max_val - scaled_gaps
    
    return scaled_data


def scale_EB_lcs_targets(EB_lcs, scale_factor=2.0):
    """
    Apply scaling to target values in the EB_lcs array.
    
    Parameters:
    -----------
    EB_lcs : numpy.ndarray
        Array of shape (n_series, n_points, 2) where:
        - n_series is the number of time series
        - n_points is the number of points per time series
        - Last dimension has [date, target]
    scale_factor : float
        Factor by which to scale the gaps
        
    Returns:
    --------
    scaled_EB_lcs : numpy.ndarray
        Array with same shape as input, with target values scaled
    """
    # Create a copy to avoid modifying the original data
    scaled_EB_lcs = EB_lcs.copy()
    
    # Loop through each time series
    for i in range(scaled_EB_lcs.shape[0]):
        # Extract the target values for this time series
        targets = scaled_EB_lcs[i, :, 1]
        
        # Apply scaling to the targets
        scaled_targets = scale_time_series(targets, scale_factor)
        
        # Update the target values in the copy
        scaled_EB_lcs[i, :, 1] = scaled_targets
    
    return scaled_EB_lcs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert processed light curve data to GluonTS arrow format")
    parser.add_argument("--input-files", nargs='+', required=True,
                        help="Path(s) to input CSV file(s) with processed light curve data")
    parser.add_argument("--output-file", required=True,
                        help="Path to output arrow file")
    parser.add_argument("--fold", action="store_true",
                        help="Fold the light curves")
    args = parser.parse_args()
  
    # Container for all light curves
    all_lc_list = []
    for input_file in args.input_files:
        print(f"Processing file: {input_file}")
        processed_df = pd.read_csv(input_file)
        if args.fold:
            processed_df = processed_df.set_index(['ps1_objid', 'phase']).sort_index()
        else:
            processed_df = processed_df.set_index(['ps1_objid', 'mjd']).sort_index()
        mag_arrays = processed_df.groupby('ps1_objid')['mag'].apply(np.array).to_dict()
        lc_list = list(mag_arrays.values())
        print(f"  Found {len(lc_list)} light curves")
        all_lc_list.extend(lc_list)
    print(f"Total light curves: {len(all_lc_list)}")
    
    # Get the min and max of the scaled data, ignoring NaN values
    min_list = [np.nanmin(x/np.nanmean(x)) for x in all_lc_list]
    max_list = [np.nanmax(x/np.nanmean(x)) for x in all_lc_list]
    min_min = np.nanmin(min_list)
    max_max = np.nanmax(max_list)
    print("min_min after mean scaling:", min_min, "max_max after mean scaling:", max_max)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to GluonTS arrow format
    convert_to_arrow(args.output_file, time_series=all_lc_list)
    print(f"Saved arrow data to: {args.output_file}")