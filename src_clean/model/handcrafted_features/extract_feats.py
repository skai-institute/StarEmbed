from datasets import load_from_disk
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import numpy as np
import light_curve
import argparse
import time
import os

# Install FATS package NOT via pip, but rather via cloning and installing
# https://github.com/nabeelre/FATS.git
# This fork migrates everything to Python 3.
import FATS

# Suppress mathematical warnings from FATS
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


FATS_feature_names = [
    'PeriodLS', 'Period_fit', 'Psi_CS', 'Psi_eta', 'Autocor_length', 'Con', 'PairSlopeTrend',
    'Freq1_harmonics_amplitude_0', 'Freq1_harmonics_amplitude_1',
    'Freq1_harmonics_amplitude_2', 'Freq1_harmonics_amplitude_3',
    'Freq2_harmonics_amplitude_0', 'Freq2_harmonics_amplitude_1',
    'Freq2_harmonics_amplitude_2', 'Freq2_harmonics_amplitude_3',
    'Freq3_harmonics_amplitude_0', 'Freq3_harmonics_amplitude_1',
    'Freq3_harmonics_amplitude_2', 'Freq3_harmonics_amplitude_3',
    'Freq1_harmonics_rel_phase_0', 'Freq1_harmonics_rel_phase_1',
    'Freq1_harmonics_rel_phase_2', 'Freq1_harmonics_rel_phase_3',
    'Freq2_harmonics_rel_phase_0', 'Freq2_harmonics_rel_phase_1',
    'Freq2_harmonics_rel_phase_2', 'Freq2_harmonics_rel_phase_3',
    'Freq3_harmonics_rel_phase_0', 'Freq3_harmonics_rel_phase_1',
    'Freq3_harmonics_rel_phase_2', 'Freq3_harmonics_rel_phase_3',
    'CAR_sigma', 'CAR_tau', 'CAR_mean', 'Con', 'PairSlopeTrend',
]

LC_extractor = light_curve.Extractor(
    light_curve.Amplitude(),
    light_curve.AndersonDarlingNormal(),
    light_curve.BeyondNStd(nstd=1),
    light_curve.BeyondNStd(nstd=2),
    light_curve.BeyondNStd(nstd=3),
    light_curve.Cusum(),
    light_curve.Eta(),
    light_curve.EtaE(),
    light_curve.InterPercentileRange(0.25),
    light_curve.InterPercentileRange(0.1),
    light_curve.Kurtosis(),
    light_curve.LinearFit(),
    light_curve.LinearTrend(),
    light_curve.MagnitudePercentageRatio(),
    light_curve.MaximumSlope(),
    light_curve.Mean(),
    light_curve.Median(),
    light_curve.MedianAbsoluteDeviation(),
    light_curve.MedianBufferRangePercentage(),
    light_curve.OtsuSplit(),
    light_curve.PercentAmplitude(),
    light_curve.ReducedChi2(),
    light_curve.Skew(),
    light_curve.StandardDeviation(),
    light_curve.StetsonK(),
    light_curve.WeightedMean(),
)


def calc_FATS_features(lc):
    """Calculate features using the FATS library for a single light curve."""
    # Return array of -1 values if light curve data has no observations
    if lc is None:
        return np.full(len(FATS_feature_names), -1)

    # Prepare 2D array with required data for FATS
    # - magnitude (target)
    # - time (mjd)
    # - magnitude uncertainty (past_feat_dynamic_real)
    lc_2darr = np.array([
        lc['target'], lc['mjd'], lc['past_feat_dynamic_real']
    ])

    # Initialize FATS feature extractor with specified feature list
    FATS_feature_extractor = FATS.FeatureSpace(
        Data=['magnitude', 'time', 'error'],
        featureList=FATS_feature_names
    )
    # Calculate features and return the results
    FATS_feature_extractor = FATS_feature_extractor.calculateFeature(lc_2darr)
    return FATS_feature_extractor.result()


def calc_LC_features(lc):
    """Calculate features using the light_curve library for a single light curve."""
    # Return array of -1 values if no light curve data or insufficient points
    if lc is None or len(lc['target']) <= 4:
        return np.full(len(LC_extractor.names), -1)

    # Extract and convert light curve data to numpy arrays
    mag = np.array(lc['target'])
    magerr = np.array(lc['past_feat_dynamic_real'])
    mjd = np.array(lc['mjd'])

    # Calculate light curve features, assuming data is already sorted by time
    LC_features = LC_extractor(mjd, mag, magerr, sorted=True, check=False)
    return LC_features


def process_batch(batch_data, batch_idx, batch_size, bands_to_process, FATS_columns, LC_columns):
    """Process a batch of astronomical objects and return their features."""
    batch_FATS_features = pd.DataFrame(columns=FATS_columns)
    batch_LC_features = pd.DataFrame(columns=LC_columns)

    batch_fats_time = 0
    batch_lc_time = 0

    for i, star in enumerate(batch_data['bands_data']):
        # Calculate index in the original dataset
        star_idx = batch_idx * batch_size + i

        # Calculate FATS features for both bands
        start_time = time.time()
        # Generalize to any number of bands present
        band_FATS_feats = []
        for band in bands_to_process:
            try:
                feats = calc_FATS_features(star[band])
            except ValueError:
                feats = np.full(len(FATS_feature_names), -2)
            band_FATS_feats.append(feats)
        batch_fats_time += time.time() - start_time
        batch_FATS_features.loc[star_idx, :] = np.concatenate(band_FATS_feats)

        # Calculate light_curve features for both bands
        start_time = time.time()
        band_LC_feats = []
        for band in bands_to_process:
            try:
                feats = calc_LC_features(star[band])
            except ValueError:
                feats = np.full(len(LC_extractor.names), -2)
            band_LC_feats.append(feats)
        batch_lc_time += time.time() - start_time
        batch_LC_features.loc[star_idx, :] = np.concatenate(band_LC_feats)

    print("Finished", batch_idx)
    return batch_FATS_features, batch_LC_features, batch_fats_time, batch_lc_time


def compile_handcrafted_features(data, bands_to_process, FATS_columns, LC_columns, num_workers):
    """Compile all handcrafted features for a dataset of astronomical objects."""
    FATS_features = pd.DataFrame(columns=FATS_columns)
    LC_features = pd.DataFrame(columns=LC_columns)

    # Track time spent on feature calculation for performance monitoring
    total_fats_time = 0
    total_lc_time = 0

    # Set up multiprocessing parameters
    num_workers = num_workers - 1  # Leave one CPU free
    batch_size = max(1, len(data) // (num_workers * 2))  # make 2*num_workers batches

    # Create batches
    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batches.append((
            batch, i // batch_size, batch_size,
            bands_to_process, FATS_columns, LC_columns
        ))

    # Set up progress reporting
    print(f"Processing {len(data)} objects with {num_workers} workers in {len(batches)} batches")

    # Process batches in parallel
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.starmap(process_batch, batches),
            total=len(batches),
            desc="Computing handcrafted features"
        ))

    # Combine results
    total_fats_time = 0
    total_lc_time = 0

    for batch_FATS_features, batch_LC_features, batch_fats_time, batch_lc_time in results:
        FATS_features = pd.concat([FATS_features, batch_FATS_features])
        LC_features = pd.concat([LC_features, batch_LC_features])
        total_fats_time += batch_fats_time
        total_lc_time += batch_lc_time

    # Sort dataframes by index to maintain original order
    FATS_features = FATS_features.sort_index()
    LC_features = LC_features.sort_index()

    # Report timing information for benchmarking
    print("\n-- Time spent calculating features --")
    print(f"FATS features: {total_fats_time/60:.2f} minutes")
    print(f"LC features:   {total_lc_time/60:.2f} minutes")

    # Combine all features into a single dataframe
    handcrafted_features = pd.concat((FATS_features, LC_features), axis=1)
    return handcrafted_features


if __name__ == "__main__":
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description="Extract handcrafted features from dataset")
    parser.add_argument(
        '--split', type=str, choices=['train', 'validation', 'test', 'anom'],
        required=True, help='Dataset split to process (train, validation, test, or anom)'
    )
    parser.add_argument(
        "--dataset_path", type=str,
        required=True, help='Path to dataset on disk'
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to save the extracted features as a HF dataset"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of worker processes to use for parallel feature extraction"
    )
    parser.add_argument(
        "--bands_to_process", type=str,
        required=False, help='Bands to process (comma-separated list, e.g. "I,V")'
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path
    split = args.split
    num_workers = args.num_workers
    bands_to_process = args.bands_to_process
    output_path = args.output_path

    # Load dataset
    dataset = load_from_disk(dataset_path)

    # Determine which bands are present in the dataset's light curves
    if bands_to_process is not None:
        bands_to_process = bands_to_process.split(',')
    else:
        first_example = dataset[split][0]
        if "bands_data" not in first_example:
            raise ValueError("Expected 'bands_data' key in dataset example, but not found.")
        bands_to_process = list(first_example["bands_data"].keys())
        print(f"Bands present in the dataset: {bands_to_process}")

    # Prepare column names for FATS and LC features
    FATS_columns = [
        band + '_' + feat_name for band in bands_to_process for feat_name in FATS_feature_names
    ]
    LC_columns = [
        band + '_' + feat_name for band in bands_to_process for feat_name in LC_extractor.names
    ]

    # Process the specified split
    print(f"Processing {split} split...")
    hc_feats = compile_handcrafted_features(
        dataset[split], bands_to_process, FATS_columns, LC_columns, num_workers
    )

    # Save results
    output_file = f"{output_path}/{split}_{os.path.basename(dataset_path)}.csv"
    hc_feats.to_csv(output_file, index=None)
    print(f"Features saved to {output_file}")
