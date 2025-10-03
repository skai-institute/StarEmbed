# Hand-Crafted Features Module

## Overview

The Hand-Crafted Features module represents a foundational approach to time-series analysis in astronomy, serving as both the long-standing baseline and state-of-the-art technique for characterizing light curves. This module extracts comprehensive statistical and temporal features from astronomical time-series data, providing interpretable and robust representations that have proven essential for variable star classification, anomaly detection, and astronomical discovery.

Hand-crafted features have been the cornerstone of time-domain astronomy for decades, offering transparent, physics-informed representations that capture the essential characteristics of stellar variability. Despite the emergence of deep learning approaches, these traditional features remain critical due to their interpretability, computational efficiency, and the deep domain knowledge they encapsulate.

## Features Computed in `extract_feats.py`

The `extract_feats.py` script computes a comprehensive suite of features using two primary libraries: **FATS (Feature Analysis for Time Series)** and **light_curve**. These features capture various aspects of light curve behavior including periodicity, variability, statistical properties, and temporal structure.

### FATS Features

The FATS library provides 32 carefully selected features that capture fundamental properties of astronomical time series:

#### Periodicity and Frequency Analysis
- **PeriodLS**: Period derived from Lomb-Scargle periodogram
- **Period_fit**: Best-fit period from model fitting
- **Freq1_harmonics_amplitude_0-3**: Amplitudes of the first 4 harmonics of the primary frequency
- **Freq2_harmonics_amplitude_0-3**: Amplitudes of the first 4 harmonics of the secondary frequency  
- **Freq3_harmonics_amplitude_0-3**: Amplitudes of the first 4 harmonics of the tertiary frequency
- **Freq1_harmonics_rel_phase_0-3**: Relative phases of the first 4 harmonics of the primary frequency
- **Freq2_harmonics_rel_phase_0-3**: Relative phases of the first 4 harmonics of the secondary frequency
- **Freq3_harmonics_rel_phase_0-3**: Relative phases of the first 4 harmonics of the tertiary frequency

#### Variability Indices
- **Psi_CS**: Stetson's CS variability index for consecutive observations
- **Psi_eta**: Stetson's eta variability index
- **Con**: Number of three consecutive data points brighter/dimmer than 2σ from mean
- **PairSlopeTrend**: Trend of slopes between consecutive data points

#### Statistical Properties
- **CAR_sigma**: Standard deviation parameter of Continuous Auto-Regressive model
- **CAR_tau**: Timescale parameter of Continuous Auto-Regressive model
- **CAR_mean**: Mean parameter of Continuous Auto-Regressive model
- **Autocor_length**: Length of autocorrelation function

*Reference*: [FATS: Feature Analysis for Time Series](https://github.com/isadoranun/FATS) - A comprehensive library for extracting features from astronomical time series data.

### Light Curve Features

The `light_curve` library provides 27 additional features optimized for astronomical time-series analysis:

#### Statistical Moments and Distribution
- **Amplitude**: Half the difference between maximum and minimum magnitudes
- **Mean**: Mean magnitude value
- **Median**: Median magnitude value
- **StandardDeviation**: Standard deviation of magnitudes
- **Skew**: Skewness of the magnitude distribution
- **Kurtosis**: Kurtosis of the magnitude distribution
- **MedianAbsoluteDeviation**: Median absolute deviation from median

#### Variability and Outlier Detection
- **BeyondNStd**: Percentage of points beyond N standard deviations (N=1,2,3)
- **PercentAmplitude**: Percentage amplitude of variability
- **MagnitudePercentageRatio**: Ratio of magnitude percentiles
- **MedianBufferRangePercentage**: Percentage of points within amplitude/10 of median

#### Temporal Structure and Trends
- **LinearFit**: Parameters from linear fitting (slope, intercept, chi-squared)
- **LinearTrend**: Slope of linear trend
- **MaximumSlope**: Maximum slope between consecutive observations
- **Cusum**: Cumulative sum statistic
- **Eta**: Eta variability index
- **EtaE**: Eta_e variability index

#### Distribution Analysis
- **InterPercentileRange**: Inter-percentile ranges (25th, 10th percentiles)
- **AndersonDarlingNormal**: Anderson-Darling test statistic for normality
- **ReducedChi2**: Reduced chi-squared statistic
- **StetsonK**: Stetson's K variability index
- **WeightedMean**: Weighted mean magnitude
- **OtsuSplit**: Otsu threshold-based feature

*Reference*: [light_curve Python Package](https://github.com/light-curve/light-curve-python) - A high-performance library for astronomical light curve feature extraction.

## Key Papers and References

### FATS Library
- **Nun, I. et al.** (2015): "FATS: Feature Analysis for Time Series" - The original paper introducing the FATS library and its comprehensive feature set for astronomical time series analysis.
- **Repository**: [FATS on GitHub](https://github.com/isadoranun/FATS)

### Light Curve Library  
- **Mowlavi, N. et al.** (2018): "Gaia Data Release 2: Variable stars in the colour-absolute magnitude diagram" - Demonstrates the application of light curve features in large-scale astronomical surveys.
- **Repository**: [light_curve on GitHub](https://github.com/light-curve/light-curve-python)
- **Rust Implementation**: [light-curve-feature](https://docs.rs/light-curve-feature) - High-performance Rust implementation

### Foundational Work on Hand-Crafted Features
- **Richards, J.W. et al.** (2011): "Machine Learning for Variable Star Classification" - Comprehensive study establishing hand-crafted features as the standard approach.
- **Debosscher, J. et al.** (2007): "Automated supervised classification of variable stars" - Early work demonstrating the effectiveness of statistical features.
- **Stetson, P.B.** (1996): "On the Automatic Determination of Light-Curve Parameters for Cepheid Variables" - Foundational work on variability indices.

## Usage

The module processes multi-band astronomical datasets and extracts features for each band separately, creating comprehensive feature vectors that capture both individual band characteristics and cross-band relationships. Features are computed in parallel across multiple workers for efficient processing of large datasets.

The `extract_feats.py` script can be run from the command line with the following basic syntax:

```bash
python extract_feats.py --split <split_name> --dataset_path <path_to_dataset> [--num_workers <n>] [--bands_to_process <band_list>]
```

#### Example Commands

**Process training data with default settings:**
```bash
python extract_feats.py --split train --dataset_path ../../data/catalina
```

**Process test data with custom worker count:**
```bash
python extract_feats.py --split test --dataset_path ../../data/ogle4 --num_workers 8
```

**Process specific bands only:**
```bash
python extract_feats.py --split validation --dataset_path ../../data/catalina --bands_to_process "I,V"
```

#### Required Arguments
- `--split`: Dataset split to process (`train`, `validation`, `test`, or `anom`)
- `--dataset_path`: Path to the HuggingFace dataset directory on disk

#### Optional Arguments
- `--num_workers`: Number of worker processes for parallel processing (default: 4)
- `--bands_to_process`: Comma-separated list of bands to process (e.g., "I,V,R"). If not specified, all bands present in the dataset will be processed.

