# Variable Star Light Curve Preprocessing

This tool processes light curve data from variable stars, applying phase folding, resampling, and optionally generating visualizations. It takes raw light curve data (typically from the variable star downloader) and prepares it for analysis or machine learning applications.

## Features

- Phase folding of light curves using known periods
- Resampling to regular phase intervals
- Handling of missing data points (with NaN or LastValue interpolation)
- Filtering by star class and photometric band
- Visualization of processed light curves

## Requirements

- Python 3.6+
- Required packages:
  - numpy
  - pandas
  - matplotlib
  - astropy
  - torch
  - scikit-learn

## Usage

```bash
python -m crds-pipeline.src.tasks.preproc.__main__ [options]
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input-file` | Path to input CSV file with light curve data | ./data/download/lsdb/lightcurves.csv |
| `--output-dir` | Directory to save processed data | ./data/processed |
| `--output-filename` | Name of output CSV file | processed_lightcurves.csv |
| `--band` | Photometric band to process (g, r, or i) | g |
| `--class-ids` | List of class IDs to filter by | None (all classes) |
| `--stars-per-class` | Number of stars to process per class | 100 |
| `--plot` | Generate plots of processed light curves | False |

## Examples

### Basic Processing

Process all stars from the default input file, using g-band data:

```bash
python -m crds-pipeline.src.tasks.preproc.__main__
```

### Processing Specific Classes

Process 10 stars each from classes 1 and 5:

```bash
python -m crds-pipeline.src.tasks.preproc.__main__ --class-ids 1 5 --stars-per-class 10
```

### Processing and Plotting

Process stars from class 1 and generate light curve plots:

```bash
python -m crds-pipeline.src.tasks.preproc.__main__ --class-ids 1 --stars-per-class 20 --plot
```

### Changing Bands

Process stars using r-band data instead of default g-band:

```bash
python -m crds-pipeline.src.tasks.preproc.__main__ --class-ids 1 5 --band r
```

### Custom Input/Output Paths

Specify custom input and output locations:

```bash
python -m crds-pipeline.src.tasks.preproc.__main__ --input-file ./my_data/lightcurves.csv --output-dir ./results/processed
```

## Output

The tool produces:

1. **Processed Data CSV**: Contains phase-folded, resampled light curve data with a multi-level index (star_id, phase).

2. **Light Curve Plots** (if `--plot` is specified): Visualizations saved in the output directory under `plots/class_{class_id}/star_{star_id}.png`.


## Notes

- Phase folding uses the period from the input data to convert time to phase.
- The default phase interval of 0.0025 produces 400 points per period.
- The "LastValue" interpolation fills missing phase bins with the last valid measurement.