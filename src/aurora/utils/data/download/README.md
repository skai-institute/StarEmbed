# Variable Star Data Downloader

This tool downloads light curve data for variable stars from the ZTF (Zwicky Transient Facility) catalog. It allows you to select random stars from specific variable star classes, retrieve their photometric data, generate light curve plots, and save the data to CSV files.

## Requirements

- Python 3.6+
- Required packages:
  - dask
  - matplotlib
  - lsdb
  - pandas
  - numpy
  - astropy

You can install the required packages using:

```bash
pip install dask matplotlib lsdb pandas numpy astropy
```

## Usage

```bash
python -m src.tasks.download [options]
```

### Basic Examples

1. Download 10 stars from classes 1, 2, and 5:

```bash
python -m src.tasks.download --n 10 --class-ids 1 2 5
```

2. Download 20 stars from class 1 and save to a custom directory:

```bash
python -m src.tasks.download --n 20 --class-ids 1 --output-dir ./my_variable_stars
```

3. Download 15 stars randomly from all classes:

```bash
python -m src.tasks.download --n 15
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--varstars-file` | Path to the variable stars catalog file | ./src/tasks/download/catalog/CSDR1_varstars.txt |
| `--output-dir` | Directory to save output files | ./data/download/lsdb |
| `--output-file-name` | Name of the output CSV file | lightcurves.csv |
| `--ztf-source` | URL for the ZTF data source | https://data.lsdb.io/hats/ztf_dr14/ztf_source |
| `--n` | Number of stars to select per class | 20 |
| `--class-ids` | List of class IDs to filter by (e.g., 1 2 5) | None (all classes) |
| `--seed` | Random seed for reproducible selection | 42 |

## Output

The tool produces the following outputs:

1. **Light curve plots**: Saved in the `{output_dir}/plots/{class_id}/` directories, with separate plots for each photometric band (g, r, i).

2. **CSV data**: All light curve data is saved to `{output_dir}/{output_file_name}`. If the file already exists, new data will be merged with existing data, avoiding duplicates.

## Variable Star Classes

Common variable star classes in the catalog include:

- Class 1: RR Lyrae, type ab (RRab)
- Class 2: RR Lyrae, type c (RRc) 
- Class 3: Cepheid variables
- Class 4: Eclipsing binaries, type EA
- Class 5: Eclipsing binaries, type EB
- Class 6: Eclipsing binaries, type EW

## Notes

- The tool uses a cone search to find ZTF observations near the coordinates of each variable star.
- Light curves are phased using the period information from the variable star catalog.
- If the number of stars requested exceeds the available stars in a class, all available stars will be selected.