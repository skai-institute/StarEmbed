# StarEmbed datasets

Here, you can find descriptions of all StarEmbed datasets and Python code for converting them to the standardized StarEmbed schema in HuggingFace datasets format. Each dataset has its own directory beneath `datasets/` with download instructions and a converter script.


## Standardized schema (applies to all datasets)

All datasets are converted into a unified per-star record with one or more passbands under `bands_data`. Types follow `datasets.Features` from Hugging Face Datasets.

Top-level fields:
- `sourceid`: string — Native survey/source identifier
- `numerical_id`: string — Numeric ID when provided by the survey
- `bands_data`: object keyed by band name (e.g., `C`, `V`, `I`, `g`, `r`)
  - `{band}.mjd`: sequence<float64> — Modified Julian Dates per observation
  - `{band}.target`: sequence<float64> — Photometric measurements (magnitudes unless stated otherwise)
  - `{band}.past_feat_dynamic_real`: sequence<float64> — Per-observation uncertainties or auxiliary past-known features
  - `{band}.feat_dynamic_real`: sequence<float64> — Per-observation known features aligned with observations (e.g., Δtime)
  - `{band}.length`: int64 — Number of observations for the band
- `avg_mag_V`: float64 — Catalog mean magnitude (V or survey-provided proxy)
- `period`: float64 — Period in days when available
- `class_str`: string — Variable star class label (survey taxonomy)
- `class_int`: int64 — Integer class ID (mapped consistently within each dataset)
- `ra`: float64 — Right ascension (deg)
- `dec`: float64 — Declination (deg)

Notes:
- Bands present depend on the survey. Single-band surveys use a single key (e.g., `C` for Catalina).
- `target` is a magnitude by default; if a dataset uses flux, the dataset README will state it explicitly.
- Optional fields may be missing for some surveys (e.g., no period).


## Common preprocessing

- Drop invalid timestamps (e.g., negative `mjd`).
- Sort light curves by time ascending.
- Harmonize coordinates to decimal degrees.
- Map survey-specific integer classes to human-readable strings.
- Build time-delta channel in `{band}.feat_dynamic_real` when appropriate.


## Splits and evaluation guidance

- Classification: use stratified train/val/test splits by `class_str`. Report accuracy, macro F1, and per-class metrics.
- Forecasting: avoid leakage by splitting within-series by time (e.g., last 20% as horizon). Report MAE/MSE; consider uncertainty-weighted variants when available.
- Period regression: evaluate with MAE/RMSE (optionally log-scale) and within-x% accuracy.


## Usage with Hugging Face Datasets

```python
from datasets import load_from_disk

# Load any converted dataset directory (e.g., ../../../data/catalina)
ds = load_from_disk(DATASET_DIR)
example = ds[0]

# Access a band
band = next(iter(example["bands_data"]))
mjd = example["bands_data"][band]["mjd"]
values = example["bands_data"][band]["target"]
unc = example["bands_data"][band]["past_feat_dynamic_real"]
```


## Validation and statistics

Each dataset folder may include utilities to validate schema conformance and compute basic statistics (class counts, length distributions). A unified validation script is planned.


## License and attribution

Respect each survey's data release policies and cite the original works. See individual dataset READMEs for dataset-specific terms and bibliographic references.
