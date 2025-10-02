## OGLE-IV Collection of Variable Stars (OCVS)

Optical Gravitational Lensing Experiment (OGLE-IV) Collection of Variable Stars. Primary reference: Udalski et al., 2015.

- **Homepage**: `https://www.astrouw.edu.pl/ogle/ogle4/OCVS/`
- **FTP root**: `ftp://ftp.astrouw.edu.pl/ogle/ogle4/OCVS/`

See the standardized schema, common preprocessing, and evaluation guidance in `src_clean/datasets/README.md`.


### Download raw data

Explore and download by region/type or mirror the full OCVS directory:

```bash
# Mirror OCVS (place into repo data dir)
mkdir -p ../../../data/ogle4_raw
wget -r -nH -P ../../../data/ogle4_raw ftp://ftp.astrouw.edu.pl/ogle/ogle4/OCVS/

# Resulting structure (abbrev.):
# ../../../data/ogle4_raw/OCVS/
#   blg/  gd/  gal/  lmc/  smc/
#     CEP/ RRLYR/ DSCT/ T2CEP/ ACEP/ LPV/ ECL/ HB/ ROT/ TRANSITS/
#       [catalog files, ident.dat, remarks.txt, phot*/I|V/*.dat]
```


## Convert to Hugging Face Dataset

```bash
python OGLE_to_hf.py \
  --num_workers 4
```

- **Output**: a `datasets` on-disk directory at `../../../data/ogle4` with Arrow shards and `dataset_info.json`.
- The converter iterates over specified catalogs, merges catalog, remarks, and ident metadata, and aggregates I/V band light curves per source.


## OGLE-specific schema notes

Beyond the shared StarEmbed schema, OGLE includes multi-band photometry and rich catalog metadata. Bands present: `I` and `V` under `bands_data`.

- `bands_data.I` and `bands_data.V` contain:
  - `mjd`: observation times converted from HJD/shifted HJD to MJD
  - `target`: magnitudes
  - `past_feat_dynamic_real`: magnitude uncertainties
  - `feat_dynamic_real`: Δmjd
  - `length`: observation count

Catalog/metadata fields added by the converter (availability may vary by subtype):
- `avg_mag_I`, `avg_mag_V`: average magnitudes
- `parent_type`: high-level OGLE class family (e.g., `cep`, `rrlyr`, `dsct`, `lpv`, `ecl`, `hb`, `t2cep`, `acep`, `rot`, `transits`)
- `sub_type`: subtype (e.g., `RRab`, `RRc`, `RRd`, `cepF`, `cep1O`, etc.)
- `class_str`: canonical class label; for some families this is populated from `ident.dat`
- `region`: OGLE sky region (`blg`, `lmc`, `smc`, `gd`, `gal`)
- `remarks`: free-text notes aggregated from `remarks.txt` and nonstandard features
- `OGLE_IV_id`, `OGLE_III_id`, `OGLE_II_id`, `other_id`: cross-identifications
- `ra`, `dec`: sexagesimal strings from `ident.dat` (some families use space-separated strings; see preprocessing)

Period and Fourier feature columns (repeated up to 3 periods depending on subtype):
- For period k in {1,2,3}: `period{k}`, `period{k}_unc`, `time_of_peak{k}[HJD]`, `amp{k}_I`, `fourier{k}_R21`, `fourier{k}_phi21`, `fourier{k}_R31`, `fourier{k}_phi31`
- For k=1, columns appear without the `{k}` suffix (e.g., `period`, `period_unc`, `time_of_peak[HJD]`, `amp_I`, `fourier_R21`, ...)

Note: Some families add nonstandard features (e.g., eclipsing depths, orbital params, transit metrics) which the converter compacts into `remarks` while keeping the unified schema consistent.


## OGLE class taxonomy (examples)

Families and example subtypes represented in OCVS:
- `rrlyr`: `RRab`, `RRc`, `RRd`, `aRRd`
- `cep` and `t2cep`/`acep`: `cepF`, `cep1O`, `cepF1O`, `cep1O2O`, `t2cep`, `acepF`, `acep1O`
- `dsct`: `dsct`
- `lpv`: `Miras`
- `ecl`: `ecl`, `ell`
- `hb`: `hb`
- `rot`: `rot`
- `transits`: `transits`

The converter sets `class_str` to subtype or a derived label via `ident.dat` where required.


## OGLE-specific preprocessing

- Catalog parsing per family using fixed-width or whitespace-delimited readers.
- Replacement of missing tokens (e.g., `-`, empty) with `NaN`.
- Region/type-specific `ident.dat` parsing to extract `ra`, `dec`, OGLE cross-IDs; whitespace trimmed.
- For `hb`, RA/Dec strings reformatted from space-separated to colon-separated sexagesimal.
- Nonstandard features for `hb`, `ecl`, `rot`, `transits` injected into `remarks` as `key=value` pairs.
- Light curves read across varying `phot*` directory names and both `I` and `V` bands.
- Time conversion: HJD or shifted HJD converted to MJD using detected format; Δmjd channel computed per band.
- Entries without any light curve in either band are skipped; missing single-band data represented as absent band entries.


## Limitations and notes

- Heterogeneous catalogs across regions/types require family-specific parsing; some features may be partially populated depending on subtype.
- RA/Dec are stored as sexagesimal strings from OGLE identification files; convert to decimal degrees if needed downstream.
- Sampling is irregular; band coverage varies by source and region.
- Remarks are free text and may include multiple annotations concatenated with `|`.


## License and terms of use

Respect OGLE data use policies and cite the original works. See `https://www.astrouw.edu.pl/ogle/` for terms and acknowledgements.


## Citation

If you use this dataset, please cite OGLE and this repository:

- Udalski, A., et al. 2015. “OGLE-IV: OGLE Collection of Variable Stars.” Acta Astronomica, 65, 1.
- This repository: please cite per the project’s main `README.md`.