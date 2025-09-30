# ASTROMER-1 Setup Guide

## Environment Setup

### 1. Create conda environment
Use conda to create a virtual environment with Python 3.10:
```bash
conda create -n astromer_py310 python==3.10
conda activate astromer_py310 
```

### 2. Install ASTROMER
```bash
pip install ASTROMER datasets
```

### 3. Fix Version Compatibility Issues

**IMPORTANT**: ASTROMER was built for older TensorFlow/Keras versions. The default `pip install ASTROMER` installs the newest, incompatible versions of TensorFlow and Keras that cause import errors. You need to manually downgrade:

```bash
pip install "tensorflow<2.16" "keras<3"
```
Using Python 3.10 + TensorFlow 2.15.1 + Keras 2.15.0 resolves all issues

### 4. Verify Installation
Test that ASTROMER can be imported:
```bash
python -c "from ASTROMER.models import SingleBandEncoder; print('ASTROMER import successful')"
```

## Downloading Model Weights

The original model loading method on [astromer-1 official github](https://github.com/astromer-science/python-library/tree/main?tab=readme-ov-file#how-to-use-it) have some bugs.

**PROBLEM**: ASTROMER's `from_pretraining('macho')` has a broken download URL:
- Looks for: `macho.zip` (doesn't exist) 
- Actual file: `macho_a0.zip`
- Downloads HTML 404 page → `zipfile.BadZipFile` error

**SOLUTION**: Manually download the correct weights, then use original `from_pretraining('macho')`:

### Manual Weight Download
```bash
cd src/model/astromer_1
mkdir -p weights
cd weights

# Download to weights/macho/ directory (what ASTROMER expects)
wget https://github.com/astromer-science/weights/raw/main/macho_a0.zip
unzip macho_a0.zip  # This creates macho/ directory with actual weights
```

### Test
```python
from ASTROMER.models import SingleBandEncoder

model = SingleBandEncoder()
model = model.from_pretraining('macho')
```
Should output: `[INFO] Weights already downloaded` without errors.

## Details of Embedding Generation Pipeline follow from Astromer-1 paper

Install the hugginface dataset since all our raw data is saved in this format.
```
pip install datasets
```


### Data Flow Overview

The ASTROMER embedding pipeline transforms astronomical light curves through two main stages:

1. **`make_windows()`**: Raw time series → Fixed-size (choose the last 200 time step) neural network input windows
2. **`encode_batch()`**: Time series windows → High-dimensional embeddings via pre-trained transformer

### Input Data Format

The pipeline expects HuggingFace datasets with this structure:
```python
{
  "sourceid": 12345,                    # Unique object identifier
  "bands_data": {                       # Multi-band photometric data
    "g": {                              # g-band observations
      "mjd": [59001.5, 59002.3, ...],          # Modified Julian Dates
      "target": [18.2, 18.4, ...],             # Magnitude measurements  
      "past_feat_dynamic_real": [0.05, 0.07, ...], # Measurement errors
      "class": "RRLyrae",                       # Object classification
      # ... additional metadata
    },
    "r": { /* same structure for r-band */ },
    "i": { /* same structure for i-band */ }
  }
}
```

### Stage 1: `make_windows()` - Time Series Preprocessing

**Purpose**: Convert variable-length astronomical time series into fixed-size windows suitable for transformer input.

**Input**: 
- Raw time series: `(mjd, magnitude, error)` of variable length `L`
- Target window size: `duration` (default: 200)

**Processing**:
```python
# 1. Stack time series components
arr = np.stack([mjd, mag, err], axis=1)      # Shape: (L, 3)

# 2. Handle variable lengths
if L >= duration:
    win = arr[-duration:]                     # Truncate: take recent observations
else:
    pad = np.zeros((duration - L, 3))        # Zero-pad: extend short sequences
    win = np.vstack([arr, pad])

# 3. Normalize by removing temporal mean
win = win - win.mean(axis=0, keepdims=True)
```

**Output**: 
- Valid bands: `windows_{band}` with shape `(duration, 3)`
- Missing bands: `windows_{band} = None`

**Edge Cases**:
- **Missing band data**: When `bands_data` doesn't contain the requested band → `None`
- **Short sequences**: Sequences shorter than `duration` are zero-padded
- **Long sequences**: Only the most recent `duration` observations are kept

### Stage 2: `encode_batch()` - Embedding Generation

**Purpose**: Transform time series windows into semantic embeddings using pre-trained ASTROMER transformer.

**Input**: 
- Batch of time series windows: `[(duration, 3), (duration, 3), None, ...]`
- Pre-trained SingleBandEncoder model

**Processing**:
```python
# 1. Filter valid windows (skip None values)
valid_ids, wins_valid = [], []
for i, w in enumerate(windows):
    if w is not None:
        wins_valid.append(w)
        valid_ids.append(i)

# 2. Handle empty batches
if not valid_ids:
    return {"embeddings_{band}": None}

# 3. Neural encoding via ASTROMER transformer
out = model.encode(wins_valid, ...)          # Shape: (V, duration, D)

# 4. Reconstruct full batch (map back to original positions)
emb = np.zeros((B, duration, D))            # Initialize with zeros
for loc, idx in enumerate(valid_ids):
    emb[idx] = out[loc]                     # Fill valid positions
```

**Output**:
- **Full batch invalid**: `embeddings_{band} = None`
- **Mixed batch**: `embeddings_{band}` with shape `(B, duration, D)` where:
  - `B` = batch size
  - `duration` = temporal dimension (200)
  - `D` = embedding dimension (model.hidden_size)
  - Invalid objects → zero vectors `[0, 0, ..., 0]`

### Final Dataset Structure

After processing, each example gains embedding fields:
```python
{
  "sourceid": 12345,
  "bands_data": { ... },                # Original data preserved
  "embeddings_g": [[...], [...], ...], # Shape: (duration, D) or None
  "embeddings_r": [[...], [...], ...], # Shape: (duration, D) or None  
  "embeddings_i": [[...], [...], ...], # Shape: (duration, D) or None
}
```

