# StarEmbed Benchmark Pipeline

This directory contains the optimized benchmark pipeline for evaluating stellar time series embeddings through pre-computed combined embeddings.

## **Pipeline Overview**

The pipeline consists of two main stages:

1. **Preprocessing**: Generate combined embeddings from multi-band data
2. **Evaluation**: Run benchmark scripts using pre-computed embeddings

```
Raw Dataset → compute_avg_embeddings.py → Combined Embeddings → Benchmark Scripts
```

### **Performance Benefits**
- **10x+ speedup** over on-the-fly embedding computation  
- **Memory efficient** with direct numpy array access
- **Consistent results** across all benchmark methods

---

## **Directory Structure**

```
src_clean/
├── model/
│   ├── astromer_1/                  # Astromer model embeddings
│   ├── astromer_2/                  # Astromer model embeddings  
│   ├── chronos/                     # Chronos model embeddings
│   ├── moirai/                      # Moirai model embeddings
│   └── compute_avg_embeddings.py    # Step 1: Preprocessing (works with all models)
├── benchmark/
│   ├── classification/
│   │   ├── linear_knn.py            # Step 2a: Logistic Regression + kNN
│   │   ├── mlp_pl2_wloss_standardization.py  # Step 2b: MLP with PyTorch Lightning
│   │   └── rf_hpo.py               # Step 2c: Random Forest with HPO
│   └── clustering/
│       └── clustering.py            # Step 2d: K-Means, Hierarchical + t-SNE
└── README.md                       # This file
```

---

## **Step 1: Preprocessing - Generate Combined Embeddings**

**Purpose**: Convert multi-band embeddings (column name like `embeddings_g`, `embeddings_r`) into `combined_embedding` for fast benchmark evaluation.

### **Usage**
```bash
cd /projects/b1094/StarEmbed/skai_universal_forecaster/src_clean/model

# Basic usage (automatically detects available bands)
python compute_avg_embeddings.py \
    --dataset DATASET_NAME \
    --band_combination concat

# With explicit base path
python compute_avg_embeddings.py \
    --dataset DATASET_NAME \
    --band_combination concat \
    --base_path /projects/b1094/StarEmbed/embeddings/descriptive_name_embeddings
```

### **Key Parameters**
- `--dataset`: embedding directory name
- `--band_combination`: How to combine bands
  - `concat`: Concatenate bands → shape (B, D_g + D_r) 
  - `avg`: Average bands → shape (B, D)
  - `g`/`r`: Use single band → shape (B, D_g) or (B, D_r)
- `--bands`: Which bands to look for (default: `g r` - **automatically detects which ones exist in the embedding**)

### **Output**
- `avg_embedding_g`, `avg_embedding_r`: Individual band embeddings
- `combined_embedding`: Ready-to-use combined embedding for benchmarks
- Backup copies of original data

### **Examples**
```bash
# For hand-crafted features (auto-detects g,r bands → 69-dim each)
python compute_avg_embeddings.py \
    --dataset csdr1_raw4_catflags_filtered_embs_hand_crafted_trn_val_tst_bandgr \
    --band_combination concat
# Output: "Found embedding columns for bands: ['g', 'r']"
# Result: combined_embedding shape (N, 138)

# For datasets with different bands (auto-detects i,z bands)
python compute_avg_embeddings.py \
    --dataset some_dataset_with_iz_bands \
    --bands g r i z \
    --band_combination concat
# Output: "Found embedding columns for bands: ['i', 'z']"
# Result: Uses only the bands that actually exist
```

---

## **Step 2: Evaluation - Run Benchmark Scripts**

All benchmark scripts automatically detect and use pre-computed `combined_embedding` for maximum speed.

### **Unified Embedding Extraction Logic**
All four evaluation scripts follow the same simple pattern:

1. **Check for `combined_embedding`** column in dataset
2. **Extract directly**: `np.array(dataset["combined_embedding"])`

**Requirement**: All scripts expect pre-computed `combined_embedding` from Step 1 preprocessing.


### **2a. Linear Classifier (Logistic Regression + kNN)**

**Purpose**: Fast baseline classification with sklearn models.

```bash
cd benchmark/classification

python linear_knn.py \
    --input_embs /path/to/dataset \
    --scenario concat \
    --seed 42
```

**Key Parameters**:
- `--input_embs`: Path to dataset with `combined_embedding`
- `--scenario`: Ignored if `combined_embedding` exists (legacy compatibility)
- `--seed`: Random seed for reproducibility

**Outputs**: Confusion matrices, classification reports, PDF plots

---

### **2b. MLP with PyTorch Lightning**

**Purpose**: Deep learning classification with MLPs, advanced training features.

```bash
python mlp_pl2_wloss_standardization.py \
    --input_embs /path/to/dataset \
    --scenario concat \
    --epochs 50 \
    --batch_size 1024 \
    --lr 1e-3 \
    --standardize 1 \
    --seed 42
```

**Key Parameters**:
- `--epochs`: Training epochs (default: 50)
- `--batch_size`: Batch size (default: 1024)  
- `--lr`: Learning rate (default: 1e-3)
- `--standardize`: Use StandardScaler (1=True, 0=False)
- `--hidden_layers`: Network depth (2 or 3)

**Outputs**: Training curves, validation metrics, best model checkpoints

---

### **2c. Random Forest with Hyperparameter Optimization**

**Purpose**: Tree-based classification with automated hyperparameter search.

```bash
python rf_hpo.py \
    --input-embs /path/to/dataset \
    --standardize 1 \
    --seed 42 \
    --skip-hpo \
    --best-params '{"n_estimators": 200, "max_depth": null, "min_samples_split": 10}' \
    --output-dir /path/to/output
```

**Key Parameters**:
- `--skip-hpo`: Skip hyperparameter optimization (faster)
- `--best-params`: JSON string with RF parameters (use `null` for Python `None`)
- `--standardize`: Apply feature standardization
- `--output-dir`: Results directory

**Outputs**: Cross-validation results,  performance metrics across multiple seeds

---

### **2d. Clustering Analysis (K-Means + Hierarchical + t-SNE)**

**Purpose**: Unsupervised analysis and visualization of embedding quality.

```bash
cd ../clustering

python clustering.py \
    --dataset-dir /path/to/dataset \
    --mode test \
    --scenario concat \
    --clustering-method both \
    --standardize 1 \
    --random-state 42
```

**Key Parameters**:
- `--mode`: Which split to analyze (`train`, `test`, `validation`, `both`, `all`)
- `--clustering-method`: Algorithms to run (`kmeans`, `hierarchical`, `both`)
- `--standardize`: Standardize features before clustering
- `--save-dendrogram`: Save hierarchical clustering dendrograms

**Outputs**: Clustering metrics (ARI, NMI, F1), t-SNE plots, dendrograms



---

## **Important Notes**

### **Data Requirements**
- Dataset must have standard column names: `embeddings_g`, `embeddings_r`, `class_str`
- Run `compute_avg_embeddings.py` **first** to create `combined_embedding`
- Use `rename_embedding_columns.py` if columns are named differently (e.g., `g_embedding` → `embeddings_g`)


### **Output Locations**
- Linear classifier: `/projects/b1094/StarEmbed/src/output/linear_classification/`
- MLP: Working directory + experiment name
- Random Forest: `--output-dir` parameter  
- Clustering: `/projects/b1094/StarEmbed/src/output/clustering/`

### **GPU Usage**
- Linear Classifier: CPU only
- **MLP**: Automatically detects and uses GPU if available
- Random Forest: CPU only (uses `n_jobs=-1` for parallelization)
- Clustering: CPU only

---
