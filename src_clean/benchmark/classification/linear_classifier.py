# linear_models.py
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from functools import partial
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import random
import time

# Add src_clean directory to path to find benchmark.utils
script_dir = os.path.dirname(os.path.abspath(__file__))  # /path/to/benchmark/classification/
benchmark_dir = os.path.dirname(script_dir)              # /path/to/benchmark/
src_clean_dir = os.path.dirname(benchmark_dir)           # /path/to/src_clean/
sys.path.append(src_clean_dir)

# Import unified utilities
from benchmark.utils import remove_outliers, add_label_indices, compute_embedding

def check_nan_stats(X, name="dataset"):
    total_points = X.shape[0]
    nan_mask = ~np.isfinite(X)
    rows_with_nan = np.any(nan_mask, axis=1)
    n_bad = np.sum(rows_with_nan)
    print(f"[NaN Check] {name}: {n_bad}/{total_points} samples contain NaN/Inf ({(n_bad/total_points)*100:.2f}%).")
    return n_bad

# Removed duplicate function - now using unified remove_outliers from utils


# old way to compute avg on the fly
# class ScenarioDataset:
#     def __init__(self, hf_ds, scenario="avg", label_key="label_idx"):
#         assert scenario in ("concat", "avg", "g", "r", "i", "z")  # Added more bands
#         self.ds, self.scenario, self.lkey = hf_ds, scenario, label_key
#     def __len__(self): return len(self.ds)
#     def __getitem__(self, idx):
#         rec = self.ds[idx]
#         # Use main unified embedding function - now handles pre-computed avg_embedding!
#         x_np = compute_embedding(rec, band_combination=self.scenario, hand_crafted=False, return_format="combined")
#         return x_np, rec[self.lkey]

def prepare_numpy_data(hf_train, hf_val, hf_test, scenario):
    """Prepare numpy arrays - ULTRA FAST with pre-computed combined_embedding!"""
    def to_numpy(hf_ds, name):
        start_time = time.time()
        
        # Check if we have pre-computed combined_embedding
        if len(hf_ds) > 0 and "combined_embedding" in hf_ds[0]:
            print(f"{name}: Using pre-computed combined_embedding")

            # Pull the whole column at once (fast path)
            col = hf_ds["combined_embedding"]            # list (or np array) of vectors
            # If HF stored fixed-size lists, this may already be an ndarray
            if isinstance(col, np.ndarray):
                print("combined_embedding is np array") # is not
                X = col.astype(np.float32, copy=False)   # shape: (N, D)
            else:
                # Avoid slow Python loop; np.asarray stacks in one go when shapes match
                X = np.asarray(col, dtype=np.float32)    # shape: (N, D)

            y = np.asarray(hf_ds["label_idx"])
            
        else:
            # Fallback to old method if combined_embedding not available
            print(f"[FALLBACK] {name}: combined_embedding not found, using compute_embedding with scenario '{scenario}'...")
            X, y = [], []
            
            for i in range(len(hf_ds)):
                rec = hf_ds[i]
                try:
                    x_np = compute_embedding(rec, band_combination=scenario, return_format="combined")
                    X.append(x_np)
                    y.append(rec["label_idx"])
                except Exception as e:
                    print(f"[Warning] Skipping sample {i}: {e}")
                    continue
            
            X = np.array(X)
            y = np.array(y)
        
        # Remove rows with NaN/Inf
        mask = np.all(np.isfinite(X), axis=1)
        removed = np.sum(~mask)
        if removed > 0:
            print(f"[Clean] {name}: Removed {removed}/{len(X)} samples with NaN/Inf ({removed/len(X)*100:.2f}%).")
        
        elapsed = time.time() - start_time
        print(f"[Timing] {name}: Processed {len(hf_ds)} samples in {elapsed:.2f}s ({len(hf_ds)/elapsed:.1f} samples/sec)")
        
        return X[mask], y[mask]
    
    X_train, y_train = to_numpy(hf_train, "train")
    X_val, y_val = to_numpy(hf_val, "val")
    X_test, y_test = to_numpy(hf_test, "test")
    return X_train, y_train, X_val, y_val, X_test, y_test

def plot_confusion(cm, text_labels, out_path):
    cmn = cm.astype(float) / cm.sum(1, keepdims=True)
    plt.figure(figsize=(8,6))
    sns.heatmap(cmn, annot=True, fmt=".3f", cmap="viridis",
                xticklabels=text_labels, yticklabels=text_labels)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ---------- Train classifiers --------------------------------------------
def train_logistic(X_train, y_train, X_test, y_test, out_dir, text_labels, random_state=42):
    print("\n=== Training Logistic Regression ===")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=random_state)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    pd.DataFrame(classification_report(y_test, preds, target_names=text_labels, output_dict=True)).T.to_csv(
        os.path.join(out_dir, "logistic_metrics_report.csv")
    )
    cm = confusion_matrix(y_test, preds, labels=list(range(len(text_labels))))
    with open(os.path.join(out_dir, "logistic_confusion.pkl"), "wb") as f:
        pickle.dump(cm, f)
    plot_confusion(cm, text_labels, os.path.join(out_dir, "logistic_confusion.pdf"))

def train_knn(X_train, y_train, X_test, y_test, out_dir, text_labels, k=5):
    print(f"\n=== Training kNN (k={k}) ===")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    pd.DataFrame(classification_report(y_test, preds, target_names=text_labels, output_dict=True)).T.to_csv(
        os.path.join(out_dir, "knn_metrics_report.csv")
    )
    cm = confusion_matrix(y_test, preds, labels=list(range(len(text_labels))))
    with open(os.path.join(out_dir, "knn_confusion.pkl"), "wb") as f:
        pickle.dump(cm, f)
    plot_confusion(cm, text_labels, os.path.join(out_dir, "knn_confusion.pdf"))

# ---------- Main ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="avg", 
                       choices=["concat", "avg", "g", "r", "i", "z"],
                       help="How to combine embeddings: concat, avg, or specific band (ignored if combined_embedding exists)")
    parser.add_argument("--input_embs", type=str, 
                       default="/projects/b1094/StarEmbed/embeddings/descriptive_name_embeddings/csdr1_raw_embs_moiral_small_trn_val_tst_ctx200_pdt64_psz16_bandgr")
    parser.add_argument("--out_dir", type=str, default=f"linear_results")
    parser.add_argument("--hand_crafted", type=bool, default=False)
    parser.add_argument("--k", type=int, default=5, help="k for kNN")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set all random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory under the specified base path
    base_output_dir = "/projects/b1094/StarEmbed/src/output/linear_classification/new_avg_embedding"
    experiment_name = f"{args.input_embs.split('/')[-1]}_{args.scenario}_seed{args.seed}"
    args.out_dir = os.path.join(base_output_dir, experiment_name)

    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Star loading dataset")
    # Load and clean dataset using unified utilities
    ds = load_from_disk(args.input_embs)
    ds = remove_outliers(ds, hand_crafted=args.hand_crafted)

    # Add label indices using unified utilities (works with any descriptive class names)
    ds, label2idx, text_labels = add_label_indices(ds, num_proc=4, sort_labels=True)
    
    print(f"Found classes: {text_labels}")
    print(f"Label mapping: {label2idx}")
    print("Train class distribution →", Counter(ds["train"]["label_idx"]))
    
    train_ds, val_ds, test_ds = ds["train"], ds["validation"], ds["test"]

    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_numpy_data(train_ds, val_ds, test_ds, args.scenario)
    X_train_full = np.concatenate([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    # Train & evaluate
    train_logistic(X_train_full, y_train_full, X_test, y_test, args.out_dir, text_labels, random_state=args.seed)
    train_knn(X_train_full, y_train_full, X_test, y_test, args.out_dir, text_labels, k=args.k)

    print("\nDone. Results saved in:", args.out_dir)
