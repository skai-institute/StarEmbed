# linear_models.py
import os
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

def check_nan_stats(X, name="dataset"):
    total_points = X.shape[0]
    nan_mask = ~np.isfinite(X)
    rows_with_nan = np.any(nan_mask, axis=1)
    n_bad = np.sum(rows_with_nan)
    print(f"[NaN Check] {name}: {n_bad}/{total_points} samples contain NaN/Inf ({(n_bad/total_points)*100:.2f}%).")
    return n_bad

def remove_outlier(dataset, hand_crafted=False):
    if not hand_crafted:
        print("Removing outliers from dataset")
        bad_idx_trn, bad_idx_val, bad_idx_tst = 23082, 473, 7880
        trn_idx_to_select = list(range(bad_idx_trn)) + list(range(bad_idx_trn+1,len(dataset["train"]))) 
        val_idx_to_select = list(range(bad_idx_val)) + list(range(bad_idx_val+1,len(dataset["validation"]))) 
        tst_idx_to_select = list(range(bad_idx_tst)) + list(range(bad_idx_tst+1,len(dataset["test"])))
    else:
        print("Removing outliers from hand-crafted dataset")
        bad_idx_trn, bad_idx_val, bad_idx_tst = [3010, 9693, 16524, 22151], [449], [1158]
        trn_idx_to_select = list(sorted(set(range(len(dataset["train"]))) - set(bad_idx_trn)))
        val_idx_to_select = list(sorted(set(range(len(dataset["validation"]))) - set(bad_idx_val)))
        tst_idx_to_select = list(sorted(set(range(len(dataset["test"]))) - set(bad_idx_tst)))
    dataset["train"]      = dataset["train"].select(trn_idx_to_select)
    dataset["validation"] = dataset["validation"].select(val_idx_to_select)
    dataset["test"]       = dataset["test"].select(tst_idx_to_select)
    print(f"selected {len(dataset['train'])} train, {len(dataset['validation'])} val, {len(dataset['test'])} test samples")
    return dataset

class ScenarioDataset:
    def __init__(self, hf_ds, scenario="avg", label_key="label_idx"):
        assert scenario in ("concat", "avg", "g", "r")
        self.ds, self.scenario, self.lkey = hf_ds, scenario, label_key
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        rec   = self.ds[idx]
        if "embeddings_g" in rec:
            emb_g = np.squeeze(np.array(rec["embeddings_g"], dtype=np.float32))
            emb_r = np.squeeze(np.array(rec["embeddings_r"], dtype=np.float32))
            if emb_g.ndim > 1:
                avg_g, avg_r = emb_g.mean(0), emb_r.mean(0)
            else:
                avg_g, avg_r = emb_g, emb_r
        else:
            emb_g = np.squeeze(np.array(rec["g_embedding"], dtype=np.float32))
            emb_r = np.squeeze(np.array(rec["r_embedding"], dtype=np.float32))
            avg_g, avg_r = emb_g, emb_r
        if   self.scenario == "concat": x_np = np.concatenate([avg_g, avg_r], 0)
        elif self.scenario == "avg":    x_np = 0.5 * (avg_g + avg_r)
        elif self.scenario == "g":      x_np = avg_g
        else:                           x_np = avg_r
        return x_np, rec[self.lkey]

def prepare_numpy_data(hf_train, hf_val, hf_test, scenario):
    def to_numpy(hf_ds, name):
        X, y = [], []
        for rec in ScenarioDataset(hf_ds, scenario):
            X.append(rec[0])
            y.append(rec[1])
        X = np.array(X)
        y = np.array(y)
        # Remove rows with NaN/Inf
        mask = np.all(np.isfinite(X), axis=1)
        removed = np.sum(~mask)
        if removed > 0:
            print(f"[Clean] {name}: Removed {removed}/{len(X)} samples with NaN/Inf ({removed/len(X)*100:.2f}%).")
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
    parser.add_argument("--scenario", type=str, default="avg")
    parser.add_argument("--input_embs", type=str, default="/projects/p32795/weijian/embs/csdr1_raw_embs_moiral_small_trn_val_tst_ctx200_pdt64_psz16_bandgr")
    parser.add_argument("--out_dir", type=str, default=f"linear_results")
    parser.add_argument("--hand_crafted", type=int, default=0)
    parser.add_argument("--k", type=int, default=5, help="k for kNN")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set all random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory under the specified base path
    base_output_dir = "/projects/b1094/StarEmbed/src/output/linear_classification"
    experiment_name = f"{args.input_embs.split('/')[-1]}_{args.scenario}_seed{args.seed}"
    args.out_dir = os.path.join(base_output_dir, experiment_name)

    os.makedirs(args.out_dir, exist_ok=True)
    ds = remove_outlier(load_from_disk(args.input_embs), args.hand_crafted)
    train_ds, val_ds, test_ds = ds["train"], ds["validation"], ds["test"]

    # label remap
    orig_labels   = sorted(set(train_ds["class_str"]), key=lambda s: int(s))
    label2idx     = {lab: i for i, lab in enumerate(orig_labels)}
    class_name_map = {
        "1":  "EW",
        "2":  "EA",
        "4":  "RRab",
        "5":  "RRc",
        "6":  "RRd",
        "8":  "RS CVn",
        "13": "LPV"
    }
    def add_label(example, mapping): return {"label_idx": mapping[example["class_str"]]}
    text_labels = [ class_name_map[c] for c in orig_labels ]
    train_ds = train_ds.map(partial(add_label, mapping=label2idx), num_proc=4)
    val_ds   = val_ds.map  (partial(add_label, mapping=label2idx), num_proc=2)
    test_ds  = test_ds.map (partial(add_label, mapping=label2idx), num_proc=2)
    print("Train class distribution →", Counter(train_ds["label_idx"]))

    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_numpy_data(train_ds, val_ds, test_ds, args.scenario)
    X_train_full = np.concatenate([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    # Train & evaluate
    train_logistic(X_train_full, y_train_full, X_test, y_test, args.out_dir, text_labels, random_state=args.seed)
    train_knn(X_train_full, y_train_full, X_test, y_test, args.out_dir, text_labels, k=args.k)

    print("\nDone. Results saved in:", args.out_dir)
