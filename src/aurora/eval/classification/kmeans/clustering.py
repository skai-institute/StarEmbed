#!/usr/bin/env python3
"""
clustering_pipeline.py

A script to cluster embeddings and visualize t-SNE using a HuggingFace dataset on disk.

Usage:
    python clustering_pipeline.py  \
        --dataset-dir PATH/TO/DATASET  \
        [--perplexity 30]  \
        [--random-state 42]  \
        [--mode {train,test,both}]  \
        [--classes "all" or comma-separated list, e.g. "1,2,5"]
"""
import argparse
import numpy as np
from datasets import load_from_disk
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
    f1_score
)
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import pathlib
import os
from functools import partial
import logging
import traceback
# logger to save results to file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """
    Parse command-line arguments for dataset path, t-SNE settings, mode, and class subset.
    Returns args with dataset_dir, perplexity, random_state, mode, and classes.
    """
    p = argparse.ArgumentParser(description="Cluster & t-SNE embed pipeline")
    p.add_argument("--dataset-dir", type=str, required=True,
                   help="Path to HF DatasetDict on disk (train/validation/test)")
    p.add_argument("--perplexity", type=float, default=30,
                   help="t-SNE perplexity")
    p.add_argument("--random-state", type=int, default=42,
                   help="Seed for reproducibility")
    p.add_argument("--mode", choices=["train","test","both"], default="test",
                   help="Which split to process")
    p.add_argument("--classes", type=str, default="all",
                   help="Comma-separated list of original class labels to include, or 'all' for no filtering")
    p.add_argument("--save-dir", type=str, default=None,
                   help="Directory to save the t-SNE plots")
    p.add_argument("--concat-embs", type=int, default=1,
                   help="Concatenate embeddings of different bands, if not, do average")
    return p.parse_args()


def clustering_metrics(y_true, y_pred):
    """
    Compute ARI, NMI, and macro-F1 for clustering results.
    y_true: ground truth labels; y_pred: cluster assignments.
    Uses Hungarian matching to align cluster labels before F1.
    Returns (ari, nmi, f1).
    """
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {pred: true for true, pred in zip(row_ind, col_ind)}
    y_mapped = np.array([mapping[p] for p in y_pred])
    f1 = f1_score(y_true, y_mapped, average='macro')
    return ari, nmi, f1


def run_clustering(X, y, n_clusters, seed):
    """
    Run K-Means and Ward hierarchical clustering on X, y.
    Prints progress and returns metrics for both methods.
    """
    print(f"  → K-Means (k={n_clusters})")
    km = KMeans(n_clusters=n_clusters, random_state=seed).fit(X)
    km_met = clustering_metrics(y, km.labels_)
    print(f"  → Ward    (k={n_clusters})")
    Z = linkage(X, method='ward')
    hier = fcluster(Z, t=n_clusters, criterion='maxclust')
    hier_met = clustering_metrics(y, hier)
    return km_met, hier_met


def plot_tsne(X, labels, title, perplexity, seed, save_dir=None):
    """
    Compute and show t-SNE embedding for X, colored by labels.
    X: feature matrix; labels: class labels; title: plot title.
    """
    print(f"  → t-SNE ({title})")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed)
    X2 = tsne.fit_transform(X)
    plt.figure(figsize=(7,6))
    for lbl in np.unique(labels):
        mask = (labels == lbl)
        plt.scatter(X2[mask,0], X2[mask,1], s=30, label=str(lbl), alpha=0.7)
    plt.legend(title="Class", bbox_to_anchor=(1.02,1))
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    if save_dir:
        fname = f"{'-'.join(title.split(' '))}.pdf"
        print(f"Saving t-SNE plot to {os.path.join(save_dir, fname)}")
        plt.savefig(os.path.join(save_dir, fname))
        plt.close()
    else:
        print(f"Displaying t-SNE plot for {title}")
        plt.show()

def cal_avg_embedding(example, concat=False):
    # print(len(example['embeddings_g']))
    # print(f"example['embeddings_g']: {np.array(example['embeddings_g']).shape}, example['embeddings_r']: {np.array(example['embeddings_r']).shape}")
    # shape of example['embeddings_g'] is (1, 201, 256)
    if len(example['embeddings_g']) == 1:
        avg_embedding_g = np.mean(np.array(example['embeddings_g']).squeeze(0), axis=0)
        avg_embedding_r = np.mean(np.array(example['embeddings_r']).squeeze(0), axis=0)
    else:
        avg_embedding_g = np.mean(np.array(example['embeddings_g']), axis=0)
        avg_embedding_r = np.mean(np.array(example['embeddings_r']), axis=0)
    if concat:
        avg_embedding = np.concatenate([avg_embedding_g, avg_embedding_r])
        # print(f"avg_embedding_g: {avg_embedding_g.shape}, avg_embedding_r: {avg_embedding_r.shape}, avg_embedding: {avg_embedding.shape}")
    else:
        avg_embedding = np.mean(np.array([avg_embedding_g, avg_embedding_r]), axis=0)
        # print(f"avg_embedding_g: {avg_embedding_g.shape}, avg_embedding_r: {avg_embedding_r.shape}, avg_embedding: {avg_embedding.shape}")
    example['avg_embedding'] = avg_embedding
    return example

"""
split: train, Number of examples with nan: 1
[23082]
split: validation, Number of examples with nan: 1
[473]
split: test, Number of examples with nan: 1
[7880]
"""

def remove_outlier(dataset):
    # one outlier is found in each of the split of the data. This function is a patch to remove it without regenerating the whole dataset 
    bad_idx_trn = 23082  
    bad_idx_val = 473
    bad_idx_tst = 7880                     # single row you want gone
    keep_trn = list(range(bad_idx_trn)) + list(range(bad_idx_trn + 1, len(dataset['train'])))
    keep_val = list(range(bad_idx_val)) + list(range(bad_idx_val + 1, len(dataset['validation'])))
    keep_tst = list(range(bad_idx_tst)) + list(range(bad_idx_tst + 1, len(dataset['test'])))
    dataset['train'] = dataset['train'].select(keep_trn)
    dataset['validation'] = dataset['validation'].select(keep_val)
    dataset['test'] = dataset['test'].select(keep_tst)
    return dataset



def main():
    """
    Load dataset, remap labels, filter class subset, cluster splits, and plot t-SNE.
    """
    args = parse_args()
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    if args.save_dir:
        pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        input_emb_name = pathlib.Path(args.dataset_dir).name
        result_dir = f"s{args.random_state}p{args.perplexity}m{args.mode}c{args.concat_embs}i{input_emb_name}"
        pathlib.Path(args.save_dir, result_dir).mkdir(parents=True, exist_ok=True)
        result_dir = os.path.join(args.save_dir, result_dir)

    else:
        result_dir = None

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(result_dir, 'log.txt'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    print(f"Saving results to {result_dir}")
    logger.info(f"Saving results to {result_dir}")


    # Step 1: Load dataset
    print("Step 1: Loading dataset...")
    logger.info("Step 1: Loading dataset...")

    ds = load_from_disk(args.dataset_dir)
    ds = remove_outlier(ds)

    # only for testing
    for split in ds.keys():
        ds[split] = ds[split].select(range(100))

    # calculate the average embedding for clustering
    for split in ds.keys():
        ds[split] = ds[split].map(partial(cal_avg_embedding, concat=bool(args.concat_embs)), remove_columns=["bands_data"], num_proc=8)

    # Step 2: Remap labels and keep originals
    print("Step 2: Building label2idx mapping and saving original labels...")
    logger.info("Step 2: Building label2idx mapping and saving original labels...")
    orig_labels = sorted(set(ds['train']['class_str']))
    orig_labels = ['1', '13', '2', '4', '5', '6', '8'] # only for testing
    label2idx   = {lab: i for i, lab in enumerate(orig_labels)}
    def remap(ex):
        ex['label_idx'] = label2idx[ex['class_str']]
        return ex
    ds = ds.map(remap, num_proc=8)

    y_train_orig = np.array(ds['train']['class_str'])
    y_test_orig  = np.array(ds['test']['class_str'])

    # Step 3: Convert embeddings and idx to NumPy
    print("Step 3: Converting embeddings & labels to NumPy arrays...")
    logger.info("Step 3: Converting embeddings & labels to NumPy arrays...")
    ds.set_format(type='numpy', columns=['avg_embedding','label_idx'])
    X_train     = np.vstack(ds['train']['avg_embedding'])
    y_train_idx = np.array(ds['train']['label_idx'])
    X_test      = np.vstack(ds['test']['avg_embedding'])
    y_test_idx  = np.array(ds['test']['label_idx'])

    # Step 4: Filter by class subset
    if args.classes.lower() != 'all':
        # parse provided class labels
        tokens = [tok.strip() for tok in args.classes.split(',')]
        classes = []
        for tok in tokens:
            try:
                classes.append(int(tok))
            except ValueError:
                classes.append(tok)
        print(f"Step 4: Filtering to classes {classes}...")
        logger.info(f"Step 4: Filtering to classes {classes}...")
        mask_tr = np.isin(y_train_orig, classes)
        mask_te = np.isin(y_test_orig,  classes)
        X_train    = X_train[mask_tr]
        y_train_idx= y_train_idx[mask_tr]
        y_train_orig= y_train_orig[mask_tr]
        X_test     = X_test[mask_te]
        y_test_idx = y_test_idx[mask_te]
        y_test_orig= y_test_orig[mask_te]
    else:
        print("Step 4: No filtering (using all classes)")
        logger.info("Step 4: No filtering (using all classes)")

    # determine number of clusters after filtering
    n_clusters = len(np.unique(y_train_idx))

    try:

        # Step 5: Cluster TRAIN split
        if args.mode in ('train','both'):
            print("\nStep 5: Clustering TRAIN split")
            logger.info("Step 5: Clustering TRAIN split")
            km_tr, hier_tr = run_clustering(X_train, y_train_idx, n_clusters, args.random_state)
            km_tr_ari, km_tr_nmi, km_tr_f1 = km_tr
            h_tr_ari, h_tr_nmi, h_tr_f1    = hier_tr
            print(f"Train → K-Means  ARI={km_tr_ari:.4f}, NMI={km_tr_nmi:.4f}, F1={km_tr_f1:.4f}")
            print(f"Train → Ward      ARI={h_tr_ari:.4f}, NMI={h_tr_nmi:.4f}, F1={h_tr_f1:.4f}")
            logger.info(f"Train → K-Means  ARI={km_tr_ari:.4f}, NMI={km_tr_nmi:.4f}, F1={km_tr_f1:.4f}")
            logger.info(f"Train → Ward      ARI={h_tr_ari:.4f}, NMI={h_tr_nmi:.4f}, F1={h_tr_f1:.4f}")

        # Step 6: Cluster TEST split
        if args.mode in ('test','both'):
            print("\nStep 6: Clustering TEST split")
            km_te, hier_te = run_clustering(X_test, y_test_idx, n_clusters, args.random_state)
            km_te_ari, km_te_nmi, km_te_f1 = km_te
            h_te_ari, h_te_nmi, h_te_f1    = hier_te
            print(f"Test  → K-Means  ARI={km_te_ari:.4f}, NMI={km_te_nmi:.4f}, F1={km_te_f1:.4f}")
            print(f"Test  → Ward      ARI={h_te_ari:.4f}, NMI={h_te_nmi:.4f}, F1={h_te_f1:.4f}")
            logger.info(f"Test  → K-Means  ARI={km_te_ari:.4f}, NMI={km_te_nmi:.4f}, F1={km_te_f1:.4f}")
            logger.info(f"Test  → Ward      ARI={h_te_ari:.4f}, NMI={h_te_nmi:.4f}, F1={h_te_f1:.4f}")


        # Step 7: t-SNE TRAIN split
        if args.mode in ('train','both'):
            print("\nStep 7: t-SNE TRAIN (original labels)")
            logger.info("\nStep 7: t-SNE TRAIN (original labels)")
            plot_tsne(X_train, y_train_orig, "t-SNE on Filtered Train", args.perplexity, args.random_state, result_dir)

        # Step 8: t-SNE TEST split
        if args.mode in ('test','both'):
            print("\nStep 8: t-SNE TEST (original labels)")
            logger.info("\nStep 8: t-SNE TEST (original labels)")
            plot_tsne(X_test, y_test_orig, "t-SNE on Filtered Test", args.perplexity, args.random_state, result_dir)

        logger.info("Clustering pipeline completed successfully.")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        # also log the traceback
        logger.error(traceback.format_exc())
        raise e


if __name__ == '__main__':
    main()