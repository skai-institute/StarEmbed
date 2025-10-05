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
import sys
import os
from datasets import load_from_disk
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
    f1_score
)
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
import pathlib
import logging
import traceback

# Add src_clean to path for importing benchmark.utils  
script_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_dir = os.path.dirname(script_dir)
src_clean_dir = os.path.dirname(benchmark_dir)
sys.path.append(src_clean_dir)

from benchmark.utils import remove_outliers, add_label_indices, compute_embedding_batch

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
    p.add_argument("--mode", choices=["train","test","validation","both","all"], default="test",
                   help="Which split to process (all combines train+validation+test into one dataset)")
    p.add_argument("--classes", type=str, default="all",
                   help="Comma-separated list of original class labels to include, or 'all' for no filtering")
    p.add_argument("--scenario", type=str, default="concat", 
                   choices=["concat", "avg", "g", "r", "i", "z"],
                   help="How to combine multi-band embeddings: concat, avg, or specific band (default: concat)")
    p.add_argument("--hand-crafted", type=int, default=0, help="Use hand-crafted features for clustering")
    p.add_argument("--standardize", type=int, default=0, 
                   help="Apply StandardScaler to features before clustering and t-SNE (0=False, 1=True)")
    p.add_argument("--output-dir", type=str, default="/projects/b1094/StarEmbed/src/output/clustering",
                   help="Base output directory for results (default: /projects/b1094/StarEmbed/src/output/clustering)")
    p.add_argument("--clustering-method", choices=["kmeans", "hierarchical", "both"], default="both",
                   help="Which clustering method to run (default: both)")
    p.add_argument("--save-dendrogram", action="store_true",
                   help="Save dendrogram plot for hierarchical clustering")
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


def run_clustering(X, y, n_clusters, seed, method="both", save_dendrogram=False, save_dir=None, split_name=""):
    """
    Run K-Means and/or Ward hierarchical clustering on X, y.
    Prints progress and returns metrics for requested methods.
    
    Args:
        X: feature matrix
        y: true labels
        n_clusters: number of clusters
        seed: random seed
        method: "kmeans", "hierarchical", or "both"
        save_dendrogram: whether to save dendrogram for hierarchical clustering
        save_dir: directory to save dendrogram
        split_name: name of the split for dendrogram filename
    
    Returns:
        tuple of (km_metrics, hier_metrics) where each is None if method not run
    """
    km_met = None
    hier_met = None
    
    if method in ["kmeans", "both"]:
        print(f"  → K-Means (k={n_clusters})")
        km = KMeans(n_clusters=n_clusters, random_state=seed).fit(X)
        km_met = clustering_metrics(y, km.labels_)
    
    if method in ["hierarchical", "both"]:
        print(f"  → Ward    (k={n_clusters})")
        Z = linkage(X, method='ward')
        hier = fcluster(Z, t=n_clusters, criterion='maxclust')
        hier_met = clustering_metrics(y, hier)
        
        # Save dendrogram if requested
        if save_dendrogram and save_dir:
            plt.figure(figsize=(12, 8))
            dn = dendrogram(Z, 
                      truncate_mode='lastp',
                      p=30,  # show last 30 merges
                      show_leaf_counts=True,
                      leaf_rotation=90)
            
            # Add horizontal line showing cut for desired number of clusters
            # Find the linkage distance that gives n_clusters
            distances = sorted(set(Z[:, 2]))
            for dist in reversed(distances):
                temp_clusters = fcluster(Z, t=dist, criterion='distance')
                if len(set(temp_clusters)) == n_clusters:
                    plt.axhline(y=dist, color='red', linestyle='--', linewidth=2, 
                              label=f'Cut for {n_clusters} clusters')
                    break
            
            plt.title(f'Hierarchical Clustering Dendrogram - {split_name}')
            plt.xlabel('Cluster Size (or Index)')
            plt.ylabel('Distance')
            plt.legend()
            
            dendrogram_file = os.path.join(save_dir, f'dendrogram_{split_name.lower().replace(" ", "_")}.pdf')
            plt.savefig(dendrogram_file, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"  → Dendrogram saved to {dendrogram_file}")
    
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

# Note: cal_avg_embedding function is now imported from utils.py for unified processing

"""
split: train, Number of examples with nan: 1
[23082]
split: validation, Number of examples with nan: 1
[473]
split: test, Number of examples with nan: 1
[7880]
"""

def main():
    """
    Load dataset, remap labels, filter class subset, cluster splits, and plot t-SNE.
    """
    args = parse_args()
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    # Create output directory - use the specified path directly
    input_emb_name = pathlib.Path(args.dataset_dir).name
    experiment_name = f"{input_emb_name}_{args.mode}_{args.scenario}_std{args.standardize}_p{args.perplexity}_seed{args.random_state}"
    result_dir = os.path.join(args.output_dir, experiment_name)
    
    # Create the directory structure
    pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)

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
    ds = remove_outliers(ds, hand_crafted=bool(args.hand_crafted))

    # Use pre-computed combined_embedding (must exist from compute_avg_embeddings.py)
    print("Using pre-computed combined_embedding")


    # Step 2: Add label indices using unified utility
    print("Step 2: Building label2idx mapping and saving original labels...")
    logger.info("Step 2: Building label2idx mapping and saving original labels...")
    
    # add "label_idx" column to the dataset splits
    ds, label2idx, orig_labels = add_label_indices(ds, num_proc=8, sort_labels=True)
    print(f"Automatic label order: {orig_labels}")
    logger.info(f"Automatic label order: {orig_labels}")

    y_train_orig = np.array(ds['train']['class_str'])
    y_test_orig  = np.array(ds['test']['class_str'])
    y_valid_orig = np.array(ds['validation']['class_str'])

    # Step 3: Convert embeddings and idx to NumPy
    print("Step 3: Converting embeddings & labels to NumPy arrays...")
    logger.info("Step 3: Converting embeddings & labels to NumPy arrays...")
    
    # Only set format for splits that have label_idx (standard splits)
    standard_splits = [split for split in ds.keys() if split in ['train', 'validation', 'test']]
    for split in standard_splits:
        ds[split].set_format(type='numpy', columns=['combined_embedding','label_idx'])
    
    # Extract embeddings directly from combined_embedding
    X_train = np.array(ds['train']['combined_embedding'], dtype=np.float32)
    X_test = np.array(ds['test']['combined_embedding'], dtype=np.float32) 
    X_valid = np.array(ds['validation']['combined_embedding'], dtype=np.float32)
    
    y_train_idx = np.array(ds['train']['label_idx'])
    y_test_idx  = np.array(ds['test']['label_idx'])
    y_valid_idx = np.array(ds['validation']['label_idx'])

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
        mask_val = np.isin(y_valid_orig, classes)

        X_train    = X_train[mask_tr]
        y_train_idx= y_train_idx[mask_tr]
        y_train_orig= y_train_orig[mask_tr]
        X_test     = X_test[mask_te]
        y_test_idx = y_test_idx[mask_te]
        y_test_orig= y_test_orig[mask_te]
        X_valid    = X_valid[mask_val]
        y_valid_idx = y_valid_idx[mask_val]
        y_valid_orig= y_valid_orig[mask_val]

    else:
        print("Step 4: No filtering (using all classes)")
        logger.info("Step 4: No filtering (using all classes)")

    # determine number of clusters after filtering
    n_clusters = len(np.unique(y_train_idx))

    # Step 4.5: Apply standardization if requested
    if args.standardize:
        print("Step 4.5: Applying StandardScaler to features...")
        logger.info("Step 4.5: Applying StandardScaler to features...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_valid = scaler.transform(X_valid)
        print(f"Features standardized. Train shape: {X_train.shape}, Test shape: {X_test.shape}, Valid shape: {X_valid.shape}")
        logger.info(f"Features standardized. Train shape: {X_train.shape}, Test shape: {X_test.shape}, Valid shape: {X_valid.shape}")
    else:
        print("Step 4.5: Skipping standardization")
        logger.info("Step 4.5: Skipping standardization")

    # Step 4.6: Combine all splits if mode is 'all'
    if args.mode == 'all':
        print("Step 4.6: Combining all splits (train+validation+test) for clustering...")
        logger.info("Step 4.6: Combining all splits (train+validation+test) for clustering...")
        X_all = np.vstack([X_train, X_valid, X_test])
        y_all_idx = np.concatenate([y_train_idx, y_valid_idx, y_test_idx])
        y_all_orig = np.concatenate([y_train_orig, y_valid_orig, y_test_orig])
        print(f"Combined dataset shape: {X_all.shape}")
        logger.info(f"Combined dataset shape: {X_all.shape}")
    else:
        print("Step 4.6: Using individual splits")
        logger.info("Step 4.6: Using individual splits")

    try:

        # Step 5: Cluster ALL splits combined
        if args.mode == 'all':
            print("\nStep 5: Clustering ALL splits combined (train+validation+test)")
            logger.info("Step 5: Clustering ALL splits combined (train+validation+test)")
            km_all, hier_all = run_clustering(X_all, y_all_idx, n_clusters, args.random_state, 
                                            args.clustering_method, args.save_dendrogram, 
                                            result_dir, "All Splits")
            
            if km_all:
                km_all_ari, km_all_nmi, km_all_f1 = km_all
                print(f"All   → K-Means  ARI={km_all_ari:.4f}, NMI={km_all_nmi:.4f}, F1={km_all_f1:.4f}")
                logger.info(f"All   → K-Means  ARI={km_all_ari:.4f}, NMI={km_all_nmi:.4f}, F1={km_all_f1:.4f}")
            
            if hier_all:
                h_all_ari, h_all_nmi, h_all_f1 = hier_all
                print(f"All   → Ward      ARI={h_all_ari:.4f}, NMI={h_all_nmi:.4f}, F1={h_all_f1:.4f}")
                logger.info(f"All   → Ward      ARI={h_all_ari:.4f}, NMI={h_all_nmi:.4f}, F1={h_all_f1:.4f}")

        # Step 6: Cluster TRAIN split
        if args.mode in ('train','both'):
            print("\nStep 6: Clustering TRAIN split")
            logger.info("Step 6: Clustering TRAIN split")
            km_tr, hier_tr = run_clustering(X_train, y_train_idx, n_clusters, args.random_state,
                                          args.clustering_method, args.save_dendrogram, 
                                          result_dir, "Train Split")
            
            if km_tr:
                km_tr_ari, km_tr_nmi, km_tr_f1 = km_tr
                print(f"Train → K-Means  ARI={km_tr_ari:.4f}, NMI={km_tr_nmi:.4f}, F1={km_tr_f1:.4f}")
                logger.info(f"Train → K-Means  ARI={km_tr_ari:.4f}, NMI={km_tr_nmi:.4f}, F1={km_tr_f1:.4f}")
            
            if hier_tr:
                h_tr_ari, h_tr_nmi, h_tr_f1 = hier_tr
                print(f"Train → Ward      ARI={h_tr_ari:.4f}, NMI={h_tr_nmi:.4f}, F1={h_tr_f1:.4f}")
                logger.info(f"Train → Ward      ARI={h_tr_ari:.4f}, NMI={h_tr_nmi:.4f}, F1={h_tr_f1:.4f}")

        # Step 7: Cluster VALIDATION split
        if args.mode == 'validation':
            print("\nStep 7: Clustering VALIDATION split")
            logger.info("Step 7: Clustering VALIDATION split")
            km_val, hier_val = run_clustering(X_valid, y_valid_idx, n_clusters, args.random_state,
                                            args.clustering_method, args.save_dendrogram, 
                                            result_dir, "Validation Split")
            
            if km_val:
                km_val_ari, km_val_nmi, km_val_f1 = km_val
                print(f"Valid → K-Means  ARI={km_val_ari:.4f}, NMI={km_val_nmi:.4f}, F1={km_val_f1:.4f}")
                logger.info(f"Valid → K-Means  ARI={km_val_ari:.4f}, NMI={km_val_nmi:.4f}, F1={km_val_f1:.4f}")
            
            if hier_val:
                h_val_ari, h_val_nmi, h_val_f1 = hier_val
                print(f"Valid → Ward      ARI={h_val_ari:.4f}, NMI={h_val_nmi:.4f}, F1={h_val_f1:.4f}")
                logger.info(f"Valid → Ward      ARI={h_val_ari:.4f}, NMI={h_val_nmi:.4f}, F1={h_val_f1:.4f}")

        # Step 8: Cluster TEST split
        if args.mode in ('test','both'):
            print("\nStep 8: Clustering TEST split")
            logger.info("Step 8: Clustering TEST split")
            km_te, hier_te = run_clustering(X_test, y_test_idx, n_clusters, args.random_state,
                                          args.clustering_method, args.save_dendrogram, 
                                          result_dir, "Test Split")
            
            if km_te:
                km_te_ari, km_te_nmi, km_te_f1 = km_te
                print(f"Test  → K-Means  ARI={km_te_ari:.4f}, NMI={km_te_nmi:.4f}, F1={km_te_f1:.4f}")
                logger.info(f"Test  → K-Means  ARI={km_te_ari:.4f}, NMI={km_te_nmi:.4f}, F1={km_te_f1:.4f}")
            
            if hier_te:
                h_te_ari, h_te_nmi, h_te_f1 = hier_te
                print(f"Test  → Ward      ARI={h_te_ari:.4f}, NMI={h_te_nmi:.4f}, F1={h_te_f1:.4f}")
                logger.info(f"Test  → Ward      ARI={h_te_ari:.4f}, NMI={h_te_nmi:.4f}, F1={h_te_f1:.4f}")


        # Step 9: t-SNE ALL splits combined
        if args.mode == 'all':
            print("\nStep 9: t-SNE ALL splits combined (original labels)")
            logger.info("Step 9: t-SNE ALL splits combined (original labels)")
            plot_tsne(X_all, y_all_orig, "t-SNE on Combined Dataset", args.perplexity, args.random_state, result_dir)

        # Step 10: t-SNE TRAIN split
        if args.mode in ('train','both'):
            print("\nStep 10: t-SNE TRAIN (original labels)")
            logger.info("Step 10: t-SNE TRAIN (original labels)")
            plot_tsne(X_train, y_train_orig, "t-SNE on Filtered Train", args.perplexity, args.random_state, result_dir)

        # Step 11: t-SNE VALIDATION split
        if args.mode == 'validation':
            print("\nStep 11: t-SNE VALIDATION (original labels)")
            logger.info("Step 11: t-SNE VALIDATION (original labels)")
            plot_tsne(X_valid, y_valid_orig, "t-SNE on Filtered Validation", args.perplexity, args.random_state, result_dir)

        # Step 12: t-SNE TEST split
        if args.mode in ('test','both'):
            print("\nStep 12: t-SNE TEST (original labels)")
            logger.info("Step 12: t-SNE TEST (original labels)")
            plot_tsne(X_test, y_test_orig, "t-SNE on Filtered Test", args.perplexity, args.random_state, result_dir)

        logger.info("Clustering pipeline completed successfully.")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        # also log the traceback
        logger.error(traceback.format_exc())
        raise e


if __name__ == '__main__':
    main()