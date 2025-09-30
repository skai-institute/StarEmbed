#!/usr/bin/env python3
"""
Plot comparison of all classifiers for different embedding methods.
Easy to switch between embedding methods by changing one variable.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

def load_confusion_matrix(file_path):
    """Load confusion matrix from pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def plot_comparison_subplots(cms, method_names, class_names, save_path):
    """Plot all confusion matrices in subplots for comparison."""
    n_methods = len(cms)
    
    if n_methods <= 3:
        fig, axes = plt.subplots(1, n_methods, figsize=(8*n_methods, 6))
        if n_methods == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
    
    for i, (cm, method) in enumerate(zip(cms, method_names)):
        cmn = cm.astype(float) / cm.sum(1, keepdims=True)
        
        sns.heatmap(cmn, annot=True, fmt=".3f", cmap="viridis",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[i], annot_kws={"size": 12})
        
        axes[i].set_title(f'{method} Classifier', fontsize=18, fontweight='bold')
        axes[i].set_xlabel("Predicted", fontsize=16)
        axes[i].set_ylabel("True", fontsize=16)
        axes[i].tick_params(axis='x', rotation=45, labelsize=16)
        axes[i].tick_params(axis='y', rotation=0, labelsize=16)
    
    # Hide unused subplots if any
    for i in range(len(cms), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # SPECIFY EMBEDDING METHOD HERE - Change this variable to switch methods
    embedding_method = "moirai"  # Options: astromer_1, astromer_2, chronos-bolt, handcrafted_feature, moirai, random
    
    print(f"Processing embedding method: {embedding_method}")
    
    # Class names mapping - matches the order used in all classifier scripts
    # Based on sorted class_str by integer: ['1', '2', '4', '5', '6', '8', '13']
    class_names = ["EW", "EA", "RRab", "RRc", "RRd", "RS CVn", "LPV"]
    
    # Base directory
    base_dir = "/projects/b1094/StarEmbed/src/output/mean_confusion_matrices"
    method_dir = os.path.join(base_dir, embedding_method)
    
    if not os.path.exists(method_dir):
        print(f"Error: Directory not found: {method_dir}")
        print(f"Available methods: {[d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]}")
        return
    
    # Required files - automatically construct paths based on embedding method
    files = {
        'KNN': os.path.join(method_dir, "knn_confusion.pkl"),
        'Logistic': os.path.join(method_dir, "logistic_confusion.pkl"), 
        'MLP': os.path.join(method_dir, "mean_mlp_confusion.pkl"),
        'Random Forest': None  # Will be determined based on method
    }
    
    # Find the RF confusion matrix file (has variable naming)
    try:
        rf_files = [f for f in os.listdir(method_dir) if f.endswith('rf_confusion_matrix_mean.pkl')]
        if rf_files:
            files['Random Forest'] = os.path.join(method_dir, rf_files[0])
            print(f"Found RF file: {rf_files[0]}")
        else:
            print(f"Warning: No RF confusion matrix found in {method_dir}")
    except Exception as e:
        print(f"Error accessing directory {method_dir}: {e}")
        return
    
    # Load matrices
    results = {}
    for method, file_path in files.items():
        if file_path and os.path.exists(file_path):
            try:
                if file_path.endswith('.npy'):
                    cm = np.load(file_path)
                else:
                    cm = load_confusion_matrix(file_path)
                results[method] = cm
                print(f"Loaded {method}: {cm.shape}")
            except Exception as e:
                print(f"Error loading {method} from {file_path}: {e}")
        elif file_path:
            print(f"Warning: {method} file not found: {file_path}")
        else:
            print(f"Warning: {method} file path not determined")
    
    if len(results) < 2:
        print("Error: Need at least 2 methods for comparison")
        print(f"Found methods: {list(results.keys())}")
        return
    
    # Create comparison plot
    method_order = ['KNN', 'Logistic', 'Random Forest', 'MLP']
    cms = []
    method_names = []
    
    for method in method_order:
        if method in results:
            cms.append(results[method])
            method_names.append(method)
    
    # Save comparison plot in the method directory
    comparison_path = os.path.join(method_dir, f"comparison_all_methods_{embedding_method}.pdf")
    plot_comparison_subplots(cms, method_names, class_names, comparison_path)
    print(f"Saved comparison plot to: {comparison_path}")
    
    # Print accuracy summary
    print("\nAccuracy Summary:")
    for method, cm in zip(method_names, cms):
        accuracy = np.trace(cm) / np.sum(cm)
        print(f"{method:15}: {accuracy:.4f} ({accuracy*100:.2f}%)")

if __name__ == "__main__":
    main()
