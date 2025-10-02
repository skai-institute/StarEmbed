#!/usr/bin/env python3
"""
Compute mean confusion matrix from three pickle files.
Specify the paths in the file below.
"""
import numpy as np
import pickle

# SPECIFY YOUR PATHS HERE
input_paths = [
    "/projects/p32795/weijian/skai_universal_forecaster/src/aurora/eval/classification/mlp/results_wloss_used_for_paper.lfs/csdr1_raw_embs_moiral_small_trn_val_tst_ctx200_pdt64_psz16_bandgr_bs64_lr0.001_do0.0_concat_s100/version_0/confusion_data.pkl",
    "/projects/p32795/weijian/skai_universal_forecaster/src/aurora/eval/classification/mlp/results_wloss_used_for_paper.lfs/csdr1_raw_embs_moiral_small_trn_val_tst_ctx200_pdt64_psz16_bandgr_bs64_lr0.001_do0.0_concat_s200/version_0/confusion_data.pkl", 
]

output_path = "/projects/b1094/StarEmbed/src/output/mean_confusion_matrices/moirai/mean_mlp_confusion.pkl"

def load_confusion_matrix(file_path):
    """Load confusion matrix from pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_confusion_matrix(cm, file_path):
    """Save confusion matrix to pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(cm, f)

def main():
    print("Loading confusion matrices...")
    
    input_files = input_paths
    output_file = output_path
    
    # Load confusion matrices
    matrices = []
    for i, file_path in enumerate(input_files):
        try:
            cm = load_confusion_matrix(file_path)
            matrices.append(cm)
            print(f"Loaded matrix {i+1}: {cm.shape}")
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            return
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return
    
    # Check if all matrices have the same shape
    shapes = [cm.shape for cm in matrices]
    if not all(shape == shapes[0] for shape in shapes):
        print(f"Error: All matrices must have the same shape. Found: {shapes}")
        return
    
    # Compute mean
    mean_cm = np.mean(matrices, axis=0)
    print(f"Computed mean confusion matrix: {mean_cm.shape}")
    
    # Save mean confusion matrix
    try:
        save_confusion_matrix(mean_cm, output_file)
        print(f"Saved mean confusion matrix to: {output_file}")
    except Exception as e:
        print(f"Error saving {output_file}: {e}")
        return
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"Total samples: {int(np.sum(mean_cm))}")
    print(f"Mean diagonal sum: {np.trace(mean_cm):.2f}")
    if np.sum(mean_cm) > 0:
        accuracy = np.trace(mean_cm) / np.sum(mean_cm)
        print(f"Overall accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

if __name__ == "__main__":
    main()
