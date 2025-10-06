import numpy as np
import os
import json
import pathlib
import argparse
import sys
import time
from datetime import datetime
from functools import partial

from datasets import load_from_disk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid, GridSearchCV, PredefinedSplit
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

# Add src_clean to path for importing benchmark.utils
script_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_dir = os.path.dirname(script_dir)
src_clean_dir = os.path.dirname(benchmark_dir)
sys.path.append(src_clean_dir)

from benchmark.utils import remove_outliers, add_label_indices


def parse_args():
    """
    Parse command-line arguments for input paths, standardization, and hand-crafted features.
    """
    p = argparse.ArgumentParser(description="Random Forest HPO pipeline")
    p.add_argument("--input-embs", type=str, nargs='+', required=True,
                   help="Paths to HF DatasetDict on disk (can specify multiple)")
    p.add_argument("--standardize", type=int, default=0, 
                   help="Apply StandardScaler to features before training (0=False, 1=True)")
    p.add_argument("--hand-crafted", type=int, default=0, 
                   help="Use hand-crafted features (0=False, 1=True)")
    p.add_argument("--skip-hpo", action="store_true", 
                   help="Skip hyperparameter optimization and use provided best parameters")
    p.add_argument("--best-params", type=str, default=None,
                   help="JSON string with best parameters (e.g., '{\"n_estimators\": 200, \"max_depth\": 20}')")
    p.add_argument("--seed", type=int, default=42, 
                   help="Random seed for reproducibility")
    p.add_argument("--output-dir", type=str, default="/projects/b1094/StarEmbed/src/output/rf",
                   help="Output directory for results (will be created if it doesn't exist)")
    p.add_argument("--scenario", type=str, default="concat", 
                   choices=["concat", "avg", "g", "r", "i", "z"],
                   help="How to combine multi-band embeddings (default: concat)")
    return p.parse_args()


def plot_confusion_matrix(cm, text_labels, out_path):
    """
    Plot and save confusion matrix as heatmap.
    """
    cmn = cm.astype(float) / cm.sum(1, keepdims=True)
    plt.figure(figsize=(8,6))
    sns.heatmap(cmn, annot=True, fmt=".3f", cmap="viridis",
                xticklabels=text_labels, yticklabels=text_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# input_embs = [
# "/projects/b1094/StarEmbed/embeddings/embeddings_with_anom/hf_csdr1_multiband_raw4_embeddings_astromer_2_gr_sampling_True"
# # "/projects/b1094/StarEmbed/embeddings/csdr1_raw4_catflags_filtered_embs_chronos_t5_tiny_trn_val_tst_ctx200_bandgr",
# # "/projects/p32795/dennis/random",
# # "/projects/b1094/StarEmbed/embeddings/embeddings_with_anom/hf_csdr1_multiband_raw4_embeddings_astromer_1_subclass_pad_correct",
# # "/projects/b1094/StarEmbed/embeddings/embeddings_with_anom/csdr1_raw4_catflags_filtered_embs_chronos_bolt_tiny_trn_val_tst_ctx200_bandgr",
# # "/projects/b1094/StarEmbed/embeddings/embeddings_with_anom/csdr1_raw_embs_moiral_small_trn_val_tst_ctx200_pdt64_psz16_bandgr", 
# # "/projects/b1094/StarEmbed/embeddings/csdr1_raw4_catflags_filtered_embs_hand_crafted_trn_val_tst_bandgr",
# ]

param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
}

def train_model(X_train, y_train, seed, params):

    model = RandomForestClassifier(
                    random_state=seed,
                    **params
                )
    model.fit(X_train, y_train)

    return model






def main():

    # Parse command-line arguments
    args = parse_args()
    
    # Set seeds based on the main seed
    seeds = [args.seed, args.seed + 58, args.seed + 158]
    
    # Create output directory
    output_dir = args.output_dir
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Initialize results storage
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for f in args.input_embs:

        print(f"================ input embs: {f} ================")
        print(f"Hand-crafted features: {bool(args.hand_crafted)}")
        print(f"Standardization: {bool(args.standardize)}")
        print(f"Skip HPO: {args.skip_hpo}")
        print(f"Random seed: {args.seed}")

        # print(f)
        
        ds = load_from_disk(f)
        
        # Remove outliers
        ds = remove_outliers(ds, hand_crafted=bool(args.hand_crafted))

        # Add label indices using unified utility
        ds, label2idx, text_labels = add_label_indices(ds, num_proc=8, sort_labels=True)
        
        # Check for pre-computed combined_embedding
        if len(ds['train']) > 0 and "combined_embedding" in ds['train'][0]:
            print("✓ Found pre-computed combined_embedding - using ULTRA FAST mode!")
        else:
            print(f"⚠ No combined_embedding found - script requires pre-computed embeddings")
            print(f"  → Run compute_avg_embeddings.py --dataset {os.path.basename(f)} --band_combination {args.scenario} first")
            continue  # Skip this dataset

        # Set format for fast access
        format_columns = ["combined_embedding", "class_str", "label_idx"]  
        for split in ['train', 'validation', 'test']:
            if split in ds:
                ds[split].set_format(type="numpy", columns=format_columns)
                
        def batched_xy(split):
            """
            Returns X, y as 2-D (n_samples, dim) and 1-D (n_samples,) NumPy arrays.
            Uses pre-computed combined_embedding for maximum speed.
            """
            import time
            start_time = time.time()
            
            # Direct access to pre-computed combined embeddings
            X = np.array(ds[split]["combined_embedding"], dtype=np.float32)
            y = ds[split]["label_idx"]
            
            elapsed = time.time() - start_time
            print(f"Split '{split}': Loaded {X.shape} in {elapsed:.2f}s ({len(X)/elapsed:.1f} samples/sec)")
            return X, y

        X_train, y_train = batched_xy("train")
        X_val,   y_val   = batched_xy("validation")
        X_test,  y_test  = batched_xy("test")

        # Apply standardization if requested
        if args.standardize:
            print("Applying StandardScaler to features...")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            print("Features standardized")

        # Stack the features and labels for HPO if needed
        X_all = np.vstack([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])

        # Determine best parameters
        if args.skip_hpo:
            # Use provided best parameters
            if args.best_params:
                try:
                    # Handle Python None vs JSON null
                    json_str = args.best_params.replace('None', 'null')
                    best_params = json.loads(json_str)
                    print(f"Using provided best parameters: {best_params}")
                    best_cv_score = None  # No CV score available
                except json.JSONDecodeError as e:
                    print(f"Error: Invalid JSON format for best_params: {e}")
                    print(f"Provided: {args.best_params}")
                    print("Using default parameters.")
                    best_params = {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 2}
                    best_cv_score = None
            else:
                # Use default best parameters
                best_params = {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 2}
                best_cv_score = None
                print(f"No best parameters provided. Using default: {best_params}")
        else:
            # Run hyperparameter optimization
            print("Running hyperparameter optimization...")
            
            # Create the test_fold array: -1 = train, 0 = validation
            test_fold = np.concatenate([
                -1 * np.ones(len(X_train), dtype=int),  # training indices
                0 * np.ones(len(X_val), dtype=int)     # validation indices
            ])

            # Create the PredefinedSplit object
            ps = PredefinedSplit(test_fold)

            # Base estimator
            rf = RandomForestClassifier(random_state=args.seed)

            grid = GridSearchCV(
                estimator=rf,
                param_grid=param_grid,
                cv=ps,
                scoring='f1_macro',
                n_jobs=-1,
                verbose=3
            )

            grid.fit(X_all, y_all)
            best_params = grid.best_params_
            best_cv_score = float(grid.best_score_)
            print(f"Best params: {best_params}")
            print(f"Best CV score: {best_cv_score}")

        # Train models with different seeds and evaluate
        # Train final model on full training data (train + val)
        X_train_full = np.concatenate([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])
        
        # Store classification reports and metrics for each seed
        all_classification_reports = []
        all_confusion_matrices = []
        f1_arr = []
        acc_arr = []
        precision_arr = []
        recall_arr = []
        
        emb_name = os.path.basename(f)
        
        # Train and evaluate model for each seed
        for i, s in enumerate(seeds):
            print(f"Training model with seed {s}...")
            
            model = RandomForestClassifier(
                random_state=s,
                **best_params,
            )
            model.fit(X_train_full, y_train_full)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='macro', zero_division=0
            )
            
            acc_arr.append(acc)
            f1_arr.append(f1)
            precision_arr.append(precision)
            recall_arr.append(recall)
            
            # Generate classification report for this seed
            class_report = classification_report(y_test, y_pred, target_names=text_labels, output_dict=True)
            all_classification_reports.append(class_report)
            
            # Save individual classification report
            report_df = pd.DataFrame(class_report).T
            individual_report_file = os.path.join(output_dir, f"{emb_name}_rf_classification_report_seed{s}.csv")
            report_df.to_csv(individual_report_file)
            
            # Generate confusion matrix for this seed
            cm = confusion_matrix(y_test, y_pred, labels=list(range(len(text_labels))))
            all_confusion_matrices.append(cm)
            
            # Save individual confusion matrix
            cm_file = os.path.join(output_dir, f"{emb_name}_rf_confusion_matrix_seed{s}.pkl")
            with open(cm_file, "wb") as f_cm:
                pickle.dump(cm, f_cm)
            
            # Plot individual confusion matrix
            cm_plot_file = os.path.join(output_dir, f"{emb_name}_rf_confusion_matrix_seed{s}.pdf")
            plot_confusion_matrix(cm, text_labels, cm_plot_file)
            
            print(f"Seed {s} - F1: {f1:.4f}, Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        # Compute mean classification report across all seeds
        mean_report = {}
        std_report = {}
        
        # Get all unique keys from classification reports (classes + metrics)
        all_keys = set()
        for report in all_classification_reports:
            all_keys.update(report.keys())
        
        for key in all_keys:
            if key in ['accuracy']:  # accuracy is a single value
                values = [report[key] for report in all_classification_reports if key in report]
                mean_report[key] = np.mean(values)
                std_report[key] = np.std(values)
            elif isinstance(all_classification_reports[0].get(key), dict):  # class-specific metrics
                mean_report[key] = {}
                std_report[key] = {}
                metric_keys = set()
                for report in all_classification_reports:
                    if key in report:
                        metric_keys.update(report[key].keys())
                
                for metric in metric_keys:
                    values = [report[key][metric] for report in all_classification_reports 
                             if key in report and metric in report[key]]
                    if values:
                        mean_report[key][metric] = np.mean(values)
                        std_report[key][metric] = np.std(values)
        
        # Save mean classification report
        mean_report_df = pd.DataFrame(mean_report).T
        mean_report_file = os.path.join(output_dir, f"{emb_name}_rf_classification_report_mean.csv")
        mean_report_df.to_csv(mean_report_file)
        
        # Save std classification report
        std_report_df = pd.DataFrame(std_report).T
        std_report_file = os.path.join(output_dir, f"{emb_name}_rf_classification_report_std.csv")
        std_report_df.to_csv(std_report_file)
        
        # Compute and save mean confusion matrix
        mean_cm = np.mean(all_confusion_matrices, axis=0)
        std_cm = np.std(all_confusion_matrices, axis=0)
        
        mean_cm_file = os.path.join(output_dir, f"{emb_name}_rf_confusion_matrix_mean.pkl")
        with open(mean_cm_file, "wb") as f_cm:
            pickle.dump(mean_cm, f_cm)
            
        std_cm_file = os.path.join(output_dir, f"{emb_name}_rf_confusion_matrix_std.pkl")
        with open(std_cm_file, "wb") as f_cm:
            pickle.dump(std_cm, f_cm)
        
        # Plot mean confusion matrix
        mean_cm_plot_file = os.path.join(output_dir, f"{emb_name}_rf_confusion_matrix_mean.pdf")
        plot_confusion_matrix(mean_cm, text_labels, mean_cm_plot_file)
        
        print(f"Saved individual reports for seeds: {seeds}")
        print(f"Saved mean classification report to: {mean_report_file}")
        print(f"Saved std classification report to: {std_report_file}")
        print(f"Saved mean confusion matrix plot to: {mean_cm_plot_file}")

        print(f"Overall Results across {len(seeds)} seeds:")
        print('f1: ', np.mean(f1_arr), '|',np.std(f1_arr))
        print('acc: ',np.mean(acc_arr),'|', np.std(acc_arr))
        print('precision: ', np.mean(precision_arr),'|', np.std(precision_arr))
        print('recall: ', np.mean(recall_arr),'|', np.std(recall_arr))

        print('-------------------------------')

        # Store results for this embedding
        emb_name = os.path.basename(f)
        all_results[emb_name] = {
            'embedding_path': f,
            'best_params': best_params,
            'best_cv_score': best_cv_score,  # Will be None if HPO was skipped
            'hpo_skipped': args.skip_hpo,
            'seeds_used': seeds,
            'main_seed': args.seed,
            'metrics': {
                'f1_mean': float(np.mean(f1_arr)),
                'f1_std': float(np.std(f1_arr)),
                'accuracy_mean': float(np.mean(acc_arr)),
                'accuracy_std': float(np.std(acc_arr)),
                'precision_mean': float(np.mean(precision_arr)),
                'precision_std': float(np.std(precision_arr)),
                'recall_mean': float(np.mean(recall_arr)),
                'recall_std': float(np.std(recall_arr)),
            },
            'individual_results': {
                'f1_scores': [float(x) for x in f1_arr],
                'accuracies': [float(x) for x in acc_arr],
                'precisions': [float(x) for x in precision_arr],
                'recalls': [float(x) for x in recall_arr],
            }
        }

        # Save individual results for this embedding
        individual_result_file = os.path.join(output_dir, f"{emb_name}_results.json")
        with open(individual_result_file, 'w') as f_out:
            json.dump(all_results[emb_name], f_out, indent=2)
        print(f"Saved individual results to: {individual_result_file}")


    # Save combined results
    combined_result_file = os.path.join(output_dir, f"{emb_name}_rf_hpo_results_{timestamp}.json")
    with open(combined_result_file, 'w') as f_out:
        json.dump(all_results, f_out, indent=2)
    print(f"Saved combined results to: {combined_result_file}")

    # Create summary table
    summary_file = os.path.join(output_dir, f"{emb_name}_rf_hpo_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f_out:
        f_out.write("Random Forest Results\n")
        f_out.write("=" * 60 + "\n\n")
        
        for emb_name, results in all_results.items():
            f_out.write(f"Embedding: {emb_name}\n")
            f_out.write(f"Path: {results['embedding_path']}\n")
            if results['hpo_skipped']:
                f_out.write("HPO: Skipped (used provided/default parameters)\n")
            else:
                f_out.write(f"Best CV Score: {results['best_cv_score']:.4f}\n")
            f_out.write(f"Best Params: {results['best_params']}\n")
            f_out.write(f"Test Metrics (mean ± std):\n")
            f_out.write(f"  F1:        {results['metrics']['f1_mean']:.4f} ± {results['metrics']['f1_std']:.4f}\n")
            f_out.write(f"  Accuracy:  {results['metrics']['accuracy_mean']:.4f} ± {results['metrics']['accuracy_std']:.4f}\n")
            f_out.write(f"  Precision: {results['metrics']['precision_mean']:.4f} ± {results['metrics']['precision_std']:.4f}\n")
            f_out.write(f"  Recall:    {results['metrics']['recall_mean']:.4f} ± {results['metrics']['recall_std']:.4f}\n")
            f_out.write("-" * 40 + "\n\n")
    
    print(f"Saved summary to: {summary_file}")



        # print(X_train.shape)




if __name__=='__main__':
    main()