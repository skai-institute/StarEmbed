#!/usr/bin/env python3
"""
Example usage of the updated rf_hpo.py script with new functionality.
The script uses multiple seeds based on the main seed for statistical significance.
"""

# Example 1: Run with hyperparameter optimization (default behavior)
# python rf_hpo.py --input-embs /path/to/embeddings --standardize 1 --seed 42

# Example 2: Skip HPO and use provided best parameters
# python rf_hpo.py --input-embs /path/to/embeddings --skip-hpo --best-params '{"n_estimators": 200, "max_depth": 20, "min_samples_split": 5}' --seed 42

# Example 3: Skip HPO and use default parameters
# python rf_hpo.py --input-embs /path/to/embeddings --skip-hpo --seed 42

# Example 4: Run with hand-crafted features and skip HPO
# python rf_hpo.py --input-embs /path/to/embeddings --hand-crafted 1 --skip-hpo --seed 42

# The script automatically generates 3 seeds based on the main seed:
# [seed, seed+58, seed+158] for statistical significance

# The script will generate:
# 1. Classification report (CSV format) - using main seed model
# 2. Confusion matrix (both pickle and PDF plot) - using main seed model
# 3. Results JSON with mean ± std metrics across all seeds
# 4. Summary text file

print("See the comments in this file for usage examples of rf_hpo.py")
