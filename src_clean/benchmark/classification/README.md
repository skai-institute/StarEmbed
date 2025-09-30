```
python -m pip install requirements.txt
```
src/bash_script
# Random Forest

## Usage

```bash
# With hyperparameter optimization
python rf_hpo.py --input-embs /path/to/embeddings --seed 42

# Skip HPO with custom parameters
python rf_hpo.py --input-embs /path/to/embeddings --skip-hpo --best-params '{"n_estimators": 200, "max_depth": 20}' --seed 42

# Skip HPO with default parameters
python rf_hpo.py --input-embs /path/to/embeddings --skip-hpo --seed 42
```

## Options

- `--standardize 1`: Apply feature standardization
- `--hand-crafted 1`: Use hand-crafted features
- `--skip-hpo`: Skip hyperparameter optimization
- `--best-params`: JSON string with parameters
- `--seed`: Random seed (generates 3 seeds for statistics)
- `--output-dir`: Custom output directory (default: `/projects/b1094/StarEmbed/src/output/rf`)

## Output

- `{emb_name}_rf_classification_report.csv`: Classification metrics per class
- `{emb_name}_rf_confusion_matrix.pkl/.pdf`: Confusion matrix data and plot
- `{emb_name}_results.json`: Full results with mean ± std metrics
- `{emb_name}_rf_hpo_summary_{timestamp}.txt`: Summary report

