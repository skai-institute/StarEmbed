#!/bin/bash
#SBATCH --account=xxx
#SBATCH --job-name=handcraft_gr_hp_grid
#SBATCH --nodes=1
#SBATCH --output=error_log/%x_%A_%a.out
#SBATCH --error=error_log/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --array=0-23      
#SBATCH --partition=gengpu   
#SBATCH --gres=gpu:h100:1      
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --open-mode=append
#SBATCH --mem=60G
#SBATCH --mail-type=ALL ## you can receive e-mail alerts from SLURM when your job begins and when your job finishes (completed, failed, etc)
#SBATCH --mail-user=xxx ## your email

# uni2ts/data/projects
module purge
module load gcc/11.2.0
eval "$(conda shell.bash hook)"
conda activate ag
cd /xxx/mlp_handcrafted_feature_results

# parameter grid
batch_sizes=(32 64 128 256)
lrs=(1e-2 1e-3 1e-4)
dropouts=(0.0 0.1)
layers=3
seed=42
INPUTS="/xxx/csdr1_raw4_catflags_filtered_embs_hand_crafted_trn_val_tst_bandgr/"

# pick out exactly one combo
idx=0
for bs in "${batch_sizes[@]}"; do
  for lr in "${lrs[@]}"; do
    for do in "${dropouts[@]}"; do
      if [ "$idx" -eq "$SLURM_ARRAY_TASK_ID" ]; then
        echo "▶ TASK $SLURM_ARRAY_TASK_ID → BS=$bs, LR=$lr, DO=$do"
        srun python /xxx/mlp_pl2_wloss_standardization.py \
          --batch_size  "$bs" \
          --lr          "$lr" \
          --dropout     "$do" \
          --hidden_layers "$layers" \
          --seed        "$seed" \
          --input_embs  "$INPUTS" \
          --out_dir     "/xxx/classification/mlp" \
          --epochs      50 \
          --scenario    concat \
          --hand_crafted 1
        exit 0
      fi
      idx=$((idx+1))
    done
  done
done