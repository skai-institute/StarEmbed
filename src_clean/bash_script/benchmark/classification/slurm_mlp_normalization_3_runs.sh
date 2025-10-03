#!/bin/bash
#SBATCH --account=xxx
#SBATCH --job-name=3run_handcraft_st_gr
#SBATCH --nodes=1
#SBATCH --output=error_log/%x_%A_%a.out
#SBATCH --error=error_log/%x_%A_%a.err
#SBATCH --time=01:00:00      
#SBATCH --partition=gengpu   
#SBATCH --gres=gpu:h100:1      
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --open-mode=append
#SBATCH --mem=60G
#SBATCH --mail-type=ALL ## you can receive e-mail alerts from SLURM when your job begins and when your job finishes (completed, failed, etc)
#SBATCH --mail-user=xxx ## your email

module purge all
module load gcc/11.2.0
eval "$(conda shell.bash hook)"
conda activate ag
cd /xxx/mlp_handcrafted_feature_results




# # handcrafted feature normalization first parameters
INPUTS="/xxx/csdr1_raw4_catflags_filtered_embs_hand_crafted_trn_val_tst_bandgr/"
for BATCH_SIZE in 32; do
    for LR in 1e-4; do
        for DROPOUT in 0.0; do
            for LAYERS in 3; do
                for see in 100 200; do
                    CUDA_VISIBLE_DEVICES=0 srun python /xxx/mlp_pl2_wloss_standardization.py --batch_size $BATCH_SIZE --lr $LR --dropout $DROPOUT --hidden_layers $LAYERS --out_dir "/projects/p32795/weijian/skai_universal_forecaster/src/aurora/eval/classification/mlp" --epochs 50 --input_embs $INPUTS --scenario concat --hand_crafted 1 --seed $see
                    # echo "testing $path" # testing
                done
            done
        done
    done
done






