#!/bin/bash
#SBATCH --account=p32626
#SBATCH --job-name=1run_handcraft_st_gr
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/src/output/log/classification/mlp/%x_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/src/output/log/classification/mlp/%x_%A_%a.err
#SBATCH --time=01:00:00      
#SBATCH --partition=gengpu   
#SBATCH --gres=gpu:h100:1      
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --open-mode=append
#SBATCH --mem=40G
#SBATCH --mail-type=ALL ## you can receive e-mail alerts from SLURM when your job begins and when your job finishes (completed, failed, etc)
#SBATCH --mail-user=b0976960890@gmail.com ## your email

module purge all
module load gcc/11.2.0
eval "$(conda shell.bash hook)"
conda activate ag





# # handcrafted feature normalization first parameters
INPUTS="/projects/b1094/StarEmbed/embeddings/descriptive_name_embeddings/csdr1_raw4_catflags_filtered_embs_hand_crafted_trn_val_tst_bandgr"
for BATCH_SIZE in 32; do
    for LR in 1e-4; do
        for DROPOUT in 0.0; do
            for LAYERS in 3; do
                for see in 200; do
                    CUDA_VISIBLE_DEVICES=0 srun python /projects/b1094/StarEmbed/skai_universal_forecaster/src_clean/benchmark/classification/mlp_pl2_wloss_standardization.py --batch_size $BATCH_SIZE --lr $LR --dropout $DROPOUT --hidden_layers $LAYERS --out_dir "/projects/p32795/weijian/skai_universal_forecaster/src/aurora/eval/classification/mlp/new_avg" --epochs 50 --input_embs $INPUTS --scenario concat --hand_crafted 1 --seed $see
                    # echo "testing $path" # testing
                done
            done
        done
    done
done






