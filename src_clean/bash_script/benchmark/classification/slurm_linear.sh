#!/bin/bash
#SBATCH --account=p32626
#SBATCH --job-name=linear_classification
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/src/output/log/classification/%x_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/src/output/log/classification/%x_%A_%a.err
#SBATCH --time=1:00:00      
#SBATCH --partition=gengpu   
#SBATCH --gres=gpu:h100:1      
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --open-mode=append
#SBATCH --mem=60G
#SBATCH --mail-type=ALL ## you can receive e-mail alerts from SLURM when your job begins and when your job finishes (completed, failed, etc)
#SBATCH --mail-user=b0976960890@gmail.com ## your email


module purge all
module load gcc/11.2.0
eval "$(conda shell.bash hook)"
conda activate ag
cd /projects/b1094/StarEmbed/skai_universal_forecaster/src_clean


for seed in 42; do
    # python linear_knn.py --input_embs /projects/b1094/StarEmbed/embeddings/embeddings_with_anom/csdr1_raw4_catflags_filtered_embs_chronos_bolt_tiny_trn_val_tst_ctx200_bandgr --scenario concat --seed $seed
    # python linear_knn.py --input_embs /projects/b1094/StarEmbed/embeddings/embeddings_with_anom/csdr1_raw_embs_moiral_small_trn_val_tst_ctx200_pdt64_psz16_bandgr --scenario concat --seed $seed
    # python linear_knn.py --input_embs /projects/b1094/StarEmbed/embeddings/embeddings_with_anom/hf_csdr1_multiband_raw4_embeddings_astromer_1_subclass_pad_correct --scenario concat --seed $seed
    # python linear_knn.py --input_embs /projects/b1094/StarEmbed/embeddings/embeddings_with_anom/hf_csdr1_multiband_raw4_embeddings_astromer_2_gr_sampling_True --scenario concat --seed $seed
    python benchmark/classification/linear_knn.py --input_embs /projects/b1094/StarEmbed/embeddings/descriptive_name_embeddings/csdr1_raw4_catflags_filtered_embs_hand_crafted_trn_val_tst_bandgr  --hand_crafted True --seed $seed
    # python linear_knn.py --input_embs /projects/p32795/dentnis/random --scenario concat --seed $seed
done

# for seed in 100 200; do
#     python linear_knn.py --input_embs /projects/b1094/StarEmbed/embeddings/embeddings_with_anom/csdr1_raw4_catflags_filtered_embs_chronos_t5_tiny_trn_val_tst_ctx200_bandgr --scenario concat --seed $seed
# done
