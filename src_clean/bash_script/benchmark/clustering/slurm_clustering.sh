#!/bin/bash
#SBATCH --account=p32626
#SBATCH --job-name=clustering
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/src/output/log/%x_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/src/output/log/%x_%A_%a.err
#SBATCH --time=01:00:00      
#SBATCH --partition=gengpu   
#SBATCH --gres=gpu:h100:1      
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=append
#SBATCH --mem=40G
#SBATCH --mail-type=ALL ## you can receive e-mail alerts from SLURM when your job begins and when your job finishes (completed, failed, etc)
#SBATCH --mail-user=b0976960890@gmail.com ## your email

module purge all
module load gcc/11.2.0
eval "$(conda shell.bash hook)"
conda activate ag
cd /projects/b1094/StarEmbed/src/benchmark/clustering



CLUSTERING_SCRIPT="/projects/b1094/StarEmbed/src/benchmark/clustering/clustering.py"
# OUTPUT_DIR="/projects/p32795/weijian/skai_universal_forecaster/src/aurora/eval/classification/kmeans/clustering4"

for SEED in 42; do
    for CONCAT_EMBS in 1; do
        python $CLUSTERING_SCRIPT \
            --dataset-dir "/projects/b1094/StarEmbed/embeddings/csdr1_raw4_catflags_filtered_embs_chronos_t5_tiny_trn_val_tst_ctx200_bandgr" \
            --mode all \
            --perplexity 30 \
            --random-state $SEED \
            --concat-embs $CONCAT_EMBS \
            --standardize 1 \
            --output-dir "/projects/b1094/StarEmbed/src/output/clustering/all_split_all_standardize" \
            --clustering-method hierarchical \
            --save-dendrogram

        python $CLUSTERING_SCRIPT \
            --dataset-dir "/projects/b1094/StarEmbed/embeddings/embeddings_with_anom/csdr1_raw4_catflags_filtered_embs_chronos_bolt_tiny_trn_val_tst_ctx200_bandgr" \
            --mode all \
            --perplexity 30 \
            --random-state $SEED \
            --concat-embs $CONCAT_EMBS \
            --standardize 1 \
            --output-dir "/projects/b1094/StarEmbed/src/output/clustering/all_split_all_standardize" \
            --clustering-method hierarchical \
            --save-dendrogram


        # python $CLUSTERING_SCRIPT \
        #     --dataset-dir "/projects/p32795/weijian/embs/csdr1_raw4_catflags_filtered_embs_chronos_bolt_trn_val_tst_ctx200_bandgr" \
        #     --save-dir "$OUTPUT_DIR" \
        #     --mode test \
        #     --perplexity 30 \
        #     --random-state $SEED \
        #     --concat-embs $CONCAT_EMBS &

        # python $CLUSTERING_SCRIPT \
        #     --dataset-dir "/projects/p32795/weijian/embs/csdr1_raw4_catflags_filtered_embs_chronos_t5_trn_val_tst_ctx200_bandgr" \
        #     --save-dir "$OUTPUT_DIR" \
        #     --mode test \
        #     --perplexity 30 \
        #     --random-state $SEED \
        #     --concat-embs $CONCAT_EMBS &

        python $CLUSTERING_SCRIPT \
            --dataset-dir "/projects/b1094/StarEmbed/embeddings/embeddings_with_anom/csdr1_raw_embs_moiral_small_trn_val_tst_ctx200_pdt64_psz16_bandgr" \
            --mode all \
            --perplexity 30 \
            --random-state $SEED \
            --concat-embs $CONCAT_EMBS \
            --standardize 1 \
            --output-dir "/projects/b1094/StarEmbed/src/output/clustering/all_split_all_standardize" \
            --clustering-method hierarchical \
            --save-dendrogram




        python $CLUSTERING_SCRIPT \
            --dataset-dir "/projects/b1094/StarEmbed/embeddings/embeddings_with_anom/hf_csdr1_multiband_raw4_embeddings_astromer_1_subclass_pad_correct" \
            --mode all \
            --perplexity 30 \
            --random-state $SEED \
            --concat-embs $CONCAT_EMBS \
            --standardize 1 \
            --output-dir "/projects/b1094/StarEmbed/src/output/clustering/all_split_all_standardize" \
            --clustering-method hierarchical \
            --save-dendrogram


        # python $CLUSTERING_SCRIPT \
        #     --dataset-dir "/projects/p32795/weijian/embs/csdr1_raw_embs_moiral_trn_val_tst_ctx200_pdt64_psz16_bandgr" \
        #     --save-dir "$OUTPUT_DIR" \
        #     --mode test \
        #     --perplexity 30 \
        #     --random-state $SEED \
        #     --concat-embs $CONCAT_EMBS &

        # python $CLUSTERING_SCRIPT \
        #     --dataset-dir "/projects/b1094/StarEmbed/embeddings/csdr1_raw4_catflags_filtered_embs_hand_crafted_trn_val_tst_bandgr" \
        #     --mode all \
        #     --perplexity 30 \
        #     --random-state $SEED \
        #     --concat-embs $CONCAT_EMBS \
        #     --standardize 1 \
        #     --hand-crafted 1

        python $CLUSTERING_SCRIPT \
            --dataset-dir "/projects/p32795/dennis/random" \
            --mode all \
            --perplexity 30 \
            --random-state $SEED \
            --concat-embs $CONCAT_EMBS \
            --hand-crafted 1 \
            --standardize 1 \
            --output-dir "/projects/b1094/StarEmbed/src/output/clustering/all_split_all_standardize" \
            --clustering-method hierarchical \
            --save-dendrogram



        python $CLUSTERING_SCRIPT \
            --dataset-dir "/projects/b1094/StarEmbed/embeddings/embeddings_with_anom/hf_csdr1_multiband_raw4_embeddings_astromer_2_gr_sampling_True" \
            --mode all \
            --perplexity 30 \
            --random-state $SEED \
            --concat-embs $CONCAT_EMBS \
            --standardize 1 \
            --output-dir "/projects/b1094/StarEmbed/src/output/clustering/all_split_all_standardize" \
            --clustering-method hierarchical \
            --save-dendrogram

        # wait
    done
done

