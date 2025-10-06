





CLUSTERING_SCRIPT="/projects/b1094/StarEmbed/skai_universal_forecaster/src_clean/benchmark/clustering/clustering.py"
# OUTPUT_DIR="/projects/p32795/weijian/skai_universal_forecaster/src/aurora/eval/classification/kmeans/clustering4"

for SEED in 42; do
    for CONCAT_EMBS in 1; do
        python $CLUSTERING_SCRIPT \
            --dataset-dir "/projects/b1094/StarEmbed/embeddings/descriptive_name_embeddings/csdr1_raw4_catflags_filtered_embs_chronos_t5_tiny_trn_val_tst_ctx200_bandgr" \
            --mode all \
            --perplexity 30 \
            --random-state $SEED \
            --scenario concat \
            --standardize 1 \
            --output-dir "/projects/b1094/StarEmbed/src/output/clustering/all_split_all_standardize/test_new_avg" \
            --clustering-method hierarchical \
            # --save-dendrogram

        # python $CLUSTERING_SCRIPT \
        #     --dataset-dir "/projects/b1094/StarEmbed/embeddings/embeddings_with_anom/csdr1_raw4_catflags_filtered_embs_chronos_bolt_tiny_trn_val_tst_ctx200_bandgr" \
        #     --mode all \
        #     --perplexity 30 \
        #     --random-state $SEED \
        #     --concat-embs $CONCAT_EMBS \
        #     --standardize 1 \
        #     --output-dir "/projects/b1094/StarEmbed/src/output/clustering/all_split_all_standardize" \
        #     --clustering-method hierarchical \
        #     --save-dendrogram


        # python $CLUSTERING_SCRIPT \
        #     --dataset-dir "/projects/b1094/StarEmbed/embeddings/embeddings_with_anom/csdr1_raw_embs_moiral_small_trn_val_tst_ctx200_pdt64_psz16_bandgr" \
        #     --mode all \
        #     --perplexity 30 \
        #     --random-state $SEED \
        #     --concat-embs $CONCAT_EMBS \
        #     --standardize 1 \
        #     --output-dir "/projects/b1094/StarEmbed/src/output/clustering/all_split_all_standardize" \
        #     --clustering-method hierarchical \
        #     --save-dendrogram

        # python $CLUSTERING_SCRIPT \
        #     --dataset-dir "/projects/b1094/StarEmbed/embeddings/embeddings_with_anom/hf_csdr1_multiband_raw4_embeddings_astromer_1_subclass_pad_correct" \
        #     --mode all \
        #     --perplexity 30 \
        #     --random-state $SEED \
        #     --concat-embs $CONCAT_EMBS \
        #     --standardize 1 \
        #     --output-dir "/projects/b1094/StarEmbed/src/output/clustering/all_split_all_standardize" \
        #     --clustering-method hierarchical \
        #     --save-dendrogram


        # # python $CLUSTERING_SCRIPT \
        # #     --dataset-dir "/projects/b1094/StarEmbed/embeddings/csdr1_raw4_catflags_filtered_embs_hand_crafted_trn_val_tst_bandgr" \
        # #     --mode all \
        # #     --perplexity 30 \
        # #     --random-state $SEED \
        # #     --concat-embs $CONCAT_EMBS \
        # #     --standardize 1 \
        # #     --hand-crafted 1

        # python $CLUSTERING_SCRIPT \
        #     --dataset-dir "/projects/p32795/dennis/random" \
        #     --mode all \
        #     --perplexity 30 \
        #     --random-state $SEED \
        #     --concat-embs $CONCAT_EMBS \
        #     --hand-crafted 1 \
        #     --standardize 1 \
        #     --output-dir "/projects/b1094/StarEmbed/src/output/clustering/all_split_all_standardize" \
        #     --clustering-method hierarchical \
        #     --save-dendrogram



        # python $CLUSTERING_SCRIPT \
        #     --dataset-dir "/projects/b1094/StarEmbed/embeddings/embeddings_with_anom/hf_csdr1_multiband_raw4_embeddings_astromer_2_gr_sampling_True" \
        #     --mode all \
        #     --perplexity 30 \
        #     --random-state $SEED \
        #     --concat-embs $CONCAT_EMBS \
        #     --standardize 1 \
        #     --output-dir "/projects/b1094/StarEmbed/src/output/clustering/all_split_all_standardize" \
        #     --clustering-method hierarchical \
        #     --save-dendrogram

        # wait
    done
done