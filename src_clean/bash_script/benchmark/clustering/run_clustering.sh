





CLUSTERING_SCRIPT="/projects/b1094/StarEmbed/src/benchmark/clustering/clustering.py"
OUTPUT_DIR="/projects/b1094/StarEmbed/src/benchmark/clustering/clustering"

for SEED in 42 100 200; do
    for CONCAT_EMBS in 1; do
        # python $CLUSTERING_SCRIPT \
        #     --dataset-dir "/projects/b1094/StarEmbed/embeddings/embeddings_with_anom/csdr1_raw4_catflags_filtered_embs_chronos_bolt_tiny_trn_val_tst_ctx200_bandgr" \
        #     --save-dir "$OUTPUT_DIR" \
        #     --mode test \
        #     --perplexity 30 \
        #     --random-state $SEED \
        #     --concat-embs $CONCAT_EMBS &

        # python $CLUSTERING_SCRIPT \
        #     --dataset-dir "/projects/p32795/weijian/embs/csdr1_raw4_catflags_filtered_embs_chronos_bolt_tiny_trn_val_tst_ctx200_bandgr" \
        #     --save-dir "$OUTPUT_DIR" \
        #     --mode test \
        #     --perplexity 30 \
        #     --random-state $SEED \
        #     --concat-embs $CONCAT_EMBS &

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

        # python $CLUSTERING_SCRIPT \
        #     --dataset-dir "/projects/p32795/weijian/embs/csdr1_raw_embs_moiral_small_trn_val_tst_ctx200_pdt64_psz16_bandgr" \
        #     --save-dir "$OUTPUT_DIR" \
        #     --mode test \
        #     --perplexity 30 \
        #     --random-state $SEED \
        #     --concat-embs $CONCAT_EMBS &

        # python $CLUSTERING_SCRIPT \
        #     --dataset-dir "/projects/p32795/weijian/embs/csdr1_raw_embs_moiral_trn_val_tst_ctx200_pdt64_psz16_bandgr" \
        #     --save-dir "$OUTPUT_DIR" \
        #     --mode test \
        #     --perplexity 30 \
        #     --random-state $SEED \
        #     --concat-embs $CONCAT_EMBS &

        # python $CLUSTERING_SCRIPT \
        #     --dataset-dir "/projects/p32795/weijian/embs/csdr1_raw4_catflags_filtered_embs_hand_crafted_trn_val_tst_bandgr" \
        #     --mode test \
        #     --perplexity 30 \
        #     --random-state $SEED \
        #     --concat-embs $CONCAT_EMBS \
        #     --standardize 1 \
        #     --hand_crafted 1 &

        # python $CLUSTERING_SCRIPT \
        #     --dataset-dir "/projects/p32795/dennis/random" \
        #     --save-dir "$OUTPUT_DIR" \
        #     --mode test \
        #     --perplexity 30 \
        #     --random-state $SEED \
        #     --concat-embs $CONCAT_EMBS \
        #     --hand-crafted 1 &

        # python $CLUSTERING_SCRIPT \
        #     --dataset-dir "/projects/b1094/StarEmbed/embeddings/embeddings_with_anom/hf_csdr1_multiband_raw4_embeddings_astromer_2_gr_sampling_True" \
        #     --mode test \
        #     --perplexity 30 \
        #     --random-state $SEED \
        #     --concat-embs $CONCAT_EMBS &

        # wait
    done
done


# python clustering_tmp.py --dataset-dir "/projects/p32795/weijian/embs/csdr1_raw4_catflags_filtered_embs_chronos_bolt_trn_val_tst_ctx200_bandgr" --save-dir "/projects/p32795/weijian/skai_universal_forecaster/outputs/clustering_tmp" --mode test --perplexity 30 --random-state 42 --concat-embs 1


# python "/projects/p32795/weijian/skai_universal_forecaster/src/aurora/eval/classification/kmeans/clustering.py" --dataset-dir "/projects/p32795/weijian/embs/csdr1_raw4_catflags_filtered_embs_hand_crafted_trn_val_tst_bandgr" --save-dir "/projects/p32795/weijian/skai_universal_forecaster/src/aurora/eval/classification/kmeans/clustering4" --mode test --perplexity 30 --random-state 42 --concat-embs 0