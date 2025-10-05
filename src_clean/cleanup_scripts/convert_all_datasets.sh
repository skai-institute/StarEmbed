#!/bin/bash

# Simple script to convert datasets from numeric class IDs to descriptive names

CONVERSION_SCRIPT="/projects/b1094/StarEmbed/skai_universal_forecaster/src_clean/cleanup_scripts/convert_class_names.py"
OUTPUT_BASE_DIR="/projects/b1094/StarEmbed/embeddings/descriptive_name_embeddings"

# List of input paths to convert
INPUT_PATHS=(
    "/projects/b1094/StarEmbed/embeddings/embeddings_with_anom/csdr1_raw4_catflags_filtered_embs_chronos_bolt_tiny_trn_val_tst_ctx200_bandgr"
    "/projects/b1094/StarEmbed/embeddings/embeddings_with_anom/csdr1_raw4_catflags_filtered_embs_chronos_t5_tiny_trn_val_tst_ctx200_bandgr"
    "/projects/b1094/StarEmbed/embeddings/embeddings_with_anom/csdr1_raw_embs_moiral_small_trn_val_tst_ctx200_pdt64_psz16_bandgr"
    "/projects/b1094/StarEmbed/embeddings/embeddings_with_anom/hf_csdr1_multiband_raw4_embeddings_astromer_1_subclass_pad_correct"
    # "/projects/b1094/StarEmbed/embeddings/embeddings_with_anom/hf_csdr1_multiband_raw4_embeddings_astromer_2_gr_sampling_True"
    # "/projects/b1094/StarEmbed/embeddings/csdr1_raw4_catflags_filtered_embs_hand_crafted_trn_val_tst_bandgr"
    # "/projects/p32795/dennis/random"
    # "/projects/p32795/hongyu/hf_csdr1_multiband_raw_lc_subclass_class_str_v2"
)

# Create output directory
# mkdir -p "$OUTPUT_BASE_DIR"

# Loop through each input path and convert
for input_path in "${INPUT_PATHS[@]}"; do
    dataset_name=$(basename "$input_path")
    output_path="$OUTPUT_BASE_DIR/$dataset_name"
    
    echo "Converting: $dataset_name"
    echo "  Input:  $input_path"
    echo "  Output: $output_path"
    
    python "$CONVERSION_SCRIPT" \
        --input-path "$input_path" \
        --output-path "$output_path" \
        --num-proc 4
    
    echo "---"
done

echo "All conversions complete!"
echo "Converted datasets saved to: $OUTPUT_BASE_DIR"
