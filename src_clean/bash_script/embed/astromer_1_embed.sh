

# to the directory contain weights/macho
cd /projects/b1094/StarEmbed/src/model/astromer_1

python /projects/b1094/StarEmbed/src/model/astromer_1/embed.py \
    --input_path /projects/p32795/hongyu/hf_macho_70-10-20 \
    --output_path /projects/b1094/StarEmbed/embeddings/hf_macho_unlabel_embeddings_astromer_1 \
    --model_name macho \
    --bands g r \
    --splits validation \
    --duration 200 \
    --enc_batch 1024 \
    --preproc_procs 8 \  # number of worker for preprocessing the light curve into windows

