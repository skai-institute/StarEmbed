#!/bin/bash
#SBATCH --account=p32795
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=16:00:00
#SBATCH --job-name=astromer_2_embed
#SBATCH --output=logs/astromer_2_embed_%j.out
#SBATCH --error=logs/astromer_2_embed_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

echo "Starting ASTROMER-2 embedding generation..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"

# Activate conda environment for ASTROMER-2 (TensorFlow)
source /projects/p32795/miniforge3/etc/profile.d/conda.sh
conda activate astromer_tf

# Set default parameters (can be overridden by command line arguments)
INPUT_PATH=${INPUT_PATH:-/projects/p32795/hongyu/hf_csdr1_multiband_raw_lc_minority_class_str_v2}
OUTPUT_PATH=${OUTPUT_PATH:-/projects/p32795/hongyu/csdr1_raw_embs_astromer_2}
MODEL_WEIGHTS=${MODEL_WEIGHTS:-/projects/p32626/uni2ts/data/weights/macho-clean}
BANDS=${BANDS:-"g r"}
SPLITS=${SPLITS:-"validation"}
DURATION=${DURATION:-200}
ENC_BATCH=${ENC_BATCH:-1024}
PREPROC_PROCS=${PREPROC_PROCS:-16}
FILTER_GR=${FILTER_GR:-"--filter_gr"}

echo "Parameters:"
echo "  Input path: $INPUT_PATH"
echo "  Output path: $OUTPUT_PATH"
echo "  Model weights: $MODEL_WEIGHTS"
echo "  Bands: $BANDS"
echo "  Splits: $SPLITS"
echo "  Duration: $DURATION"
echo "  Encoding batch size: $ENC_BATCH"
echo "  Preprocessing processes: $PREPROC_PROCS"
echo "  Filter GR: $FILTER_GR"
echo ""

# Change to the required working directory for ASTROMER-2 (main-code repo)
cd /projects/p32626/uni2ts/data/main-code

# Run the embedding generation script
python /projects/b1094/StarEmbed/src/model/astromer_2/embed.py \
    --input_path $INPUT_PATH \
    --output_path $OUTPUT_PATH \
    --model_weights $MODEL_WEIGHTS \
    --bands $BANDS \
    --splits $SPLITS \
    --duration $DURATION \
    --enc_batch $ENC_BATCH \
    --preproc_procs $PREPROC_PROCS \
    $FILTER_GR

echo "ASTROMER-2 embedding generation completed at $(date)"
echo "Exit code: $?"
