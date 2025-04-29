#!/bin/bash

# Activate conda environment and set environment variables
conda activate chronos
export HF_HOME=/home/magics/hdd/sky_ws/huggingface_ws
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /home/magics/hdd/sky_ws/skai_universal_forecaster

# Run identifier (change this as needed)
RUN_ID="run-7"

# Configuration
MODEL_BASE_PATH="/home/magics/hdd/sky_ws/skai_universal_forecaster/data/train"
DATA_PATH="./data/processed/$RUN_ID/processed_lightcurves.csv"
OUTPUT_BASE_DIR="./data/eval"
N_CLUSTERS=17

# Array of checkpoint numbers to evaluate
# CHECKPOINTS=(5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 "final")
CHECKPOINTS=(5000 10000 15000 20000 25000 30000 35000 40000 45000)

echo "Starting evaluation for checkpoints: ${CHECKPOINTS[@]}"
echo "Using data path: $DATA_PATH"
echo "Using run ID: $RUN_ID"

for checkpoint in "${CHECKPOINTS[@]}"; do
    echo "===================================================="
    echo "Processing checkpoint: $checkpoint"
    
    # Determine model path
    if [ "$checkpoint" == "final" ]; then
        MODEL_PATH="$MODEL_BASE_PATH/$RUN_ID/run-0/checkpoint-final"
    else
        MODEL_PATH="$MODEL_BASE_PATH/$RUN_ID/run-0/checkpoint-$checkpoint"
    fi
    
    # Set output directories
    KMEANS_OUTPUT_DIR="$OUTPUT_BASE_DIR/$RUN_ID/checkpoint-$checkpoint/kmeans"
    FORECAST_OUTPUT_DIR="$OUTPUT_BASE_DIR/$RUN_ID/checkpoint-$checkpoint/forecast"
    
    echo "Model path: $MODEL_PATH"
    echo "KMeans output directory: $KMEANS_OUTPUT_DIR"
    echo "Forecast output directory: $FORECAST_OUTPUT_DIR"
    
    # Create output directories if they don't exist
    mkdir -p "$KMEANS_OUTPUT_DIR"
    mkdir -p "$FORECAST_OUTPUT_DIR"
    
    # Run KMeans clustering
    echo "Running KMeans clustering..."
    python -m src.tasks.eval.classification.kmeans.v2 \
      --model-path "$MODEL_PATH" \
      --data-path "$DATA_PATH" \
      --output-dir "$KMEANS_OUTPUT_DIR" \
      --n-clusters "$N_CLUSTERS"
    
    # Run forecasting
    echo "Running forecasting..."
    python -m src.tasks.eval.forecast.chronos.v2 \
      --model-path "$MODEL_PATH" \
      --data-path "$DATA_PATH" \
      --output-dir "$FORECAST_OUTPUT_DIR"
    
    echo "Completed processing checkpoint: $checkpoint"
    echo "===================================================="
done

echo "All checkpoint evaluations completed!" 