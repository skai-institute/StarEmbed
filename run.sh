conda activate chronos
export HF_HOME=/home/magics/hdd/sky_ws/huggingface_ws
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /home/magics/hdd/sky_ws/skai_universal_forecaster

########################################################
# run-2 100 stars
python -m src.tasks.preproc \
  --input-file ./outputs/download/run-2/lightcurves.csv \
  --output-dir ./outputs/processed/run-2 \
  --stars-per-class 100 \
  --class-ids 1 5 \
  --band g \
  --plot

python -m src.tasks.train.chronos.generate_data \
  --input-files ./outputs/processed/run-2/processed_lightcurves.csv \
  --output-file ./outputs/train/run-2/data/data.arrow

# train
python -m src.tasks.train.chronos.train --config src/tasks/train/chronos/configs/run-2.yml

# eval
python -m src.tasks.eval.forecast.chronos \
  --model-path /home/magics/hdd/sky_ws/skai_universal_forecaster/outputs/train/run-2/output/overfit/small-crds1/run-0/checkpoint-20000 \
  --data-path ./outputs/processed/run-2/processed_lightcurves.csv \
  --output-dir ./outputs/eval/run-2/forecast

python -m src.tasks.eval.classification.kmeans \
  --model-path /home/magics/hdd/sky_ws/skai_universal_forecaster/outputs/train/run-2/output/overfit/small-crds1/run-0/checkpoint-20000 \
  --data-path ./outputs/processed/run-2/processed_lightcurves.csv \
  --output-dir ./outputs/eval/run-2/kmeans

########################################################
# run-3 4 stars
python -m src.tasks.preproc \
  --input-file ./outputs/download/run-2/lightcurves.csv \
  --output-dir ./outputs/processed/run-3 \
  --stars-per-class 4 \
  --class-ids 1 5 \
  --band g \
  --plot

python -m src.tasks.train.chronos.generate_data \
  --input-files ./outputs/processed/run-3/processed_lightcurves.csv \
  --output-file ./outputs/train/run-3/data/data.arrow

# train
python -m src.tasks.train.chronos.train --config src/tasks/train/chronos/configs/run-3.yml

# eval
python -m src.tasks.eval.forecast.chronos \
  --model-path /home/magics/hdd/sky_ws/skai_universal_forecaster/outputs/train/run-3/output/overfit/small-crds1/run-0/checkpoint-20000 \
  --data-path ./outputs/processed/run-3/processed_lightcurves.csv \
  --output-dir ./outputs/eval/run-3/forecast

python -m src.tasks.eval.classification.kmeans \
  --model-path /home/magics/hdd/sky_ws/skai_universal_forecaster/outputs/train/run-3/output/overfit/small-crds1/run-0/checkpoint-20000 \
  --data-path ./outputs/processed/run-3/processed_lightcurves.csv \
  --output-dir ./outputs/eval/run-3/kmeans

########################################################
# run-4 100 stars filtering bad flag 

python -m src.tasks.preproc \
  --input-file ./outputs/download/run-2/lightcurves.csv \
  --output-dir ./outputs/processed/run-4 \
  --stars-per-class 100 \
  --class-ids 1 5 \
  --band g \
  --plot


