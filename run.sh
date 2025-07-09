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
  --model-path /home/magics/hdd/sky_ws/skai_universal_forecaster/data/train/run-2/output/run-0/checkpoint-10000 \
  --data-path ./data/processed/run-2/processed_lightcurves.csv \
  --output-dir ./data/eval/run-2/forecast

python -m src.tasks.eval.classification.kmeans \
  --model-path /home/magics/hdd/sky_ws/skai_universal_forecaster/data/train/run-2/output/run-0/checkpoint-10000 \
  --data-path ./data/processed/run-2/processed_lightcurves.csv \
  --output-dir ./data/eval/run-2/kmeans

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
  --input-file ./data/download/run-2/lightcurves.csv \
  --output-dir ./data/processed/run-4 \
  --stars-per-class 100 \
  --class-ids 1 5 \
  --band g \
  --plot


python -m src.tasks.train.chronos.generate_data \
  --input-files ./data/processed/run-4/processed_lightcurves.csv \
  --output-file ./data/train/run-4/data/data.arrow

# train
python -m src.tasks.train.chronos.train --config src/tasks/train/chronos/configs/run-4.yml

# eval
python -m src.tasks.eval.forecast.chronos \
  --model-path /home/magics/hdd/sky_ws/skai_universal_forecaster/data/train/run-4/run-0/checkpoint-final \
  --data-path ./data/processed/run-4/processed_lightcurves.csv \
  --output-dir ./data/eval/run-4/forecast

python -m src.tasks.eval.classification.kmeans \
  --model-path /home/magics/hdd/sky_ws/skai_universal_forecaster/data/train/run-4/run-0/checkpoint-final \
  --data-path ./data/processed/run-4/processed_lightcurves.csv \
  --output-dir ./data/eval/run-4/kmeans


########################################################
# fine tune

# train
# python -m src.tasks.train.chronos.train --config src/tasks/train/chronos/configs/run-4.yml

########################################################
# run-6 100 stars, raw data, without filtering bad flag

# download
python -m src.tasks.download \
  --output-dir ./data/download/run-3 \
  --n 100

python -m src.tasks.preproc.v2 \
  --input-file ./data/download/run-2/lightcurves.csv \
  --output-dir ./data/processed/run-6 \
  --stars-per-class 100 \
  --class-ids 1 5 \
  --band g \
  --plot

python -m src.tasks.train.chronos.generate_data \
  --input-files ./data/processed/run-6/processed_lightcurves.csv \
  --output-file ./data/train/run-6/data/data.arrow

python -m src.tasks.train.chronos.train --config src/tasks/train/chronos/configs/run-6.yml


python -m src.tasks.eval.classification.kmeans.v2 \
  --model-path amazon/chronos-bolt-base \
  --data-path ./data/processed/run-6/processed_lightcurves.csv \
  --output-dir ./data/eval/run-6/chronos-bolt/kmeans

python -m src.tasks.eval.forecast.chronos.v2 \
  --model-path amazon/chronos-bolt-base \
  --data-path ./data/processed/run-6/processed_lightcurves.csv \
  --output-dir ./data/eval/run-6/chronos-bolt/forecast

python -m src.tasks.eval.classification.kmeans.v2 \
  --model-path amazon/chronos-t5-small \
  --data-path ./data/processed/run-6/processed_lightcurves.csv \
  --output-dir ./data/eval/run-6/chronos/kmeans

#********************************TODO TODO********************************
# python -m src.tasks.eval.classification.kmeans.v2 \
#   --model-path /home/magics/hdd/sky_ws/skai_universal_forecaster/data/train/run-6/run-0/checkpoint-25000 \
#   --data-path ./data/processed/run-6/processed_lightcurves.csv \
#   --output-dir ./data/eval/run-6/finetuned-chronos/kmeans

# python -m src.tasks.eval.classification.kmeans.v2 \
#   --model-path /home/magics/hdd/sky_ws/skai_universal_forecaster/data/train/run-6/run-0/checkpoint-30000 \
#   --data-path ./data/processed/run-6/processed_lightcurves.csv \
#   --output-dir ./data/eval/run-6/finetuned-chronos/kmeans

# python -m src.tasks.eval.classification.kmeans.v2 \
#   --model-path /home/magics/hdd/sky_ws/skai_universal_forecaster/data/train/run-6/run-0/checkpoint-35000 \
#   --data-path ./data/processed/run-6/processed_lightcurves.csv \
#   --output-dir ./data/eval/run-6/finetuned-chronos/kmeans

# python -m src.tasks.eval.classification.kmeans.v2 \
#   --model-path /home/magics/hdd/sky_ws/skai_universal_forecaster/data/train/run-6/run-0/checkpoint-40000 \
#   --data-path ./data/processed/run-6/processed_lightcurves.csv \
#   --output-dir ./data/eval/run-6/finetuned-chronos/kmeans

python -m src.tasks.eval.forecast.chronos.v2 \
  --model-path /home/magics/hdd/sky_ws/skai_universal_forecaster/data/train/run-6/run-0/checkpoint-final \
  --data-path ./data/processed/run-6/processed_lightcurves.csv \
  --output-dir ./data/eval/run-6/finetuned-chronos/forecast \
  --plot

# run-7 fine tune on all classes
# python -m src.tasks.preproc.v2 \
#   --input-file ./data/download/run-3/lightcurves.csv \
#   --output-dir ./data/processed/run-7 \
#   --stars-per-class 100 \
#   --band g \
#   --plot

# python -m src.tasks.train.chronos.generate_data \
#   --input-files ./data/processed/run-7/processed_lightcurves.csv \
#   --output-file ./data/train/run-7/data/data.arrow

# python -m src.tasks.train.chronos.train --config src/tasks/train/chronos/configs/run-7.yml

# python -m src.tasks.eval.classification.kmeans.v2 \
#   --model-path amazon/chronos-bolt-base \
#   --data-path ./data/processed/run-7/processed_lightcurves.csv \
#   --output-dir ./data/eval/run-7/chronos-bolt/kmeans \
#   --n-clusters 17

# python -m src.tasks.eval.classification.kmeans.v2 \
#   --model-path amazon/chronos-t5-small \
#   --data-path ./data/processed/run-7/processed_lightcurves.csv \
#   --output-dir ./data/eval/run-7/chronos-t5-small/kmeans \
#   --n-clusters 17

# python -m src.tasks.eval.forecast.chronos.v2 \
#   --model-path amazon/chronos-bolt-base \
#   --data-path ./data/processed/run-7/processed_lightcurves.csv \
#   --output-dir ./data/eval/run-7/chronos-bolt/forecast 

# python -m src.tasks.eval.classification.kmeans.v2 \
#   --model-path /home/magics/hdd/sky_ws/skai_universal_forecaster/data/train/run-7/run-0/checkpoint-5000 \
#   --data-path ./data/processed/run-7/processed_lightcurves.csv \
#   --output-dir ./data/eval/run-7/finetuned-chronos/kmeans \
#   --n-clusters 17

# python -m src.tasks.eval.classification.kmeans.v2 \
#   --model-path /home/magics/hdd/sky_ws/skai_universal_forecaster/data/train/run-7/run-0/checkpoint-10000 \
#   --data-path ./data/processed/run-7/processed_lightcurves.csv \
#   --output-dir ./data/eval/run-7/finetuned-chronos/kmeans \
#   --n-clusters 17

# python -m src.tasks.eval.classification.kmeans.v2 \
#   --model-path /home/magics/hdd/sky_ws/skai_universal_forecaster/data/train/run-7/run-0/checkpoint-15000 \
#   --data-path ./data/processed/run-7/processed_lightcurves.csv \
#   --output-dir ./data/eval/run-7/finetuned-chronos/kmeans \
#   --n-clusters 17

# python -m src.tasks.eval.classification.kmeans.v2 \
#   --model-path /home/magics/hdd/sky_ws/skai_universal_forecaster/data/train/run-7/run-0/checkpoint-20000 \
#   --data-path ./data/processed/run-7/processed_lightcurves.csv \
#   --output-dir ./data/eval/run-7/finetuned-chronos/kmeans \
#   --n-clusters 17

# python -m src.tasks.eval.classification.kmeans.v2 \
#   --model-path /home/magics/hdd/sky_ws/skai_universal_forecaster/data/train/run-7/run-0/checkpoint-25000 \
#   --data-path ./data/processed/run-7/processed_lightcurves.csv \
#   --output-dir ./data/eval/run-7/finetuned-chronos/kmeans \
#   --n-clusters 17

# python -m src.tasks.eval.classification.kmeans.v2 \
#   --model-path /home/magics/hdd/sky_ws/skai_universal_forecaster/data/train/run-7/run-0/checkpoint-30000 \
#   --data-path ./data/processed/run-7/processed_lightcurves.csv \
#   --output-dir ./data/eval/run-7/finetuned-chronos/kmeans \
#   --n-clusters 17

python -m src.tasks.eval.classification.kmeans.v2 \
  --model-path /home/magics/hdd/sky_ws/skai_universal_forecaster/data/train/run-7/run-0/checkpoint-35000 \
  --data-path ./data/processed/run-7/processed_lightcurves.csv \
  --output-dir ./data/eval/run-7/finetuned-chronos/kmeans \
  --n-clusters 17


python -m src.tasks.eval.forecast.chronos.v2 \
  --model-path /home/magics/hdd/sky_ws/skai_universal_forecaster/data/train/run-7/run-0/checkpoint-final \
  --data-path ./data/processed/run-7/processed_lightcurves.csv \
  --output-dir ./data/eval/run-7/finetuned-chronos/forecast



# use base model

python -m src.tasks.train.chronos.train --config src/tasks/train/chronos/configs/run-8.yml 