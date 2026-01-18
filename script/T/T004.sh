#!/bin/bash
#PBS -P yp87
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=16GB
#PBS -l walltime=08:00:00
#PBS -l wd
#PBS -l storage=scratch/yp87

module load cuda/12.6.2

source /scratch/yp87/sl5952/TPsiAct/.venv/bin/activate
export HF_HOME="/scratch/yp87/sl5952/CARROT/.cache"
export HF_HUB_OFFLINE=1

cd ../..

# T004: TPsiAct with ViT backbone
python3 -u src/main.py \
  --dataset cub_200_2011 \
  --download \
  --backbone vit_small_patch16_224 \
  --pretrained \
  --use-tpsiact \
  --tpsiact-nu 5.0 \
  --epochs 300 \
  --batch-size 32 \
  --lr 5e-4 \
  --weight-decay 1e-4 \
  --optimizer adamw \
  --scheduler cosine \
  --warmup-epochs 10 \
  --knn-k 200 \
  --knn-chunk-size 200 \
  --eval-knn-every 5 \
  --seed 42 \
  --num-workers 8 \
  --save-summary-json \
  --save-summary-csv \
  --save-final-pt \
  --save-dir ./experiments \
  --experiment-name "T004_tpsiact_vit_small_cub200" \
  >> "T004.log" 2>&1
