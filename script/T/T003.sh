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

# T003: TPsiAct experiment on CUB-200-2011
# Uses ViT backbone with TPsiAct activation
python3 -u src/main.py \
  --dataset cub_200_2011 \
  --download \
  --backbone vit_base_patch16_224.augreg2_in21k_ft_in1k \
  --pretrained \
  --use-tpsiact \
  --tpsiact-nu 5.0 \
  --epochs 300 \
  --batch-size 32 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --optimizer adamw \
  --scheduler cosine \
  --warmup-epochs 10 \
  --label-smoothing 0.1 \
  --knn-k 200 \
  --knn-chunk-size 200 \
  --eval-knn-every 5 \
  --seed 42 \
  --num-workers 8 \
  --save-summary-json \
  --save-summary-csv \
  --save-final-pt \
  --save-dir ./experiments \
  --experiment-name "T003_tpsiact_vit_small_cub200_nu5" \
  >> "T003.log" 2>&1