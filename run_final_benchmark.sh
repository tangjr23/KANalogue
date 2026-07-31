#!/bin/bash
# Final KANalogue benchmark: seed=42 + 10 random seeds on MNIST/FMNIST
# Uses both GPUs, avoids OOM by running sequentially per GPU

set -e
BASE_DIR="/mnt/data/aipss/KANanlogue"
PYTHON="/home/aipss/miniconda3/envs/50torch/bin/python"
export PYTHONPATH="$BASE_DIR"

# Configs: name, dataset, hidden, basis_types, lr
CONFIGS=(
  "MNIST_2dim|MNIST|64|Z15 ZAZ_21|0.000195"
  "MNIST_3dim|MNIST|64|Z21 Z15 ZAZ_21|0.000195"
  "FMNIST_2dim|FMNIST|256|Z15 ZAZ_21|0.000546"
  "FMNIST_3dim|FMNIST|256|Z21 Z15 ZAZ_21|0.00018"
)

# 10 random seeds generated with seed(1)
SEEDS=(2202 9326 1034 4180 1932 8118 7365 7738 6220 3440)

EXP_DIR="results/new_structure_exps/final_kanalogue"
SUMMARY="results/final_kanalogue_summary.json"

mkdir -p results

run_one() {
  local name=$1 ds=$2 hidden=$3 basis=$4 lr=$5 seed=$6 gpu=$7
  local d="$EXP_DIR/${name}/seed_${seed}"
  [ -f "$d/${ds}/configs/"*"_best_config.json" ] 2>/dev/null && { echo "  SKIP $name s=$seed"; return 0; }
  echo "  RUN  $name s=$seed GPU=$gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PYTHON -m kanalogue.cli.train \
    --dataset $ds --device cuda:0 --hidden $hidden \
    --td_basis_types $basis --learning_rate $lr --batch_size 32 \
    --acti PosHC --norm_layer batch --fit_mode univariate --basis_type pos-larger \
    --max_epochs 2000 --patience 15 \
    --exp_name "final_kanalogue/${name}/seed_${seed}" \
    --seed $seed --search_mode custom --model_type tdkan \
    > /tmp/final_${name}_${seed}.log 2>&1
  echo "  DONE $name s=$seed"
}

# === TASK 1: Seed 42 on GPU 0 ===
echo "============================================"
echo "  TASK 1: Seed 42 (4 configs)"
echo "============================================"
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name ds hidden basis lr <<< "$cfg"
  run_one "$name" "$ds" "$hidden" "$basis" "$lr" 42 0
done

# === TASK 2: 10 seeds on both GPUs (round-robin) ===
echo ""
echo "============================================"
echo "  TASK 2: 10 random seeds (4 configs x 10 = 40 runs)"
echo "============================================"
gpu=0
count=0
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name ds hidden basis lr <<< "$cfg"
  for seed in "${SEEDS[@]}"; do
    run_one "$name" "$ds" "$hidden" "$basis" "$lr" "$seed" "$gpu"
    # Alternate GPUs
    ((count++))
    gpu=$((count % 2))
  done
done

echo ""
echo "============================================"
echo "  ALL DONE"
echo "============================================"
