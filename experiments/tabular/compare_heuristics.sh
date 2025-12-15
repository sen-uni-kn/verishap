#!/bin/bash

HERE="$(dirname "$0")"
TIMEOUT=900  # 15min
HARD_TIMEOUT=960  # +60 seconds for setup, etc
WARMUP_TIMEOUT=300  # 5min for downloading data and JAX compilation

BATCH_SIZE=16384

# Sorted by input dimension
NETWORKS=(
  "diabetes-mlp-32x1.eqx"
  "adult-mlp-32x2.eqx"
  "obesity-mlp-32x2.eqx"
  "vehicles-mlp-32x2.eqx"
  "german-mlp-8x1.eqx"
  "mushroom-mlp-8x1.eqx"
  "automobile-mlp-32x2.eqx"
  "steel-mlp-8x2.eqx"
  "hepatitis-mlp-8x2.eqx"
  "breast_cancer-mlp-32x2.eqx"
  "infrared_temperature-mlp-32x2.eqx"
  "ionosphere-mlp-32x2.eqx"
  "dropout-mlp-32x2.eqx"
  "annealing-mlp-32x2.eqx"
  "support2-mlp-8x2.eqx"
  "diabetes130-mlp-32x2.eqx"
)

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi

for network in "${NETWORKS[@]}"; do
  network_filename=$(basename "$network")
  network_name="${network_filename%.*}"

  printf "\n\nRunning Warmup for Branch and Bound on ${network}...\n"

  OUT_DIR="$HERE/output/compare_heuristics/${TIMESTAMP}/warmup/BaB/${network_name}"
  mkdir -p "$OUT_DIR"
  timeout "$WARMUP_TIMEOUT" \
    python -m experiments.tabular.bound \
      --model "experiments/tabular/resources/${network}" \
      --input 0 --output-feature 0 \
      --shap-variant "zero-baseline" \
      --bound-method "bab" \
      --bound-options "batch_size=$BATCH_SIZE" \
      --out "$OUT_DIR" \
      --max-iters 2 \
      --timeout "$WARMUP_TIMEOUT"

for i in {1..5}  # repeat runtime measurements five times
do
  for compute_bounds in "ibp" "crown_ibp" "crown" "alpha-crown"
  do
    for select_strategy in "max-diam" "min-diam" "fifo" "lifo"
    do
      for split_strategy in "longest-edge" "smears" \
        "lirpa-weights" "strong-branching-better" "strong-branching-worse" \
        "smart-branching-ibp-better" "smart-branching-ibp-worse"
      do
        for network in "${NETWORKS[@]}"; do
          OUT_DIR="$HERE/output/compare_heuristics/${TIMESTAMP}/repeatition_${i}/${compute_bounds}/${select_strategy}/${split_strategy}/${network_name}"
          mkdir -p "$OUT_DIR"
          timeout "$HARD_TIMEOUT" \
            python -m experiments.tabular.bound \
              --model "experiments/tabular/resources/${network}" \
              --input 0 --output-feature 0 \
              --shap-variant "zero-baseline" \
              --bound-method "bab" \
              --bound-options "batch_size=$BATCH_SIZE,compute_bounds=$compute_bounds,select_strategy=$select_strategy,split_strategy=$split_strategy" \
              --out "$OUT_DIR" \
              --timeout "$TIMEOUT"
              --silent
        done
      done
    done
  done
done
