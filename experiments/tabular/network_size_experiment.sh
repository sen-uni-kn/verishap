#!/bin/bash
HERE="$(dirname "$0")"
SHAP_VARIANT="${1-mean-baseline}"

TIMEOUT=600
HARD_TIMEOUT=800  # +200 seconds for setup, etc
BAB_WARMUP_TIMEOUT=300  # 5min for downloading data and JAX compilation

BATCH_SIZE=4096
# Sorted by effective input dimension
NETWORKS=(
  "mushroom-mlp-1x1.eqx"
  "mushroom-mlp-2x1.eqx"
  "mushroom-mlp-4x1.eqx"
  "mushroom-mlp-8x1.eqx"
  "mushroom-mlp-16x1.eqx"
  "mushroom-mlp-32x1.eqx"
  "mushroom-mlp-64x1.eqx"
  "mushroom-mlp-128x1.eqx"
  "mushroom-mlp-256x1.eqx"
  "mushroom-mlp-512x1.eqx"
  "mushroom-mlp-1024x1.eqx"
  "mushroom-mlp-2048x1.eqx"
  "mushroom-mlp-4096x1.eqx"
  "mushroom-mlp-8192x1.eqx"
  "mushroom-mlp-16384x1.eqx"
  "mushroom-mlp-32768x1.eqx"
  "mushroom-mlp-1024x2.eqx"
  "mushroom-mlp-1024x4.eqx"
  "mushroom-mlp-1024x8.eqx"
  "mushroom-mlp-1024x16.eqx"
)

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi
EXPERIMENT_DIR="$HERE/output/network_size_experiment/${SHAP_VARIANT}_${TIMESTAMP}"

for network in "${NETWORKS[@]}"; do
  network_filename=$(basename "$network")
  network_name="${network_filename%.*}"

  printf "\n\nRunning Warmup for Branch and Bound on ${network}...\n"

  OUT_DIR="${EXPERIMENT_DIR}/warmup/BaB/${network_name}"
  mkdir -p "$OUT_DIR"
  python -m experiments.tabular.bound \
    --model "experiments/resources/${network}" \
    --input 0 --output-feature 0 \
    --shap-variant "${SHAP_VARIANT}" \
    --bound-method "bab" \
    --bound-options "batch_size=$BATCH_SIZE" \
    --out "$OUT_DIR" \
    --max-iters 2 \
    --timeout "$BAB_WARMUP_TIMEOUT"

  sleep 15s
  printf "\n\nRunning Branch and Bound on ${network}...\n"

  for i in {1..5}; do # repeat runtime measurements five times
    OUT_DIR="${EXPERIMENT_DIR}/BaB/${network_name}/repeatition_${i}"
    mkdir -p "$OUT_DIR"
    python -m experiments.tabular.bound \
      --model "experiments/resources/${network}" \
      --input 0 --output-feature 0 \
      --shap-variant "${SHAP_VARIANT}" \
      --bound-method "bab" \
      --bound-options "batch_size=$BATCH_SIZE" \
      --out "$OUT_DIR" \
      --timeout "$TIMEOUT" \
      --silent
    sleep 15s
  done
done

printf "Experiment complete.\n"
