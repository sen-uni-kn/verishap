#!/bin/bash

HERE="$(dirname "$0")"
TIMEOUT=420
HARD_TIMEOUT=620  # +200 seconds for setup, etc
BAB_WARMUP_TIMEOUT=300  # 5min for downloading data and JAX compilation
EXACTSHAP_WARMUP_TIMEOUT=60

BATCH_SIZE=4096
# Sorted by effective input dimension
NETWORKS=(
  # "iris-mlp-32x2.eqx"
  # "california-mlp-64x2.eqx"
  # "diabetes-mlp-32x1.eqx"
  # "spambase-mlp-32x2.eqx"
  # "covertype-mlp-32x2.eqx"
  "adult-mlp-32x2.eqx"
  "obesity-mlp-32x2.eqx"
  # "vehicles-mlp-32x2.eqx"
  # "parkinsons-mlp-32x2.eqx"
  "german-mlp-8x1.eqx"
  # "cdc_diabetes-mlp-128x3.eqx"
  "mushroom-mlp-8x1.eqx"
  "default-mlp-64x3.eqx"
  "automobile-mlp-32x2.eqx"
  "dropout-mlp-32x2.eqx"
  "annealing-mlp-32x2.eqx"
  "steel-mlp-8x2.eqx"
  "hepatitis-mlp-8x2.eqx"
  "breast_cancer-mlp-32x2.eqx"
  "infrared_temperature-mlp-32x2.eqx"
  "ionosphere-mlp-32x2.eqx"
  "support2-mlp-8x2.eqx"
  "diabetes130-mlp-32x2.eqx"
  "lung_cancer-mlp-4x2.eqx"
  "online_news-mlp-8x2.eqx"
  "sonar-mlp-32x2.eqx"
  # "corrgroups60-mlp-16x2.eqx"
  # "independentlinear60-mlp-16x2.eqx"
  "handwritten_digits-mlp-64x2.eqx"
  "nhanesi-mlp-8x1.eqx"
  # "rt_iot-mlp-32x2.eqx"
  # "bankruptcy-mlp-32x2.eqx"
  # "communitiesandcrime-mlp-32x2.eqx"
)

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi

for network in "${NETWORKS[@]}"; do
  network_filename=$(basename "$network")
  network_name="${network_filename%.*}"

  printf "\n\nRunning Warmup for Branch and Bound on ${network}...\n"

  OUT_DIR="$HERE/output/compare_to_exactshap/${TIMESTAMP}/warmup/BaB/${network_name}"
  mkdir -p "$OUT_DIR"
  python -m experiments.tabular.bound \
    --model "experiments/resources/${network}" \
    --input 0 --output-feature 0 \
    --shap-variant "zero-baseline" \
    --bound-method "bab" \
    --bound-options "batch_size=$BATCH_SIZE" \
    --out "$OUT_DIR" \
    --max-iters 2 \
    --timeout "$BAB_WARMUP_TIMEOUT"

  sleep 15s
  printf "\n\nRunning Branch and Bound on ${network}...\n"

  OUT_DIR="$HERE/output/compare_to_exactshap/${TIMESTAMP}/BaB/${network_name}"
  mkdir -p "$OUT_DIR"
  python -m experiments.tabular.bound \
    --model "experiments/resources/${network}" \
    --input 0 --output-feature 0 \
    --shap-variant "zero-baseline" \
    --bound-method "bab" \
    --bound-options "batch_size=$BATCH_SIZE" \
    --out "$OUT_DIR" \
    --timeout "$TIMEOUT" \
    --silent
  sleep 15s
done

for network in "${NETWORKS[@]}"; do
  network_filename=$(basename "$network")
  network_name="${network_filename%.*}"

  printf "\n\nRunning Warmup for ExactSHAP on ${network}...\n"

  OUT_DIR="$HERE/output/compare_to_exactshap/${TIMESTAMP}/warmup/ExactSHAP/${network_name}"
  mkdir -p "$OUT_DIR"
  timeout "$EXACTSHAP_WARMUP_TIMEOUT" \
    python -m experiments.tabular.exact_shap \
      --model "experiments/resources/${network}" \
      --input 0 --output-feature 0 \
      --shap-variant "zero-baseline" \
      --out "$OUT_DIR" \
    
  sleep 15s
  printf "\n\nRunning ExactSHAP on ${network}...\n"

  OUT_DIR="$HERE/output/compare_to_exactshap/${TIMESTAMP}/ExactSHAP/${network_name}"
  mkdir -p "$OUT_DIR"
  timeout "$TIMEOUT" \
    python -m experiments.tabular.exact_shap \
      --model "experiments/resources/${network}" \
      --input 0 --output-feature 0 \
      --shap-variant "zero-baseline" \
      --out "$OUT_DIR" \
      --silent
  
  retVal=$?
  if [ $retVal -eq 124 ]; then  # timeout returns 124 if the timeout is reached
    break
  fi
  sleep 15s
done

printf "Experiment complete.\n"
