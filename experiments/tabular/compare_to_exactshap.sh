#!/bin/bash

HERE="$(dirname "$0")"
TIMEOUT=900  # 15min
HARD_TIMEOUT=960  # +60 seconds for setup, etc
BAB_WARMUP_TIMEOUT=300  # 5min for downloading data and JAX compilation
EXACTSHAP_WARMUP_TIMEOUT=60

BATCH_SIZE=16384
NETWORKS=(
  "adult-mlp-32x2.eqx"
  "annealing-mlp-32x2.eqx"
  "automobile-mlp-32x2.eqx"
  "bankruptcy-mlp-32x2.eqx"
  "breast_cancer-mlp-32x2.eqx"
  "california-mlp-64x2.eqx"
  "cdc_diabetes-mlp-128x3.eqx"
  "communitiesandcrime-mlp-32x2.eqx"
  "covertype-mlp-32x2.eqx"
  "default-mlp-64x3.eqx"
  "diabetes-mlp-32x1.eqx"
  "diabetes130-mlp-32x2.eqx"
  "dropout-mlp-32x2.eqx"
  "german-mlp-8x1.eqx"
  "handwritten_digits-mlp-64x2.eqx"
  "hepatitis-mlp-8x2.eqx"
  "infrared_temperature-mlp-32x2.eqx"
  "ionosphere-mlp-32x2.eqx"
  "iris-mlp-32x2.eqx"
  "lung_cancer-mlp-4x2.eqx"
  "mushroom-mlp-8x1.eqx"
  "nhanesi-mlp-8x1.eqx"
  "obesity-mlp-32x2.eqx"
  "online_news-mlp-8x2.eqx"
  "parkinsons-mlp-32x2.eqx"
  "rt_iot-mlp-32x2.eqx"
  "sonar-mlp-32x2.eqx"
  "spambase-mlp-32x2.eqx"
  "steel-mlp-8x2.eqx"
  "support2-mlp-8x2.eqx"
  "vehicles-mlp-32x2.eqx"
)

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi

RUN_EXACTSHAP=true

for network in "${NETWORKS[@]}"; do
  if [ "$RUN_EXACTSHAP" = true ]; then
    printf "Running Warmup for ExactSHAP on ${network}...\n"

    OUT_DIR="$HERE/output/compare_to_exactshap/${TIMESTAMP}/warmup/ExactSHAP/${network}"
    mkdir -p "$OUT_DIR"
    timeout "$TIMEOUT" \
      python -m experiments.tabular.exact_shap \
        --model "experiments/tabular/resources/${network}" \
        --input 0 --output-feature 0 \
        --shap-variant "zero-baseline" \
        --out "$OUT_DIR" \

    printf "Running ExactSHAP on ${network}...\n"

    OUT_DIR="$HERE/output/compare_to_exactshap/${TIMESTAMP}/ExactSHAP/${network}"
    mkdir -p "$OUT_DIR"
    timeout "$EXACTSHAP_WARMUP_TIMEOUT" \
      python -m experiments.tabular.exact_shap \
        --model "experiments/tabular/resources/${network}" \
        --input 0 --output-feature 0 \
        --shap-variant "zero-baseline" \
        --out "$OUT_DIR" \
        --silent
    
    retVal=$?
    if [ $retVal -eq 124 ]; then  # timeout returns 124 if the timeout is reached
      RUN_EXACTSHAP=false
    fi
  fi

  printf "Running Warmup for Branch and Bound on ${network}...\n"

  OUT_DIR="$HERE/output/compare_to_exactshap/${TIMESTAMP}/warmup/Ours/${network}"
  mkdir -p "$OUT_DIR"
  timeout "$BAB_WARMUP_TIMEOUT" \
    python -m experiments.tabular.bound \
      --model "experiments/tabular/resources/${network}" \
      --input 0 --output-feature 0 \
      --shap-variant "zero-baseline" \
      --bound-method "bab" \
      --bound-options "batch_size=$BATCH_SIZE" \
      --out "$OUT_DIR" \
      --max-iters 10 \
      --timeout "$BAB_WARMUP_TIMEOUT"

  printf "Running Branch and Bound on ${network}...\n"

  OUT_DIR="$HERE/output/compare_to_exactshap/${TIMESTAMP}/warmup/Ours/${network}"
  mkdir -p "$OUT_DIR"
  timeout "$HARD_TIMEOUT" \
    python -m experiments.tabular.bound \
      --model "experiments/tabular/resources/${network}" \
      --input 0 --output-feature 0 \
      --shap-variant "zero-baseline" \
      --bound-method "bab" \
      --bound-options "batch_size=$BATCH_SIZE" \
      --out "$OUT_DIR" \
      --timeout "$TIMEOUT" \
      --silent
done

printf "Experiment complete.\n"