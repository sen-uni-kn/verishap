#!/bin/bash

HERE="$(dirname "$0")"
TIMEOUT=900  # 15min
HARD_TIMEOUT=1100  # +200 seconds for setup, saving results, etc
BAB_WARMUP_TIMEOUT=300  # 5min for downloading data and JAX compilation
EXACTSHAP_WARMUP_TIMEOUT=60

BATCH_SIZE=4096
NETWORK="$1"
SIZE="$2"
network_filename=$(basename "$NETWORK")
network_name="${network_filename%.*}"

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi
EXPERIMENT_DIR = "${network_name}_${TIMESTAMP}"


WARMUP_SIZE=$((SIZE - 15))
printf "\n\nRunning Warmup for ExactSHAP excluding ${WARMUP_SIZE} features...\n"

OUT_DIR="$HERE/output/compare_to_exactshap/${EXPERIMENT_DIR}/warmup/ExactSHAP/"
mkdir -p "$OUT_DIR"
timeout "$EXACTSHAP_WARMUP_TIMEOUT" \
  python -m experiments.tabular.exact_shap \
    --model "${NETWORK}" \
    --set-to-baseline ${WARMUP_SIZE} \
    --input 0 --output-feature 0 \
    --shap-variant "zero-baseline" \
    --out "$OUT_DIR" \

sleep 15s
printf "\n\nRunning Warmup for Branch and Bound excluding ${WARMUP_SIZE} features...\n"

OUT_DIR="$HERE/output/compare_to_exactshap/${TIMESTAMP}/warmup/BaB/${network_name}"
mkdir -p "$OUT_DIR"
timeout "$BAB_WARMUP_TIMEOUT" \
  python -m experiments.tabular.bound \
    --model "${NETWORK}" \
    --set-to-baseline ${WARMUP_SIZE} \
    --input 0 --output-feature 0 \
    --shap-variant "zero-baseline" \
    --bound-method "bab" \
    --bound-options "batch_size=$BATCH_SIZE" \
    --out "$OUT_DIR" \
    --max-iters 2 \
    --timeout "$BAB_WARMUP_TIMEOUT"

RUN_EXACTSHAP=true  # stop running ExactSHAP after it times out once
for ((i=$(($SIZE - 10)); i>=0; i--)); do
  if [ "$RUN_EXACTSHAP" = true ]; then
    sleep 15s
    printf "\n\nRunning ExactSHAP excluding ${i} features...\n"

    OUT_DIR="$HERE/output/compare_to_exactshap/${EXPERIMENT_DIR}/ExactSHAP/${i}"
    mkdir -p "$OUT_DIR"
    timeout --kill-after=60 "$TIMEOUT" \
      python -m experiments.tabular.exact_shap \
        --model "${NETWORK}" \
        --set-to-baseline $i \
        --input 0 --output-feature 0 \
        --shap-variant "zero-baseline" \
        --out "$OUT_DIR" \
        --silent
    
    retVal=$?
    if [ $retVal -eq 124 ]; then  # timeout returns 124 if the timeout is reached
      RUN_EXACTSHAP=false
    fi
  fi

  sleep 15s
  printf "\n\nRunning Branch and Bound excluding ${i} features...\n"

  OUT_DIR="$HERE/output/compare_to_exactshap/${EXPERIMENT_DIR}/BaB/${i}"
  mkdir -p "$OUT_DIR"
  timeout --kill-after=60 "$HARD_TIMEOUT" \
    python -m experiments.tabular.bound \
      --model "${NETWORK}" \
      --set-to-baseline $i \
      --input 0 --output-feature 0 \
      --shap-variant "zero-baseline" \
      --bound-method "bab" \
      --bound-options "batch_size=$BATCH_SIZE" \
      --out "$OUT_DIR" \
      --timeout "$TIMEOUT" \
      --silent
done

printf "Experiment complete.\n"
