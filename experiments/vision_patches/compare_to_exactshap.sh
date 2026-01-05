#!/bin/bash
HERE="$(dirname "$0")"
NETWORK="${1-experiments/resources/mnist-cnn.eqx}"
network_filename=$(basename "$NETWORK")
network_name="${network_filename%.*}"

TIMEOUT=420
HARD_TIMEOUT=620  # +200 seconds for setup, etc
BAB_WARMUP_TIMEOUT=300  # 5min for downloading data and JAX compilation
EXACTSHAP_WARMUP_TIMEOUT=60

BATCH_SIZE=4096

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi
EXPERIMENT_DIR="$HERE/output/compare_to_exactshap/${network_name}_${TIMESTAMP}"

for num_patches in $(seq 5 10); do
  printf "\n\nRunning Warmup for Branch and Bound for ${num_patches} patches...\n"

  OUT_DIR="${EXPERIMENT_DIR}/warmup/BaB/${num_patches}_patches"
  mkdir -p "$OUT_DIR"
  python -m experiments.vision_patches.bound \
    --model "${NETWORK}" \
    --num-patches "${num_patches}" \
    --input 0 --output-feature 0 \
    --shap-variant "zero-baseline" \
    --bound-method "bab" \
    --bound-options "batch_size=$BATCH_SIZE" \
    --out "$OUT_DIR" \
    --max-iters 2 \
    --timeout "$BAB_WARMUP_TIMEOUT"

  sleep 15s
  printf "\n\nRunning Branch and Bound on ${network}...\n"

  OUT_DIR="${EXPERIMENT_DIR}/BaB/${num_patches}_patches"
  mkdir -p "$OUT_DIR"
  python -m experiments.vision_patches.bound \
    --model "${NETWORK}" \
    --num-patches "${num_patches}" \
    --input 0 --output-feature 0 \
    --shap-variant "zero-baseline" \
    --bound-method "bab" \
    --bound-options "batch_size=$BATCH_SIZE" \
    --out "$OUT_DIR" \
    --timeout "$TIMEOUT" \
    --silent
  sleep 15s
done


for num_patches in $(seq 5 10); do
  printf "\n\nRunning Warmup for ExactSHAP on ${num_patches} patches...\n"

  OUT_DIR="${EXPERIMENT_DIR}/warmup/ExactSHAP/${num_patches}_patches"
  mkdir -p "$OUT_DIR"
  timeout "$EXACTSHAP_WARMUP_TIMEOUT" \
    python -m experiments.vision_patches.exact_shap \
      --model "${NETWORK}" \
      --num-patches "${num_patches}" \
      --input 0 --output-feature 0 \
      --shap-variant "zero-baseline" \
      --out "$OUT_DIR" \
    
  sleep 15s
  printf "\n\nRunning ExactSHAP on ${network}...\n"

  OUT_DIR="$HERE/output/compare_to_exactshap/${TIMESTAMP}/ExactSHAP/${network_name}"
  mkdir -p "$OUT_DIR"
  timeout "$TIMEOUT" \
    python -m experiments.tabular.exact_shap \
      --model "${NETWORK}" \
      --num-patches "${num_patches}" \
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
