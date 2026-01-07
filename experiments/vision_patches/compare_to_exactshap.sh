#!/bin/bash
HERE="$(dirname "$0")"
SHAP_VARIANT="${1-zero-baseline}"
NETWORK="${2-experiments/resources/mnist-cnn.eqx}"
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
EXPERIMENT_DIR="$HERE/output/compare_to_exactshap/${network_name}_${SHAP_VARIANT}_${TIMESTAMP}"

for num_patches in $(seq 5 11); do  # values above 11 go out of memory.
  printf "\n\nRunning Warmup for Branch and Bound for ${num_patches} patches...\n"

  OUT_DIR="${EXPERIMENT_DIR}/warmup/BaB/${num_patches}_patches"
  mkdir -p "$OUT_DIR"
  python -m experiments.vision_patches.bound \
    --model "${NETWORK}" \
    --num-patches "${num_patches}" \
    --input 0 --output-feature 0 \
    --shap-variant "${SHAP_VARIANT}" \
    --bound-method "bab" \
    --bound-options "batch_size=$BATCH_SIZE" \
    --out "$OUT_DIR" --max-iters 2 \
    --timeout "$BAB_WARMUP_TIMEOUT"

  sleep 15s
  printf "\n\nRunning Branch and Bound on ${num_patches} patches...\n"

  OUT_DIR="${EXPERIMENT_DIR}/BaB/${num_patches}_patches"
  mkdir -p "$OUT_DIR"
  python -m experiments.vision_patches.bound \
    --model "${NETWORK}" \
    --num-patches "${num_patches}" \
    --input 0 --output-feature 0 \
    --shap-variant "${SHAP_VARIANT}" \
    --bound-method "bab" \
    --bound-options "batch_size=$BATCH_SIZE" \
    --out "$OUT_DIR" \
    --timeout "$TIMEOUT" \
    --silent
  sleep 15s
done

# All values above 5 run out of memory.
for num_patches in $(seq 5 6); do
  printf "\n\nRunning Warmup for ExactSHAP on ${num_patches} patches...\n"

  OUT_DIR="${EXPERIMENT_DIR}/warmup/ExactSHAP/${num_patches}_patches"
  mkdir -p "$OUT_DIR"
  timeout "$EXACTSHAP_WARMUP_TIMEOUT" \
    python -m experiments.vision_patches.exact_shap \
      --model "${NETWORK}" \
      --num-patches "${num_patches}" \
      --input 0 --output-feature 0 \
      --shap-variant "${SHAP_VARIANT}" \
      --out "$OUT_DIR" \
    
  sleep 15s
  printf "\n\nRunning ExactSHAP on ${num_patches} patches...\n"

  OUT_DIR="${EXPERIMENT_DIR}/ExactSHAP/${num_patches}_patches"
  mkdir -p "$OUT_DIR"
  timeout "$TIMEOUT" \
    python -m experiments.vision_patches.exact_shap \
      --model "${NETWORK}" \
      --num-patches "${num_patches}" \
      --input 0 --output-feature 0 \
      --shap-variant "${SHAP_VARIANT}" \
      --out "$OUT_DIR" \
      --silent
  sleep 15s
done

printf "Experiment complete.\n"
