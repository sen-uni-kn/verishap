#!/bin/bash
HERE="$(dirname "$0")"
MNIST_SAMPLE="${1-43}"
MNIST_OUTPUT="${2-4}"
SHAP_VARIANT="${3-mean-baseline}"
BATCH_SIZE="${4-4096}"
NETWORK="${5-experiments/resources/mnist-cnn.eqx}"
MIN_PATCHES="${6-5}"
MAX_PATCHES="${7-7}"
network_filename=$(basename "$NETWORK")
network_name="${network_filename%.*}"

TIMEOUT=900
HARD_TIMEOUT=1100  # +200 seconds for setup, etc
BAB_WARMUP_TIMEOUT=300  # 5min for downloading data and JAX compilation

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi
EXPERIMENT_DIR="$HERE/output/${network_name}_${SHAP_VARIANT}_${TIMESTAMP}"

for num_patches in $(seq ${MIN_PATCHES} ${MAX_PATCHES}); do
  printf "\n\nRunning Warmup for Branch and Bound for ${num_patches} patches...\n"

  OUT_DIR="${EXPERIMENT_DIR}/warmup/BaB/${num_patches}_patches"
  mkdir -p "$OUT_DIR"
  python -m experiments.vision_patches.bound \
    --model "${NETWORK}" \
    --num-patches "${num_patches}" \
    --input "${MNIST_SAMPLE}" --output-feature "${MNIST_OUTPUT}" \
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
    --input "${MNIST_SAMPLE}" --output-feature "${MNIST_OUTPUT}" \
    --shap-variant "${SHAP_VARIANT}" \
    --bound-method "bab" \
    --bound-options "batch_size=$BATCH_SIZE" \
    --out "$OUT_DIR" \
    --timeout "$TIMEOUT" \
    --silent
  sleep 15s
done

printf "Experiment complete.\n"

