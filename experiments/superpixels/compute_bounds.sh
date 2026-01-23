#!/bin/bash
HERE="$(dirname "$0")"
SAMPLE="${1-0}"
OUTPUT_FEATURE="${2-0}"
SHAP_VARIANT="${3-mean-baseline}"
BATCH_SIZE="${4-4096}"
NETWORK="${5-experiments/resources/cifar10-cnn.eqx}"
MIN_FEATURES="${6-25}"
MAX_FEATURES="${7-30}"
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

for num_features in $(seq ${MIN_FEATURES} ${MAX_FEATURES}); do
  printf "\n\nRunning Warmup for Branch and Bound for ${num_features} features...\n"

  OUT_DIR="${EXPERIMENT_DIR}/warmup/BaB/${num_features}_features"
  mkdir -p "$OUT_DIR"
  python -m experiments.vision_patches.bound \
    --model "${NETWORK}" \
    --num-features "${num_features}" \
    --input "${SAMPLE}" --output-feature "${OUTPUT_FEATURE}" \
    --shap-variant "${SHAP_VARIANT}" \
    --bound-method "bab" \
    --bound-options "batch_size=$BATCH_SIZE" \
    --out "$OUT_DIR" --max-iters 2 \
    --timeout "$BAB_WARMUP_TIMEOUT"

  sleep 15s
  printf "\n\nRunning Branch and Bound on ${num_features} features...\n"

  OUT_DIR="${EXPERIMENT_DIR}/BaB/${num_features}_features"
  mkdir -p "$OUT_DIR"
  python -m experiments.vision_patches.bound \
    --model "${NETWORK}" \
    --num-features "${num_features}" \
    --input "${SAMPLE}" --output-feature "${OUTPUT_FEATURE}" \
    --shap-variant "${SHAP_VARIANT}" \
    --bound-method "bab" \
    --bound-options "batch_size=$BATCH_SIZE" \
    --out "$OUT_DIR" \
    --timeout "$TIMEOUT" \
    --silent
  sleep 15s
done

printf "Experiment complete.\n"

