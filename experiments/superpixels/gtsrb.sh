#!/bin/bash
HERE="$(dirname "$0")"
SHAP_VARIANT="${1-mean-baseline}"
BATCH_SIZE="${2-1024}"
NETWORK="${3-experiments/resources/gtsrb-cnn2.eqxparams}"
network_filename=$(basename "$NETWORK")
network_name="${network_filename%.*}"

TIMEOUT=7200
HARD_TIMEOUT=7800  # +200 seconds for setup, etc
BAB_WARMUP_TIMEOUT=300  # 5min for downloading data and JAX compilation

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi
EXPERIMENT_DIR="$HERE/output/${network_name}_${SHAP_VARIANT}_${TIMESTAMP}"

printf "\n\nRunning Warmup for Branch and Bound for 25 features...\n"

OUT_DIR="${EXPERIMENT_DIR}/warmup/BaB/25_features"
mkdir -p "$OUT_DIR"
python -m experiments.superpixels.bound \
  --model "${NETWORK}" \
  --num-features "25" \
  --output-feature "16" \
  --shap-variant "${SHAP_VARIANT}" \
  --bound-method "bab" \
  --bound-options "batch_size=$BATCH_SIZE" \
  --out "$OUT_DIR" --max-iters 2 \
  --timeout "$BAB_WARMUP_TIMEOUT"

sleep 15s
printf "\n\nRunning Branch and Bound on 26 features...\n"

OUT_DIR="${EXPERIMENT_DIR}/BaB/26_features"
mkdir -p "$OUT_DIR"
python -m experiments.superpixels.bound \
  --model "${NETWORK}" \
  --num-features "26" \
  --output-feature "38" \
  --shap-variant "${SHAP_VARIANT}" \
  --bound-method "bab" \
  --bound-options "batch_size=$BATCH_SIZE" \
  --out "$OUT_DIR" \
  --timeout "$TIMEOUT" \
  --silent
sleep 15s

sleep 15s
printf "\n\nRunning Branch and Bound on 27 features...\n"

OUT_DIR="${EXPERIMENT_DIR}/BaB/27_features"
mkdir -p "$OUT_DIR"
python -m experiments.superpixels.bound \
  --model "${NETWORK}" \
  --num-features "27" \
  --output-feature "11" \
  --shap-variant "${SHAP_VARIANT}" \
  --bound-method "bab" \
  --bound-options "batch_size=$BATCH_SIZE" \
  --out "$OUT_DIR" \
  --timeout "$TIMEOUT" \
  --silent

printf "Experiment complete.\n"

