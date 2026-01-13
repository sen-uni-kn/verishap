#!/bin/bash
HERE="$(dirname "$0")"
SHAP_VARIANT="${1-marginal}"

TIMEOUT=600
HARD_TIMEOUT=800  # +200 seconds for setup, etc
BAB_WARMUP_TIMEOUT=300  # 5min for downloading data and JAX compilation
EXACTSHAP_WARMUP_TIMEOUT=60

BATCH_SIZE=4096
# Sorted by effective input dimension
NETWORKS=(
  "adult-mlp-32x2.eqx"
  "obesity-mlp-32x2.eqx"
  "german-mlp-8x1.eqx"
  "mushroom-mlp-8x1.eqx"
  "default-mlp-64x3.eqx"
  "automobile-mlp-32x2.eqx"
  "dropout-mlp-32x2.eqx"
  "annealing-mlp-32x2.eqx"
  "steel-mlp-8x2.eqx"
  "hepatitis-mlp-8x2.eqx"
  "breast_cancer-mlp-32x2.eqx"
  # "infrared_temperature-mlp-32x2.eqx"
  # "ionosphere-mlp-32x2.eqx"
  "support2-mlp-8x2.eqx"
  "diabetes130-mlp-32x2.eqx"
  "lung_cancer-mlp-4x2.eqx"
  "online_news-mlp-8x2.eqx"
  "sonar-mlp-32x2.eqx"
  # "corrgroups60-mlp-16x2.eqx"
  # "independentlinear60-mlp-16x2.eqx"
  # "handwritten_digits-mlp-64x2.eqx"
  # "nhanesi-mlp-8x1.eqx"
  # "rt_iot-mlp-32x2.eqx"
  # "bankruptcy-mlp-32x2.eqx"
  # "communitiesandcrime-mlp-32x2.eqx"
)

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi
EXPERIMENT_DIR="$HERE/output/compare_to_exactshap/${SHAP_VARIANT}_${TIMESTAMP}"

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

# For the other networks, ExactSHAP ungracefully runs out of memory.
EXACTSHAP_NETWORKS=(
  "adult-mlp-32x2.eqx"
  "obesity-mlp-32x2.eqx"
  "german-mlp-8x1.eqx"
)

for network in "${EXACTSHAP_NETWORKS[@]}"; do
  network_filename=$(basename "$network")
  network_name="${network_filename%.*}"

  printf "\n\nRunning Warmup for ExactSHAP on ${network}...\n"

  OUT_DIR="${EXPERIMENT_DIR}/warmup/ExactSHAP/${network_name}"
  mkdir -p "$OUT_DIR"
  timeout "$EXACTSHAP_WARMUP_TIMEOUT" \
    python -m experiments.tabular.exact_shap \
      --model "experiments/resources/${network}" \
      --input 0 --output-feature 0 \
      --shap-variant "${SHAP_VARIANT}" \
      --out "$OUT_DIR" \
    
  sleep 15s
  printf "\n\nRunning ExactSHAP on ${network}...\n"

  for i in {1..5}; do
    OUT_DIR="${EXPERIMENT_DIR}/ExactSHAP/${network_name}/repeatition_${i}"
    mkdir -p "$OUT_DIR"
    timeout "$TIMEOUT" \
      python -m experiments.tabular.exact_shap \
        --model "experiments/resources/${network}" \
        --input 0 --output-feature 0 \
        --shap-variant "${SHAP_VARIANT}" \
        --out "$OUT_DIR" \
        --silent
    
    retVal=$?
    if [ $retVal -eq 124 ]; then  # timeout returns 124 if the timeout is reached
      break
    fi
    sleep 15s
  done
  if [ $retVal -eq 124 ]; then
    break
  fi
done

printf "Experiment complete.\n"
