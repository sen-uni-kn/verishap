#!/bin/bash
HERE="$(dirname "$0")"
SHAP_VARIANT="${1-marginal}"

TIMEOUT=600

BATCH_SIZE=4096
# Sorted by effective input dimension
NETWORKS=(
  "mushroom-mlp-8x1.eqx"
  "default-mlp-64x3.eqx"
  "automobile-mlp-32x2.eqx"
  # "steel-mlp-8x2.eqx"
  # "breast_cancer-mlp-32x2.eqx"
  "annealing-mlp-32x2.eqx"
  "sonar-mlp-32x2.eqx"
)

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi
EXPERIMENT_DIR="$HERE/output/bab_vs_sampling/${SHAP_VARIANT}_${TIMESTAMP}"

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
    --timeout "$TIMEOUT"

  sleep 15s
  printf "\n\nRunning Branch and Bound on ${network}...\n"

  OUT_DIR="${EXPERIMENT_DIR}/BaB/${network_name}"
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

for network in "${NETWORKS[@]}"; do
  network_filename=$(basename "$network")
  network_name="${network_filename%.*}"
  for estimator in "KernelSHAP" "PermutationSHAP" "LeverageSHAP" "LinearMSR" "TreeMSR"; do
    printf "================================================================================\n"
    printf "Running Warmup for ${estimator} on ${network}\n"
    printf "================================================================================\n"

    OUT_DIR="${EXPERIMENT_DIR}/warmup/${estimator}/${network_name}"
    mkdir -p "$OUT_DIR"
    python -m experiments.tabular.estimate \
      --model "experiments/resources/${network}" \
      --input 0 --output-feature 0 \
      --shap-variant "${SHAP_VARIANT}" \
      --estimator "${estimator}" \
      --out "$OUT_DIR" \
      --num-samples "200" \
      --seed 0

    sleep 5s
    printf "================================================================================\n"
    printf "Running ${estimator} on ${network}\n"
    printf "================================================================================\n"
    
    OUT_DIR="${EXPERIMENT_DIR}/${estimator}/${network_name}/seed_${seed}/${num_samples}"
    mkdir -p "$OUT_DIR"
    python -m experiments.tabular.estimate \
      --model "experiments/resources/${network}" \
      --input 0 --output-feature 0 \
      --shap-variant "${SHAP_VARIANT}" \
      --estimator "${estimator}" \
      --timeout "$TIMEOUT" \
      --out "$OUT_DIR" \
      --num-samples 200:10100:100 \
      --seeds $(seq 0 99) \
      --silent
    
    sleep 5s
  done
done

printf "Experiment complete.\n"
