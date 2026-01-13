#!/bin/bash
HERE="$(dirname "$0")"
SHAP_VARIANT="${1-marginal}"

TIMEOUT=400  # seconds
HARD_TIMEOUT=600  # +200 seconds for setup, etc
WARMUP_TIMEOUT=300  # 5min for downloading data and JAX compilation

BATCH_SIZE=4096
# Sorted by input dimension
NETWORKS=(
  "adult-mlp-32x2.eqx"
  "obesity-mlp-32x2.eqx"
  "german-mlp-8x1.eqx"
  "mushroom-mlp-8x1.eqx"
  "default-mlp-64x3.eqx"
  "automobile-mlp-32x2.eqx"
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

  OUT_DIR="$HERE/output/compare_heuristics/${TIMESTAMP}/warmup/BaB/${network_name}"
  mkdir -p "$OUT_DIR"
  timeout "$WARMUP_TIMEOUT" \
    python -m experiments.tabular.bound \
      --model "experiments/resources/${network}" \
      --input 0 --output-feature 0 \
      --shap-variant "marginal" \
      --bound-method "bab" \
      --bound-options "batch_size=$BATCH_SIZE" \
      --out "$OUT_DIR" \
      --max-iters 2 \
      --timeout "$WARMUP_TIMEOUT"
  
  for split_strategy in "longest-edge" "smears" \
    "strong-branching-better" "strong-branching-worse" \
    "smart-branching-ibp-better" "smart-branching-ibp-worse"; do
    for i in {1..5}; do # repeat runtime measurements five times
      OUT_DIR="$HERE/output/compare_heuristics/${TIMESTAMP}/split_strategies/${split_strategy}/${network_name}/repeatition_${i}"
      mkdir -p "$OUT_DIR"
      timeout "$HARD_TIMEOUT" \
        python -m experiments.tabular.bound \
          --model "experiments/resources/${network}" \
          --input 0 --output-feature 0 \
          --shap-variant "${SHAP_VARIANT}" \
          --bound-method "bab" \
          --bound-options "batch_size=$BATCH_SIZE,split_strategy=$split_strategy" \
          --out "$OUT_DIR" \
          --timeout "$TIMEOUT"
          --silent
    done
  done

  for select_strategy in "max-diam" "min-diam" "fifo" "lifo"; do
    for i in {1..5}; do # repeat runtime measurements five times
      OUT_DIR="$HERE/output/compare_heuristics/${TIMESTAMP}/select_strategies/${select_strategy}/${network_name}/repeatition_${i}"
      mkdir -p "$OUT_DIR"
      timeout "$HARD_TIMEOUT" \
        python -m experiments.tabular.bound \
          --model "experiments/resources/${network}" \
          --input 0 --output-feature 0 \
          --shap-variant "${SHAP_VARIANT}" \
          --bound-method "bab" \
          --bound-options "batch_size=$BATCH_SIZE,select_strategy=$select_strategy" \
          --out "$OUT_DIR" \
          --timeout "$TIMEOUT"
          --silent
    done
  done

  for compute_bounds in "ibp" "crown_ibp" "crown" "alpha-crown"; do
    for i in {1..5}; do # repeat runtime measurements five times
      OUT_DIR="$HERE/output/compare_heuristics/${TIMESTAMP}/compute_bounds/${compute_bounds}/${network_name}/repeatition_${i}"
      mkdir -p "$OUT_DIR"
      timeout "$HARD_TIMEOUT" \
        python -m experiments.tabular.bound \
          --model "experiments/resources/${network}" \
          --input 0 --output-feature 0 \
          --shap-variant "zero-baseline" \
          --bound-method "bab" \
          --bound-options "batch_size=$BATCH_SIZE,compute_bounds=$compute_bounds" \
          --out "$OUT_DIR" \
          --timeout "$TIMEOUT"
          --silent
    done
  done
done
