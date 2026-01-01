#!/bin/bash
HERE="$(dirname "$0")"
TIMEOUT=900  # 15min

BATCH_SIZE=4096
NETWORK="$1"
MAX_SIZE="$2"
network_filename=$(basename "$NETWORK")
network_name="${network_filename%.*}"

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi
EXPERIMENT_DIR="$HERE/output/compare_to_exactshap2/${network_name}_${TIMESTAMP}"


WARMUP_SIZE=13
printf "================================================================================\n"
printf "Running Warmup for ExactSHAP with ${WARMUP_SIZE} effective features...\n"
printf "================================================================================\n"

OUT_DIR="${EXPERIMENT_DIR}/warmup/ExactSHAP/"
mkdir -p "$OUT_DIR"
python -m experiments.tabular.exact_shap \
  --model "${NETWORK}" \
  --effective-features ${WARMUP_SIZE} \
  --input 0 --output-feature 0 \
  --shap-variant "zero-baseline" \
  --out "$OUT_DIR"

sleep 15s
printf "================================================================================\n"
printf "Running Warmup for Branch and Bound with ${WARMUP_SIZE} effective features...\n"
printf "================================================================================\n"

OUT_DIR="${EXPERIMENT_DIR}/warmup/BaB/"
mkdir -p "$OUT_DIR"
python -m experiments.tabular.bound \
  --model "${NETWORK}" \
  --effective-features ${WARMUP_SIZE} \
  --input 0 --output-feature 0 \
  --shap-variant "zero-baseline" \
  --bound-method "bab" \
  --bound-options "batch_size=$BATCH_SIZE" \
  --out "$OUT_DIR" \
  --max-iters 2

for ((i=10; i<=MAX_SIZE; i++)); do
  if [ $i -le 28 ]; then  # values above 28 cause crashes
    sleep 15s
    printf "================================================================================\n"
    printf "Running ExactSHAP with ${i} effective features...\n"
    printf "================================================================================\n"

    OUT_DIR="${EXPERIMENT_DIR}/ExactSHAP/${i}"
    mkdir -p "$OUT_DIR"
    timeout "$TIMEOUT" \
      python -m experiments.tabular.exact_shap \
        --model "${NETWORK}" \
        --effective-features $i \
        --input 0 --output-feature 0 \
        --shap-variant "zero-baseline" \
        --out "$OUT_DIR" \
        --silent
  fi

  sleep 15s
  printf "================================================================================\n"
  printf "Running Branch and Bound with ${i} effective features...\n"
  printf "================================================================================\n"

  OUT_DIR="${EXPERIMENT_DIR}/BaB/${i}"
  mkdir -p "$OUT_DIR"
  python -m experiments.tabular.bound \
    --model "${NETWORK}" \
    --effective-features $i \
    --input 0 --output-feature 0 \
    --shap-variant "zero-baseline" \
    --bound-method "bab" \
    --bound-options "batch_size=$BATCH_SIZE" \
    --out "$OUT_DIR" \
    --timeout "$TIMEOUT" --max-iters 10000 \
    --silent
done

printf "Experiment complete.\n"
