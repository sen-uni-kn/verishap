#!/bin/bash
HERE="$(dirname "$0")"

BATCH_SIZE=4096
NETWORK="${1-experiments/resources/mnist-cnn.eqx}"
NUM_PATCHES="${2-7}"
network_filename=$(basename "$NETWORK")
network_name="${network_filename%.*}"

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi
EXPERIMENT_DIR="$HERE/output/bab_vs_sampling/${network_name}_${TIMESTAMP}"


printf "================================================================================\n"
printf "Running Warmup for Sampling\n"
printf "================================================================================\n"

for estimator in "PermutationSHAP" "LeverageSHAP"; do
    OUT_DIR="${EXPERIMENT_DIR}/warmup/${estimator}/"
    mkdir -p "$OUT_DIR"
    python -m experiments.vision_patches.estimate \
    --model "${NETWORK}" \
    --num-patches "${NUM_PATCHES}" \
    --input 0 --output-feature 0 \
    --shap-variant "zero-baseline" \
    --estimator "${estimator}" \
    --out "$OUT_DIR" \
    --num-samples "200" \
    --seed 0
done

for estimator in "PermutationSHAP" "LeverageSHAP"; do
    for seed in 0 1 2 3 4 5 6 7 8 9; do
        sleep 15s
        printf "================================================================================\n"
        printf "Running ${estimator} with seed ${seed}\n"
        printf "================================================================================\n"
        
        OUT_DIR="${EXPERIMENT_DIR}/${estimator}/seed_${seed}/"
        mkdir -p "$OUT_DIR"
        # Not silencing this call, since there is no logging during sampling anyways.
        python -m experiments.vision_patches.estimate \
        --model "${NETWORK}" \
        --num-patches "${NUM_PATCHES}" \
        --input 0 --output-feature 0 \
        --shap-variant "zero-baseline" \
        --estimator "${estimator}" \
        --out "$OUT_DIR" \
        --num-samples "200,1000,10000,100000" \
        --seed "${seed}"
        # --num-samples "200,1000,10000,100000,1000000,10000000" \
    done
done

sleep 15s
printf "================================================================================\n"
printf "Running Warmup for Branch and Bound\n"
printf "================================================================================\n"

OUT_DIR="${EXPERIMENT_DIR}/warmup/BaB/"
mkdir -p "$OUT_DIR"
python -m experiments.vision_patches.bound \
  --model "${NETWORK}" \
  --num-patches "${NUM_PATCHES}" \
  --input 0 --output-feature 0 \
  --shap-variant "zero-baseline" \
  --bound-method "bab" \
  --bound-options "batch_size=$BATCH_SIZE" \
  --out "$OUT_DIR" \
  --max-iters 2

sleep 15s
printf "================================================================================\n"
printf "Running Branch and Bound\n"
printf "================================================================================\n"

OUT_DIR="${EXPERIMENT_DIR}/BaB/"
mkdir -p "$OUT_DIR"
python -m experiments.vision_patches.bound \
  --model "${NETWORK}" \
  --num-patches "${NUM_PATCHES}" \
  --input 0 --output-feature 0 \
  --shap-variant "zero-baseline" \
  --bound-method "bab" \
  --bound-options "batch_size=$BATCH_SIZE" \
  --out "$OUT_DIR" \
  --silent
