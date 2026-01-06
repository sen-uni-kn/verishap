#!/bin/bash
HERE="$(dirname "$0")"

NUM_PATCHES="${1-7}"
SHAP_VARIANT="${2-zero-baseline}"
NETWORK="${3-experiments/resources/mnist-cnn.eqx}"
network_filename=$(basename "$NETWORK")
network_name="${network_filename%.*}"

BATCH_SIZE=4096

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi
EXPERIMENT_DIR="$HERE/output/bab_vs_sampling/${network_name}_${SHAP_VARIANT}_${TIMESTAMP}"

printf "================================================================================\n"
printf "Running Warmup for Branch and Bound\n"
printf "================================================================================\n"

OUT_DIR="${EXPERIMENT_DIR}/warmup/BaB/"
mkdir -p "$OUT_DIR"
python -m experiments.vision_patches.bound \
  --model "${NETWORK}" \
  --num-patches "${NUM_PATCHES}" \
  --input 0 --output-feature 0 \
  --shap-variant "${SHAP_VARIANT}" \
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
  --shap-variant "${SHAP_VARIANT}" \
  --bound-method "bab" \
  --bound-options "batch_size=$BATCH_SIZE" \
  --out "$OUT_DIR" \
  --silent

sleep 15s
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
    --shap-variant "${SHAP_VARIANT}" \
    --estimator "${estimator}" \
    --out "$OUT_DIR" \
    --num-samples "200" \
    --seed 0
done

for estimator in "PermutationSHAP" "LeverageSHAP"; do
    for seed in $(seq 0 100); do
        sleep 5s
        printf "================================================================================\n"
        printf "Running ${estimator} with seed ${seed}\n"
        printf "================================================================================\n"
        
        # Sampling in individual calls to have the time required for jitting in each call
        # for a fair comparison with BaB, which also includes the jitting overhead.
        OUT_DIR="${EXPERIMENT_DIR}/${estimator}/seed_${seed}/1000"
        mkdir -p "$OUT_DIR"
        python -m experiments.vision_patches.estimate \
          --model "${NETWORK}" \
          --num-patches "${NUM_PATCHES}" \
          --input 0 --output-feature 0 \
          --shap-variant "${SHAP_VARIANT}" \
          --estimator "${estimator}" \
          --out "$OUT_DIR" \
          --num-samples "1000" \
          --seed "${seed}" \
          --silent

        for num_samples in $(seq 10000 10000 160000); do
          OUT_DIR="${EXPERIMENT_DIR}/${estimator}/seed_${seed}/${num_samples}"
          mkdir -p "$OUT_DIR"
          python -m experiments.vision_patches.estimate \
            --model "${NETWORK}" \
            --num-patches "${NUM_PATCHES}" \
            --input 0 --output-feature 0 \
            --shap-variant "${SHAP_VARIANT}" \
            --estimator "${estimator}" \
            --out "$OUT_DIR" \
            --num-samples "${num_samples}" \
            --seed "${seed}" \
            --silent
        done
    done
done
