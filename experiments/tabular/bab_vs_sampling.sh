#!/bin/bash
HERE="$(dirname "$0")"
SHAP_VARIANT="${1-marginal}"

TIMEOUT=150

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

for estimator in "KernelSHAP" "PermutationSHAP" "LeverageSHAP"; do
  for network in "${NETWORKS[@]}"; do
    network_filename=$(basename "$network")
    network_name="${network_filename%.*}"

    sleep 15s
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

    for seed in $(seq 0 25); do
      sleep 5s
      printf "================================================================================\n"
      printf "Running ${estimator} on ${network} with seed ${seed}\n"
      printf "================================================================================\n"
      
      # Sampling in individual calls to have the time required for jitting in each call
      # for a fair comparison with BaB, which also includes the jitting overhead.
      for num_samples in $(seq 1000 1000 9000) $(seq 10000 10000 90000) $(seq 100000 100000 900000) $(seq 1000000 1000000 9000000) $(seq 10000000 10000000 90000000) $(seq 100000000 100000000 1000000000);
      do
        OUT_DIR="${EXPERIMENT_DIR}/${estimator}/${network_name}/seed_${seed}/${num_samples}"
        mkdir -p "$OUT_DIR"
        timeout "$TIMEOUT" \
          python -m experiments.tabular.estimate \
      	    --model "experiments/resources/${network}" \
            --input 0 --output-feature 0 \
            --shap-variant "${SHAP_VARIANT}" \
            --estimator "${estimator}" \
            --out "$OUT_DIR" \
            --num-samples "${num_samples}" \
            --seed "${seed}" \
            --silent
        
        if [ $? == 124 ]; then
          echo "Timeout reached for ${estimator} on ${network} with seed ${seed} and ${num_samples} samples"
          break
        fi
        sleep 5s
      done
    done
  done
done

printf "Experiment complete.\n"
