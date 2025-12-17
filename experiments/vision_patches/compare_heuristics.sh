#!/bin/bash

HERE="$(dirname "$0")"
NUM_REPEATS=5
NUM_PATCHES="6"

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi

for i in {1..5}  # repeat runtime measurements five times
do
  for compute_bounds in "ibp" "crown_ibp" "crown" "alpha-crown"
  do
    for select_strategy in "max-diam" "min-diam" "fifo" "lifo"
    do
      for split_strategy in "longest-edge" "smears" \
        "lirpa-weights" "strong-branching-better" "strong-branching-worse" \
        "smart-branching-ibp-better" "smart-branching-ibp-worse"
      do
        OUT_DIR="$HERE/output/compare_heuristics/patches_${NUM_PATCHES}_${TIMESTAMP}/repeatition_${i}/${compute_bounds}_${select_strategy}_${split_strategy}"
        mkdir -p "$OUT_DIR"
        python -m experiments.mnist.bound \
        --num-patches "$NUM_PATCHES" \
        --bound-options "compute_bounds=$compute_bounds,select_strategy=$select_strategy,split_strategy=$split_strategy" \
        --out "$OUT_DIR" --silent --max-iters 100
      done
    done
  done
done
