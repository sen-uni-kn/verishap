# VeriSHAP

Certified bounds on SHAP values for neural networks using branch-and-bound with linear relaxation-based bound propagation.

> **Abstract**: Shapley additive explanations (SHAP) are widely recognised as computationally intractable for neural networks, due to their exponential worst-case complexity in the number of input features. In this work, we take a first step toward scaling exact SHAP computation to larger input spaces by introducing an algorithm that leverages recent advances in neural network verification to compute arbitrarily tight exact lower and upper bounds on SHAP values for neural networks, and ultimately recover the exact SHAP values. We demonstrate that our approach scales to substantially larger input spaces than state-of-the-art exact methods. This provides an important first step toward exact SHAP computation and establishes a principled cornerstone for evaluating statistical approximation methods on higher-dimensional input spaces.

## Quick Example

```python
import jax.numpy as jnp
from jax import nn
from shap_bounds import multi_shap_bab, baseline_value

# Define or load your model in jax
def model(x):
    return nn.relu(x @ jnp.array([[1.0], [2.0], [3.0]]))

# Create a value function for SHAP computation
sample = jnp.array([1.0, 2.0, 3.0])
baseline = jnp.zeros(3)
value_fn = baseline_value(model, sample, baseline, output=0)

# Compute SHAP bounds iteratively
for bounds in multi_shap_bab(value_fn, base_mask=jnp.zeros(3)):
    print(f"SHAP bounds: {bounds}")
    # bounds converge to exact SHAP values over iterations
```

## Installation

Clone the repository and initialize the submodules:
```bash
git submodule update --init
```

Install dependencies using [uv](https://docs.astral.sh/uv/):
```bash
uv venv && uv sync --all-extras
source .venv/bin/activate
```

or using conda/pip:
```bash
conda create -n shap-bounds python=3.12
conda activate shap-bounds
pip install -e ".[all]"
```

## Running Experiments

All experiments are Python modules or Bash scripts under `experiments/`.

### MNIST

Compute SHAP bounds for image patches:

```bash
python -m experiments.vision_patches.bound \
  --model experiments/resources/mnist-cnn.eqx \
  --num-patches 7 \
  --input 43 --output-feature 4 \
  --shap-variant zero-baseline \
  --overlay-logger
```

Run with `-h` to see all available options:
```bash
python -m experiments.vision_patches.bound -h
```

### Tabular Data

Compute SHAP bounds for a tabular dataset:

```bash
python -m experiments.tabular.bound \
  --model experiments/resources/mushroom-mlp-8x1.eqx \
  --shap-variant marginal
```

### Batch Experiments

Run full experiment suites using the provided shell scripts:

| Script | Description |
|--------|-------------|
| `experiments/tabular/compare_to_exactshap.sh` | Compare VeriSHAP against ExactSHAP on tabular datasets |
| `experiments/tabular/compare_estimators.sh` | Benchmark SHAP estimators (KernelSHAP, PermutationSHAP, etc.) |
| `experiments/tabular/bab_vs_estimators.sh` | Compare branch-and-bound bounds against estimator convergence |
| `experiments/vision_patches/compute_bounds.sh` | Run MNIST patch experiments with varying patch counts |

Example:

```bash
# Compare to ExactSHAP with marginal SHAP variant
./experiments/tabular/compare_to_exactshap.sh marginal

# Run MNIST experiments (sample 43, output class 4, 5-7 patches)
./experiments/vision_patches/compute_bounds.sh 43 4 mean-baseline 4096
```
