# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository computes certified bounds on SHAP (SHapley Additive exPlanations) values using branch-and-bound algorithms with linear relaxation-based propagation (LIRPA). The core implementation uses JAX for automatic differentiation and JIT compilation.

## Development Environment Setup

### Installation
```bash
# Initialize git submodules (formalax dependency)
git submodule update --init

# Using uv (preferred)
uv venv
uv sync --all-extras
source .venv/bin/activate

# Using conda (alternative)
conda create -n shap-bounds python=3.12
conda activate shap-bounds
pip install -e ".[all]"
```

### Code Quality
```bash
# Format code
uv run ruff format src experiments tests

# Lint code
uv run ruff check .

# Run tests
uv run pytest
# Or: python -m pytest tests
```

## Running Experiments

All experiments are Python modules under `experiments/`. Run them with:
```bash
# General pattern
python -m experiments.<module_name>.bound [options]
python -m experiments.<module_name>.estimate [options]

# Examples
python -m experiments.vision_patches.bound --num-patches 5
python -m experiments.simple --mlp --input-dim 10 --feature 0
python -m experiments.tabular.bound --dataset adult --model xgboost

# Get help for any experiment
python -m experiments.<module_name>.bound -h
python -m experiments.<module_name>.estimate -h
```

### Experiment Categories
- **simple/**: Basic toy models (MLP, linear, sum) for testing
- **tabular/**: Tabular data experiments with various bound comparison scripts
- **vision_patches/**: Vision experiments with patch-based explanations (MNIST, etc.)

Many experiment directories contain bash scripts (`.sh`) that run batches of experiments with different configurations.

## Core Architecture

### Main Algorithms (src/shap_bounds/)

The library provides two main branch-and-bound algorithms:

1. **`shap_bab`** (shap_bab.py): Single-feature SHAP bounding
   - Computes bounds for one feature at a time
   - Uses `BranchData` to track coalition bounds, contribution bounds, depth, and Shapley bounds
   - Iteratively splits coalition space and refines bounds

2. **`multi_shap_bab`** (multi_shap_bab.py): Multi-feature SHAP bounding
   - Computes bounds for multiple features simultaneously
   - More efficient when computing SHAP values for all features
   - Shares computation across features
   - Uses similar branching strategy but different bound computation

### Key Components

**Value Functions (value_functions.py)**: Construct value functions for SHAP computation
- `baseline_value`: Standard SHAP with fixed baseline (e.g., zero baseline)
- `marginal_value`: Marginal SHAP averaging over background data
- `superfeature_*`: Variants for grouped features (superpixels, patches)

**Branch Management**: Priority-based and FIFO/LIFO branch stores
- `PriorityBranchStore` (priority_branch_store.py): Max-priority queue for branch selection
- `BranchQueue` (branch_queue.py): FIFO strategy
- `BranchStack` (branch_stack.py): LIFO strategy
- All stores are batched for efficient JAX operations

**Bound Computation**: Uses the `formalax` library (git submodule)
- IBP (Interval Bound Propagation)
- CROWN (linear relaxation)
- CROWN-IBP (combined approach)
- Alpha-CROWN (optimized linear relaxation)

### Branch-and-Bound Strategies

**Selection Strategies** (`select_strategy` parameter):
- `max-diam`: Select branches with largest Shapley value diameter (default)
- `min-diam`: Select branches with smallest diameter
- `fifo`: First-in-first-out
- `lifo`: Last-in-first-out (depth-first)

**Split Strategies** (`split_strategy` parameter):
- `longest-edge`: Split on dimension with largest coalition bound width
- `smears`: Split based on gradient magnitudes
- `lirpa-weights`: Use LIRPA weight magnitudes to guide splitting
- `strong-branching-better/worse`: Evaluate all splits with full bound propagation
- `smart-branching-ibp-better/worse`: Like strong branching but with fast IBP

### Coalition Representation

Coalitions are represented as continuous bounds on boolean masks:
- A `Box[lb, ub]` where `lb` and `ub` are arrays with shape matching the input features
- `lb[i] = ub[i] = 0` means feature `i` is excluded from all coalitions in the branch
- `lb[i] = ub[i] = 1` means feature `i` is included in all coalitions
- `lb[i] = 0, ub[i] = 1` means some coalitions include feature `i`, others don't

This representation allows LIRPA-based bound propagation through neural networks.

## Experiment Utilities

**datasets.py**: Dataset loading functions
- Supports SHAP library datasets, UCI ML repo, Kaggle datasets
- Custom synthetic datasets (corrgroups, independentlinear)
- Image datasets with PyTorch DataLoader interface

**models.py**: Model definitions
- Neural network models using Equinox (JAX)
- PyTorch to JAX model conversion utilities
- Pre-trained model loading

**shaplib.py**: SHAP estimation baselines
- Wrappers for various SHAP estimators (KernelSHAP, TreeSHAP, etc.)
- Used for comparison with certified bounds

**runstats.py**: Experiment statistics and resource monitoring
- CPU/GPU utilization tracking
- Memory usage monitoring
- Runtime statistics

## Dependencies

**formalax** (git submodule): Core library for formal verification and bound propagation
- Provides IBP, CROWN, Alpha-CROWN implementations
- Box abstraction for interval bounds
- Must be initialized with `git submodule update --init`

## Testing

Tests are minimal currently. Add new tests to `tests/`:
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_priority_branch_store.py
```

## Key Configuration Options

When using `shap_bab` or `multi_shap_bab`:

- `compute_bounds`: Main bound propagation method (`"crown_ibp"`, `"ibp"`, `"crown"`, `"alpha-crown"`)
- `fast_compute_bounds`: Faster method for heuristics (usually `"ibp"`)
- `batch_size`: Number of branches processed together (default: 1024)
- `jit`: Whether to JIT compile operations (default: True)
- `log`: Enable logging (boolean or Logger instance)

## Notes for Development

- The codebase uses JAX's functional programming style with immutable data structures
- All main data structures are registered as JAX PyTrees for JIT compatibility
- Use `jax.vmap` for vectorization instead of explicit loops
- Branch stores use batched operations for GPU efficiency
- Type hints use `jaxtyping` for array shape specifications (e.g., `Real[Array, " b *shape"]`)
- Ruff configuration in pyproject.toml ignores F722 (jaxtyping syntax)
