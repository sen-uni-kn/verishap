# SHAP-Bounds

## Installation
First, run
```bash
git submodule update --init
```
to pull the `formalax` dependency.

Next, create a virtual environment and install the Python dependencies.
Using [`uv`](https://docs.astral.sh/uv/):
```bash
uv venv
uv sync --all-extras
source .venv/bin/activate
```
Using conda:
```bash
conda create -n shap-bounds python=3.12
conda activate shap-bounds
pip install -e ".[all]"
```

## Run Experiments
For the MNIST experiment, run
```
python -m experiments.mnist --num-patches 5
```
to see all experiment options, run
```
python -m experiments.mnist -h
```

Similarly for the other experiments in the `experiments/` directory.

