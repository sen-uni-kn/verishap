# Formalax

Deep learning Formal Methods Library using [JAX](https://github.com/google/jax?tab=readme-ov-file#transformations).

## Installation
Clone the repository and run
```bash
pip install .
```

## Development
This project uses 
- [uv](https://docs.astral.sh/uv/concepts/projects/build/) for building,
- [pytest](https://docs.pytest.org/en/8.2.x/) and [Nox](https://nox.thea.codes/en/stable/index.html) for testing,
- [ruff](https://docs.astral.sh/ruff/) for linting and code formatting,
- the [Google docstring format](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings). 
  A comprehensive example can be found [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
- [git lfs](https://git-lfs.com/) for managing binary resource files, such as trained
  network weights.

To install `formalax` for development, run
```bash
uv venv
uv sync --all-extras
# source .venv/bin/activate or
# uv run pytest
```
in the root folder of this repository.
Alternatively, you can install `formalax` in an existing environment by running 
```bash
pip install --editable .[all]
```
in the root folder of this project.

To download the binary resources using in the tests (for example, trained network weights), you need to [install git lfs](https://github.com/git-lfs/git-lfs/wiki/Installation).
After you installed git lfs, run
```bash
git lfs install
git lfs pull
```
to download the binary resources used in the tests.

### Unit & Integration Tests
To run the tests, run
```bash
pytest
```
or run
```bash
nox
```
to run the tests in a fresh environment.

### Test Coverage
To generate a test coverage report, run
```bash
pytest --cov
```
Add the `--cov-report html` option to generate an HTML report (saved in `coverage-html`).

### Benchmarking 
The tests directory also contains several benchmarks.
To perform, for example, the IBP benchmarks, run
```bash
pytest tests/bounds/benchmarking_ibp.py --benchmark
```
For selecting individual benchmarks, you can use the same options as for selecting
pytest tests.
For example, to only run the `emnist` benchmarks, run
```bash
pytest tests/bounds/benchmarking_ibp.py --benchmark -k emnist
```

### Helpful Resources on JAX
Understanding the core of this library requires understanding the JAX core. 
The following are a few helpful resources for learning about the JAX core.
- https://jax.readthedocs.io/en/latest/autodidax.html
- Tracers:
  - https://github.com/google/jax/discussions/18270
- `jax.extend.linear_util` (abbreviated `lu`):
  - Refers to a *linear type system* and not *linear functions* 
    in the *linear algebra* sense.
  - https://github.com/google/jax/blob/main/jax/_src/linear_util.py
  - https://en.wikipedia.org/wiki/Substructural_type_system

## License
This project is licensed at the terms of the MIT license.

### Authors
- David Boetius (project lead, main contact)
- Robin Baran Aytac
- Leonard Schäffter
