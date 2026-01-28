#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
import random
from pathlib import Path

import numpy as np
import pytest

# make fixtures available to all tests
from .module_cases import (  # noqa: F401
    argument_seeds3,
    argument_seeds5,
    argument_seeds10,
)
from .nets import (  # noqa: F401
    acasxu_network,
    emnist_flax_conv,
    mnist_equinox_conv_batchnorm,
    mnist_flax_mlp,
    mnist_ibp_training_flax_conv,
    mnist_onnx_conv,
    mnist_onnx_fully_connected,
)


@pytest.fixture(autouse=True)
def seed_rngs(base_seed=0):  # reproducibility backup (seed should be reset still)
    random.seed(base_seed + 1)
    np.random.seed(base_seed + 2)


@pytest.fixture
def resource_dir():
    preferred_path = Path("resources")
    if preferred_path.exists():
        return preferred_path
    else:
        return Path("tests/resources")


def pytest_addoption(parser):
    # don't run benchmarks by default, enable with command line option
    # See:
    # https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
    parser.addoption(
        "--benchmark", action="store_true", default=False, help="Run benchmarks."
    )


def pytest_collection_modifyitems(config, items):
    # Follow-up for turning off benchmarks by default
    # https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
    if config.getoption("--benchmark"):
        # with --benchmark option, run all tests
        return
    skip_benchmark = pytest.mark.skip(
        reason="Skipping benchmarks. Run with --benchmark to execute benchmarks."
    )
    for item in items:
        # the benchmark mark comes from the pytest-benchmark plugin
        if "benchmark" in item.keywords:
            item.add_marker(skip_benchmark)

    # Custom test ordering
    def get_test_priority(item):
        """Return a tuple for sorting tests in the desired order."""
        file_path = str(item.fspath)

        # Main test directory ordering: test_core -> test_bounds -> test_verify
        if "test_core" in file_path:
            main_priority = 0
        elif "test_bounds" in file_path:
            main_priority = 1
        elif "test_verify" in file_path:
            main_priority = 2
        else:
            main_priority = 3  # Other tests run last

        # Within test_bounds, specific file ordering
        if "test_bounds" in file_path:
            if "test_ibp.py" in file_path:
                bounds_priority = 0
            elif "test_lirpa_bounds.py" in file_path:
                bounds_priority = 1
            elif "test_bwlirpa.py" in file_path:
                bounds_priority = 2
            elif "test_crown_ibp.py" in file_path:
                bounds_priority = 3
            elif "test_crown.py" in file_path:
                bounds_priority = 4
            elif "test_alpha_crown.py" in file_path:
                bounds_priority = 5
            else:
                bounds_priority = 6  # Other bounds tests
        else:
            bounds_priority = 0

        # Extract fixture-based test order from parameterized tests
        fixture_priority = 1  # Default to "advanced"
        if hasattr(item, "callspec") and item.callspec:
            # Look for 'case' parameter which contains fixture names
            if "case" in item.callspec.params:
                case_name = item.callspec.params["case"]
                # Try to get the fixture from the test module
                if hasattr(item.module, case_name):
                    fixture = getattr(item.module, case_name)
                    if hasattr(fixture, "_test_order"):
                        test_order = fixture._test_order
                        if test_order == "basic":
                            fixture_priority = 0
                        elif test_order == "advanced":
                            fixture_priority = 1
                        elif test_order == "integration":
                            fixture_priority = 2

        # Use file path and test name for stable sorting within same priority
        return (main_priority, bounds_priority, fixture_priority, file_path, item.name)

    # Sort items according to the priority function
    items.sort(key=get_test_priority)
