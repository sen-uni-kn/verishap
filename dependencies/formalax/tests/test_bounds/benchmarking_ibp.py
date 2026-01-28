#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
import pytest

from .test_ibp import TestIBP, ibp_test_once_cases

# Initialize Test class
IBP_test_instance = TestIBP()


@pytest.mark.benchmark(
    group="Bound Propagation Benchmarks",
    max_time=15,
    min_rounds=50,
    disable_gc=False,
    warmup=True,
    warmup_iterations=1,
)
@pytest.mark.parametrize("case", ibp_test_once_cases)
def test_bounds_once_benchmark(case, request, benchmark):
    benchmark(IBP_test_instance.test_bounds_once, case, request)


@pytest.mark.parametrize("case", ibp_test_once_cases)
def test_bounds_repeat5_benchmark(case, argument_seeds5, request, benchmark):
    benchmark(IBP_test_instance.test_bounds_repeat5, case, argument_seeds5, request)
