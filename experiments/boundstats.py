# Copyright 2025 David Boetius
from time import perf_counter

import jax.numpy as jnp


class BoundStats:
    def __init__(self):
        self.time_stats = {}
        self.iter_stats = {}
        self.start = None

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @property
    def runtime(self) -> float:
        assert self.start is not None
        return perf_counter() - self.start

    def record(self, i, lb, ub) -> float:
        runtime = self.runtime

        lb, ub = lb.flatten(), ub.flatten()
        lb_vs_each_ub = jnp.reshape(lb, (-1, 1)) >= jnp.reshape(ub, (1, -1))
        ub_vs_each_lb = jnp.reshape(ub, (-1, 1)) <= jnp.reshape(lb, (1, -1))

        separated = lb_vs_each_ub.any(axis=-1)
        largest = lb_vs_each_ub.all(axis=-1)
        smallest = ub_vs_each_lb.all(axis=-1)

        if separated.any() and "some_separated" not in self.time_stats:
            self.time_stats["some_separated"] = runtime
            self.iter_stats["some_separated"] = i

        if largest.any() and "largest" not in self.time_stats:
            self.time_stats["largest_shap"] = runtime
            self.iter_stats["largest_shap"] = i

        if smallest.any() and "smallest" not in self.time_stats:
            self.time_stats["smallest_shap"] = runtime
            self.iter_stats["smallest_shap"] = i

        max_range = jnp.max(ub - lb).item()
        if max_range < 1.0 and "max_range_less_than_1" not in self.time_stats:
            self.time_stats["max_range_less_than_1"] = runtime
            self.iter_stats["max_range_less_than_1"] = i
        if max_range < 0.01 and "max_range_less_than_0.01" not in self.time_stats:
            self.time_stats["max_range_less_than_0.01"] = runtime
            self.iter_stats["max_range_less_than_0.01"] = i
        if max_range < 0.001 and "max_range_less_than_0.001" not in self.time_stats:
            self.time_stats["max_range_less_than_0.001"] = runtime
            self.iter_stats["max_range_less_than_0.001"] = i
        if max_range < 0.0001 and "max_range_less_than_0.0001" not in self.time_stats:
            self.time_stats["max_range_less_than_0.0001"] = runtime
            self.iter_stats["max_range_less_than_0.0001"] = i
        if max_range < 0.00001 and "max_range_less_than_0.00001" not in self.time_stats:
            self.time_stats["max_range_less_than_0.00001"] = runtime
            self.iter_stats["max_range_less_than_0.00001"] = i
        if max_range < 0.000001 and "max_range_less_than_0.000001" not in self.time_stats:
            self.time_stats["max_range_less_than_0.000001"] = runtime
            self.iter_stats["max_range_less_than_0.000001"] = i

        return runtime
