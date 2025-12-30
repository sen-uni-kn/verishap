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

        if (lb >= 0).any() and "some_pos" not in self.time_stats:
            self.time_stats["some_pos"] = runtime
            self.iter_stats["some_pos"] = i
        if (ub <= 0).any() and "some_neg" not in self.time_stats:
            self.time_stats["some_neg"] = runtime
            self.iter_stats["some_neg"] = i

        max_range = jnp.max(ub - lb).item()
        if max_range < 1.0 and "max_ran_lt_1e0" not in self.time_stats:
            self.time_stats["max_ran_lt_1e0"] = runtime
            self.iter_stats["max_ran_lt_1e0"] = i
        if max_range < 0.1 and "max_ran_lt_1e-1" not in self.time_stats:
            self.time_stats["max_ran_lt_1e-1"] = runtime
            self.iter_stats["max_ran_lt_1e-1"] = i
        if max_range < 0.01 and "max_ran_lt_1e-2" not in self.time_stats:
            self.time_stats["max_ran_lt_1e-2"] = runtime
            self.iter_stats["max_ran_lt_1e-2"] = i
        if max_range < 0.001 and "max_ran_lt_1e-3" not in self.time_stats:
            self.time_stats["max_ran_lt_1e-3"] = runtime
            self.iter_stats["max_ran_lt_1e-3"] = i

        min_range = jnp.max(ub - lb).item()
        if min_range < 1.0 and "min_ran_lt_1e0" not in self.time_stats:
            self.time_stats["min_ran_lt_1e0"] = runtime
            self.iter_stats["min_ran_lt_1e0"] = i
        if min_range < 0.1 and "min_ran_lt_1e-1" not in self.time_stats:
            self.time_stats["min_ran_lt_1e-1"] = runtime
            self.iter_stats["min_ran_lt_1e-1"] = i
        if min_range < 0.01 and "min_ran_lt_1e-2" not in self.time_stats:
            self.time_stats["min_ran_lt_1e-2"] = runtime
            self.iter_stats["min_ran_lt_1e-2"] = i
        if min_range < 0.001 and "min_ran_lt_1e-3" not in self.time_stats:
            self.time_stats["min_ran_lt_1e-3"] = runtime
            self.iter_stats["min_ran_lt_1e-3"] = i

        ref_val = jnp.mean(jnp.abs(ub + lb) / 2)
        norm_max_range = max_range / ref_val
        if norm_max_range < 0.1 and "norm_max_ran_10percent" not in self.time_stats:
            self.time_stats["norm_max_ran_10percent"] = runtime
            self.iter_stats["norm_max_ran_10percent"] = i
        if norm_max_range < 0.01 and "norm_max_ran_1percent" not in self.time_stats:
            self.time_stats["norm_max_ran_1percent"] = runtime
            self.iter_stats["norm_max_ran_1percent"] = i
        if norm_max_range < 0.001 and "norm_max_ran_1e-1percent" not in self.time_stats:
            self.time_stats["norm_max_ran_1e-1percent"] = runtime
            self.iter_stats["norm_max_ran_1e-1percent"] = i
        if norm_max_range < 0.0001 and "norm_max_ran_1e-2percent" not in self.time_stats:
            self.time_stats["norm_max_ran_1e-2percent"] = runtime
            self.iter_stats["norm_max_ran_1e-2percent"] = i

        norm_min_range = min_range / ref_val
        if norm_min_range < 0.1 and "norm_min_ran_10percent" not in self.time_stats:
            self.time_stats["norm_min_ran_10percent"] = runtime
            self.iter_stats["norm_min_ran_10percent"] = i
        if norm_min_range < 0.01 and "norm_min_ran_1percent" not in self.time_stats:
            self.time_stats["norm_min_ran_1percent"] = runtime
            self.iter_stats["norm_min_ran_1percent"] = i
        if norm_min_range < 0.001 and "norm_min_ran_1e-1percent" not in self.time_stats:
            self.time_stats["norm_min_ran_1e-1percent"] = runtime
            self.iter_stats["norm_min_ran_1e-1percent"] = i
        if norm_min_range < 0.0001 and "norm_min_ran_1e-2percent" not in self.time_stats:
            self.time_stats["norm_min_ran_1e-2percent"] = runtime
            self.iter_stats["norm_min_ran_1e-2percent"] = i

        return runtime
