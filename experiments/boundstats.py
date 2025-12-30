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

        print("Max range:", max_range)

        if max_range < 1.0 and "ran_lt_1e0" not in self.time_stats:
            self.time_stats["ran_lt_1e0"] = runtime
            self.iter_stats["ran_lt_1e0"] = i
        if max_range < 0.1 and "ran_lt_1e-1" not in self.time_stats:
            self.time_stats["ran_lt_1e-1"] = runtime
            self.iter_stats["ran_lt_1e-1"] = i
        if max_range < 0.01 and "ran_lt_1e-2" not in self.time_stats:
            self.time_stats["ran_lt_1e-2"] = runtime
            self.iter_stats["ran_lt_1e-2"] = i
        if max_range < 0.001 and "ran_lt_1e-3" not in self.time_stats:
            self.time_stats["ran_lt_1e-3"] = runtime
            self.iter_stats["ran_lt_1e-3"] = i

        return runtime
