# Copyright 2025 David Boetius
from collections import defaultdict
from time import perf_counter


class TimeContextManager:
    def __init__(self, timer: "Timer", key: str):
        self.timer = timer
        self.key = key

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        runtime = perf_counter() - self.start
        self.timer._runtimes[self.key].append(runtime)
        self.timer.last = runtime


class Timer:
    """Time statements:

    Use as
    ```python
    timer = Timer()
    with timer["function_name"]:
        function()
    print(timer.runtimes)  # dict of lists of runtimes
    ```
    """

    def __init__(self):
        self._runtimes: dict[str, list[float]] = defaultdict(list)
        self.last = None

    @property
    def runtimes(self) -> dict[str, list[float]]:
        return dict(self._runtimes)

    def __getitem__(self, key: str) -> TimeContextManager:
        return TimeContextManager(self, key)
