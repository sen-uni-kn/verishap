# Tests README

## Adding a new network resource
For adding a new network resource, like `emnist_flax_conv`, as a fixture, follow
these steps:
 1. Define your fixture like `emnist_flax_conv` in a file in `tests/nets`.
 2. Import your fixture in `tests/nets/__init__.py`.
 3. Import your fixture in `tests/conftest.py`.
Now your fixture is available for use in all tests.
