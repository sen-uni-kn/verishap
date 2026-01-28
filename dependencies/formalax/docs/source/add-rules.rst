Adding custom Relaxation Rules
==============================

Overview
--------

Formalax natively supports a large number of common primitive operations.


This guide explains how to register a new **backwards LiRPA relaxation rule** in `formalax`.


A *backwards LiRPA rule* describes how to propagate interval bounds (as linear relaxations) **backwards** through a `JAX primitive <https://docs.jax.dev/en/latest/jax.lax.html>`_.

Each rule handles a single primitive (like `lax.add`, `lax.exp`, `lax.max`, etc.) and must be registered explicitly.

.. contents::
   :local:
   :depth: 2


Step 1: Define the Rule
------------------------

Before writing a custom rule from scratch, **check whether your primitive fits an existing reusable pattern**. This simplifies implementation and ensures consistent handling of shapes, broadcasting, and batch axes.

Use an Existing Wrapper (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``formalax`` library provides several helper functions that cover common categories of operations:

- **Unary operations**  
  Use: ``transpose_as_backwards_lirpa_rule_unary``  
  For example: ``neg``, ``exp``, ``log``, etc.

- **Bilinear operations** (two inputs, both linear)  
  Use: ``transpose_as_backwards_lirpa_rule_bilinear``  
  For example: ``lax.mul``, ``lax.dot``

- **Left-linear binary operations**  
  Use: ``transpose_as_backwards_lirpa_rule_binary_left_only``  
  For example: scalar multiplication (linear in one input)

- **Constant-valued or comparison operations**  
  Use: ``constant_bounds_backwards_lirpa_rule``  
  For example: ``lax.ge``, ``lax.eq``, etc.

If your primitive fits one of these categories, register it directly. Otherwise, or if you prefer custom control, proceed to define a rule manually.

Custom Rule Signature
^^^^^^^^^^^^^^^^^^^^^^^^

Define a Python function matching the following signature:

.. code-block:: python

    def my_custom_backwards_lirpa_rule(
        out_weights: Sequence[LiRPAWeights[jax.Array | Zero]],
        in_bounds: Sequence[Bounds | jax.Array],
        in_batch_axes: Sequence[tuple[int, ...]],
        out_batch_axes: Sequence[tuple[int, ...]],
        batch_axis_mappings: Sequence[Sequence[BatchAxisMapping]],
        backwards_lirpa: Callable,
        **kwargs,
    ) -> LiRPABounds[jax.Array | Zero]:
        # Your implementation here
        ...

This function computes the **backward propagation of LiRPA weights**, based on input bounds and batch axis mappings.


Step 2: Register the Rule
--------------------------

Register your function using:

.. code-block:: python

    from formalax.bounds._bwlirpa import register_backwards_lirpa_rule
    from jax import lax

    register_backwards_lirpa_rule(lax.my_primitive_p, my_custom_backwards_lirpa_rule)

If your rule needs to be partially applied (e.g. addition with a constant), use `functools.partial`:

.. code-block:: python

    from functools import partial

    register_backwards_lirpa_rule(
        lax.add_p,
        partial(add_with_broadcasting_backwards_lirpa_rule, alpha=1.0)
    )

Example: `lax.ge` Rule
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from jax import lax
    from functools import partial
    from formalax.bounds._bwlirpa import register_backwards_lirpa_rule
    from formalax.bounds._ibp import ibp_rule_compare_greater
    from formalax.bounds._bwlirpa_utils import constant_bounds_backwards_lirpa_rule

    register_backwards_lirpa_rule(
        lax.ge_p,
        partial(
            constant_bounds_backwards_lirpa_rule,
            partial(ibp_rule_compare_greater, lax.ge)
        )
    )

This uses an IBP-based comparator to handle boolean outputs for `x >= y`.

Step 3: Add a Test Case
------------------------

To test your rule:

1. Add a fixture in `tests/bounds/module_cases.py`:

.. code-block:: python

    @pytest.fixture
    def my_primitive_case() -> ModuleTestCaseFactory:
        return ModuleTestCaseFactory().module(lax.my_primitive).with_random_arguments([(4, 4)])

2. Add the case name to `test_once_cases` or `test_multiple_cases`.

3. Run tests:

.. code-block:: shell

    pytest tests/bounds

Utilities
---------

These are useful helpers while defining rules:

- ``Zero()``: Represents zero weights or bias.
- ``_unbroadcast(...)``: Handles shape alignment when broadcasting.
- ``LiRPAWeights``, ``LiRPABounds``: Main data types used in LiRPA rules.
- ``BatchAxisMapping``: Tracks how batch axes are shared across the computation graph.

Where to Register
-----------------

- Backwards LiRPA rules are registered in: ``formalax.bounds._bwlirpa``
- CROWN-specific rules live in: ``formalax.bounds._bwcrown``
