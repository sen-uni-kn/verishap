Adding Custom IBP Rules
=======================

Overview
--------

`formalax` natively supports a wide range of common JAX primitives.  
This guide explains how to register a new **Interval Bound Propagation (IBP) rule** in `formalax`.

An *IBP rule* describes how to propagate lower and upper bounds through a JAX primitive.  
Each rule handles a single primitive (e.g., `lax.add`, `lax.exp`, `lax.max`) and must be registered explicitly.

.. contents::
   :local:
   :depth: 2

Categories of IBP Rules
-----------------------

IBP rules generally fall into one of the following categories:

- **Monotonic functions**: Functions that are non-decreasing or non-increasing.  
  Examples: `lax.exp`, `lax.neg`, `lax.max`, `lax.min`.

- **Linear transformations**: Broadcasting, reshaping, padding, convolutions, and matrix multiplication.  
  These are linear with non-negative weights.  
  Examples: `lax.reshape`, `lax.conv_general_dilated`, `lax.dot_general`.

- **Convex functions**: Strongly convex or concave functions.  
  Examples: `lax.abs`, `lax.square`, `lax.cosh`.

- **Reduction operations**: Operations that aggregate over axes.  
  Examples: `lax.reduce_sum`, `lax.reduce_min`, `lax.reduce_max`.

---

Step 1: Define the Rule
------------------------

Before creating a custom IBP rule from scratch, check whether your primitive fits an existing pattern.  
`formalax` provides helper functions for common cases:

- **Monotonic Non-Decreasing**  
  Use: `ibp_rule_monotonic_non_decreasing`  
  Example primitives: `lax.add`, `lax.exp`, `lax.tanh`

- **Monotonic Non-Increasing**  
  Use: `ibp_rule_monotonic_non_increasing`  
  Example primitives: `lax.neg`, `lax.rsqrt`

- **Linear Mappings**  
  Use: `ibp_rule_linear`  
  Example primitives: `lax.conv_general_dilated`, `lax.dot_general`

- **Reduction Operations**  
  Use: `ibp_rule_reduce_ops`  
  Example primitives: `lax.reduce_sum`, `lax.reduce_min`, `lax.reduce_max`

- **Strongly Convex Functions**  
  Use: `ibp_rule_strongly_convex`  
  Example primitives: `lax.abs`, `lax.square`, `lax.cosh`

**Custom Rule Signature**

If your primitive does not fit these categories, define a rule with the following signature:

.. code-block:: python

    def my_custom_ibp_rule(x: Bounds | jax.Array, **kwargs) -> Bounds:
        """
        Compute IBP bounds for the primitive.
        Parameters
        ----------
        x : Bounds or concrete array
        kwargs : additional primitive-specific arguments
        Returns
        -------
        Bounds
        """
        # Implement bound computation
        ...

---

Step 2: Register the Rule
--------------------------

Register your IBP rule using:

.. code-block:: python

    from formalax.bounds._ibp import register_ibp_rule
    from jax import lax

    register_ibp_rule(lax.my_primitive_p, my_custom_ibp_rule)

If your rule requires partial application (e.g., addition with a constant), use `functools.partial`:

.. code-block:: python

    from functools import partial

    register_ibp_rule(
        lax.add_p,
        partial(ibp_rule_monotonic_non_decreasing, alpha=1.0)
    )

---

Step 3: Examples
----------------

**Example 1: Custom Subtraction Rule**

.. code-block:: python

    from formalax.bounds._ibp import register_ibp_rule, Box
    from jax import lax

    def _sub_ibp_rule(x, y):
        (x_lb, x_ub), (y_lb, y_ub) = (b.concrete for b in all_as_bounds(x, y))
        return Box(lax.sub(x_lb, y_ub), lax.sub(x_ub, y_lb))

    register_ibp_rule(lax.sub_p, _sub_ibp_rule)

**Example 2: Monotonic Non-Decreasing Rule**

.. code-block:: python

    register_ibp_rule(
        lax.exp_p,
        partial(ibp_rule_monotonic_non_decreasing, lax.exp)
    )

---

Step 4: Testing Your Rule
-------------------------

1. Add a test fixture in `tests/bounds/module_cases.py`:

.. code-block:: python

    @pytest.fixture
    def my_primitive_case():
        return ModuleTestCaseFactory().module(lax.my_primitive).with_random_arguments([(4, 4)])

2. Add the case to `test_once_cases` or `test_multiple_cases`.

3. Run tests:

.. code-block:: bash

    pytest tests/bounds

---

Utilities
---------

- `Bounds`, `Box`: Main types representing interval bounds.
- `all_as_bounds(...)`: Converts arrays or boxes into bounds.
- `Zero()`: Represents zero weights or biases.
- `ibp_rule_monotonic_non_decreasing`, `ibp_rule_monotonic_non_increasing`, `ibp_rule_linear`, `ibp_rule_strongly_convex`, `ibp_rule_reduce_ops`: Helper functions for common categories.

---

Where to Register
-----------------

- IBP rules are registered in `formalax.bounds._ibp`.
