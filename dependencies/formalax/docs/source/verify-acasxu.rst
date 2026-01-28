Verifying ACASXU safety properties
========================================

Overview
--------

This example demonstrates verifying a **safety property** for the ACASXU network using `formalax`'s `crown_ibp` bounds propagation.  

**Network Inputs (X)**

- `ρ` : Distance from ownship to intruder  
- `θ` : Angle to intruder relative to ownship direction  
- `ψ` : Heading angle of intruder relative to ownship heading direction  
- `vown` : Speed of ownship  
- `vint` : Speed of intruder

**Network Outputs (Y)**

- `COC` : Clear-of-Conflict advisory  
- `weak_right`, `strong_right`, `weak_left`, `strong_left` : Other advisories

**Property (Reluplex - φ2)**

- **Description**: If the intruder is distant and slower than the ownship, the COC advisory will **never** be the maximal score.  
- **Input constraints**:  
  - `ρ ≥ 55947.691`  
  - `vown ≥ 1145`  
  - `vint ≤ 60`  
- **Desired output property**: The score for `COC` is **not the maximal score**.

.. contents::
   :local:
   :depth: 2

Step 1: Define Input Bounds
---------------------------

.. code-block:: python

    from formalax import Box
    import jax.numpy as jnp

    input_bounds = Box(
        lower_bound=jnp.array([55947.691, -jnp.pi, -jnp.pi, 1145.0, 0.0]),
        upper_bound=jnp.array([60760.0, jnp.pi, jnp.pi, 1200.0, 60.0])
    )

Step 2: Load the ACASXU Network
-------------------------------

.. code-block:: python

    from tests.nets.acasxu import get_acasxu_network

    model = get_acasxu_network(1, 1)

Step 3: Define Verification Function
------------------------------------

The verification function computes **CROWN-IBP bounds** and checks the φ2 property.

.. code-block:: python

    from formalax import crown_ibp

    def verify_property(model, input_bounds):
        # Compute output bounds
        compute_bounds = crown_ibp(model)
        bounds = compute_bounds(input_bounds)
        out_lb, out_ub = bounds.concrete

        # COC upper bound
        coc_ub = out_ub[0]

        # Check if any other advisory lower bound exceeds COC upper bound
        diffs = [out_lb[i] - coc_ub for i in range(1, 5)]

        print("COC output bounds:", out_lb[0], out_ub[0])
        print("Alternative output bounds:")
        print("Lower:", out_lb[1:])
        print("Upper:", out_ub[1:])

        assert any(d > 0 for d in diffs), "Property violated"

Step 4: Run Verification
------------------------

.. code-block:: python

    verify_property(model, input_bounds)

Expected Result
---------------

If the property holds, at least one alternative advisory has a **lower bound exceeding COC's upper bound**, confirming that COC is not maximal under the given input constraints.
