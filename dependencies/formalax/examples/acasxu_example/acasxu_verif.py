import jax.numpy as jnp

from formalax import Box, crown_ibp
from tests.nets.acasxu import get_acasxu_network

"""
ACASXU: X = [ρ (Distance from ownship to intruder),
             θ (Angle to intruder relative to ownship dirextion)
             ψ (Heading angle of intruder relative to ownship heading direction)
             vown (Speed of ownship)
             vint (Speed of intruder)
             ]

        Y = [COC, weak_right, strong_right, weak_left, strong_left]


Reluplex - φ2 Property:

Description: If the intruder is distant and is significantly slower than the
ownship, the score of a COC advisory will never be maximal.

Input constraints:
            ρ ≥ 55947.691
            vown ≥ 1145
            vint ≤ 60

Desired output property: the score for COC is not the maximal score.
"""

input_bounds = Box(
    lower_bound=jnp.array([55947.691, -jnp.pi, -jnp.pi, 1145.0, 0.0]),
    upper_bound=jnp.array([60760.0, jnp.pi, jnp.pi, 1200.0, 60.0]),
)

model = get_acasxu_network(2, 1)


def verify_property(model, input_bounds):
    compute_bounds = crown_ibp(model)
    bounds = compute_bounds(input_bounds)
    out_lb, out_ub = bounds.concrete

    # For each i ≠ 0, check y_i_LB > y_0_UB
    coc_ub = out_ub[0]  # COC upper bound
    diffs = [out_lb[i] - coc_ub for i in range(1, 5)]

    print("COE output bounds:", out_lb[0], out_ub[0])
    print("Alternative output bounds:")
    print("Lower:", out_lb[1:])
    print("Upper:", out_ub[1:])

    assert any(d > 0 for d in diffs), "Property violated"


verify_property(model, input_bounds)
