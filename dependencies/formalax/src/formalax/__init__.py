#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT licence.
"""
Formalax - Deep Learning Formal Methods Tools in JAX.
"""
from importlib.metadata import version

__version__ = version("formalax")

from .bounds._src._bounds import Bounds
from .bounds._src._bwcrown import alpha_crown, crown, crown_ibp
from .bounds._src._ibp import ibp, ibp_jaxpr
from .sets.box import Box

__all__ = (
    Bounds,
    Box,
    alpha_crown,
    crown,
    crown_ibp,
    ibp,
    ibp_jaxpr,
)
