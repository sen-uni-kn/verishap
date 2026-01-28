#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.

from .box import Box
from .protocols import HasProjection, MinMaxSplittable, has_projection
from .singleton import Singleton

__all__ = (HasProjection, has_projection, MinMaxSplittable, Box, Singleton)
