#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.


def strict_zip(*args):
    """``zip`` but with ``strict=True`` by default."""
    return zip(*args, strict=True)
