#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
import jax.extend.linear_util as lu


@lu.transformation
def args_to_kwargs(to_key: tuple[str | None, ...], *args):
    """Convert positional arguments to keyword arguments.

    Args:
        to_key: A tuple of strings or ``None``.
            Each positional argument is converted to a keyword argument with the
            corresponding key from ``to_key?? as the keyword, unless the key
            is ``None``.
            In that case, the positional argument remains a positional argument.
    """
    args = tuple(args)

    kwargs = {
        key: arg for key, arg in zip(to_key, args, strict=False) if key is not None
    }
    args = (
        tuple(arg for key, arg in zip(to_key, args, strict=False) if key is None)
        + args[len(to_key) :]
    )
    ans = yield args, kwargs
    yield ans
