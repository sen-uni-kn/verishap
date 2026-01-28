#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
from contextlib import nullcontext
from typing import ContextManager

import jax.extend.core
from jax.extend import source_info_util


def transform_name_stack_ctx(name: str, transform_stack: bool) -> ContextManager:
    """
    Compare:
    https://github.com/google/jax/blob/5e418f5ab2692d4791816e85ed82eb0834a579cb/jax/_src/interpreters/ad.py#L230
    """
    return (
        source_info_util.transform_name_stack(name)
        if transform_stack
        else nullcontext()
    )


def eqn_name_stack_ctx(
    eqn: jax.extend.core.JaxprEqn, propagate_source_info: bool = True
) -> ContextManager:
    """
    Compare:
    https://github.com/google/jax/blob/5e418f5ab2692d4791816e85ed82eb0834a579cb/jax/_src/core.py#L495
    """
    name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
    traceback = eqn.source_info.traceback if propagate_source_info else None
    return source_info_util.user_context(traceback, name_stack=name_stack)
