#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
from functools import partial

from frozendict import frozendict


class HashablePartial:
    def __init__(self, f, *args, **kwargs):
        self.__partial = partial(f, *args, **kwargs)
        self.__kwargs = frozendict(kwargs)
        self.__hash = hash((f, args, self.__kwargs))

    def __call__(self, *args, **kwargs):
        return self.__partial(*args, **kwargs)

    @property
    def func(self):
        return self.__partial.func

    @property
    def args(self):
        return self.__partial.args

    @property
    def keywords(self):
        return self.__kwargs

    def __eq__(self, other):
        return (
            isinstance(other, HashablePartial)
            and self.__partial.func == other.__partial.func
            and self.__partial.args == other.__partial.args
            and self.__partial.keywords == other.__partial.keywords
        )

    def __hash__(self):
        return self.__hash
