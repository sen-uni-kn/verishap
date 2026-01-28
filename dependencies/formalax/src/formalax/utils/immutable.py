#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.


class Immutable:
    """A simple base class for immutable classes with ``__slots__``.

    Example usage:

        >>> class MyClass(Immutable):
        >>>    __slots__ = ("field1", "field2", "__hidden")
        >>>
        >>>    def __init__(self, field1: int, field2: int):
        >>>        hidden = ...
        >>>        super().__init__(field1=field1, field2=field2, hidden=hidden)  # use the super initializer to set the field values
        >>>
        >>> t = MyClass(1, 2)
        >>> t.field1 = 3  # raises AttributeError
    """

    __slots__ = ()

    def __init__(self, **kwargs):
        all_slots = sum(
            (
                list(cls.__slots__)
                for cls in self.__class__.__mro__
                if hasattr(cls, "__slots__")
            ),
            [],
        )
        assert len(kwargs) == len(all_slots), "Too few attributes provided."
        for kw, arg in kwargs.items():
            assert kw in all_slots, f"Unknown attribute: {kw}"
            object.__setattr__(self, kw, arg)

    def __setattr__(self, name, value):
        raise AttributeError(f"Cannot set attribute '{name}' of immutable object.")

    def __delattr__(self, name):
        raise AttributeError(f"Cannot delete attribute '{name}' of immutable object.")
