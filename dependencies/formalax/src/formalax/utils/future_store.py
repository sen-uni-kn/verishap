#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.


class FutureStoreError(Exception):
    """An error raised when a value is not yet computed."""

    pass


_not_stored = object()


class FutureStore[T]:
    """A store for a value computed later on."""

    def __init__(self):
        self._value = _not_stored

    def __call__(self) -> T:
        if self._value is _not_stored:
            raise FutureStoreError("Value not yet computed.")
        return self._value

    def assign(self, value: T):
        self._value = value


def which_store[T](
    store1: FutureStore[T], store2: FutureStore[T]
) -> tuple[bool, T]:
    """Tests which store has a value assigned.

    Args:
        store1: The first store to test.
        store2: The second store to test.

    Returns:
        Whether the first store had a value assigned
        and the value stored in either store.

    Raises:
        ValueError: if both stores have no value or both have values.
    """
    try:
        value1 = store1()
        try:
            value2 = store2()
            raise ValueError("Both stores have values assigned.")
        except FutureStoreError:
            return True, value1
    except FutureStoreError:
        try:
            value2 = store2()
            return False, value2
        except FutureStoreError:
            raise ValueError("Both stores have no value assigned.") from None
