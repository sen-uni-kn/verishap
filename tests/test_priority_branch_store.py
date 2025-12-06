import jax.numpy as jnp
import pytest

from shap_bounds.priority_branch_store import PriorityBranchStore


def make_store(batch_size: int, root_priority: float = 0.0):
    root_data = jnp.array([root_priority], dtype=float)
    return PriorityBranchStore(
        jnp.array([root_priority], dtype=float), root_data, batch_size
    )


# Insert-specific tests
def test_insert_increases_length_by_batch():
    store = make_store(batch_size=2, root_priority=1.0)
    store.insert(jnp.array([0.5, 0.2]), jnp.array([0.5, 0.2]))
    assert len(store) == 3


def test_insert_accepts_double_batch_size():
    store = make_store(batch_size=3, root_priority=1.0)
    store.insert(jnp.arange(6.0), jnp.arange(6.0))
    assert len(store) == 7  # root + 6 inserted


def test_insert_rejects_more_than_two_batches():
    store = make_store(batch_size=2, root_priority=1.0)
    with pytest.raises(AssertionError):
        store.insert(jnp.arange(5.0), jnp.arange(5.0))


def test_insert_updates_highest_priority_when_larger():
    store = make_store(batch_size=2, root_priority=0.1)
    store.insert(jnp.array([0.9]), jnp.array([0.9]))
    assert store.highest_priority() == pytest.approx(0.9)


def test_insert_keeps_highest_priority_when_lower():
    store = make_store(batch_size=2, root_priority=0.9)
    store.insert(jnp.array([0.2, 0.3]), jnp.array([0.2, 0.3]))
    assert store.highest_priority() == pytest.approx(0.9)


def test_insert_with_multiple_leaves_keeps_shapes_and_values():
    root = (jnp.array([[1.0]]), jnp.array([[2.0]]))
    store = PriorityBranchStore(jnp.array([1.0]), root, batch_size=2)
    priorities = jnp.array([3.0, 2.5])
    data = (jnp.array([[3.0], [2.5]]), jnp.array([[4.0], [3.5]]))
    store.insert(priorities, data)

    extracted = store.extract_max()
    assert extracted[0].shape == (2, 1)
    assert extracted[1].shape == (2, 1)
    assert jnp.allclose(extracted[0][:, 0], jnp.array([3.0, 2.5]))
    assert jnp.allclose(extracted[1][:, 0], jnp.array([4.0, 3.5]))


def test_insert_records_multiple_batches_count():
    store = make_store(batch_size=4, root_priority=0.0)
    store.insert(jnp.array([0.1, 0.2]), jnp.array([0.1, 0.2]))
    store.insert(jnp.array([0.3, 0.4, 0.5, 0.6]), jnp.array([0.3, 0.4, 0.5, 0.6]))
    assert len(store) == 7


@pytest.mark.parametrize(
    ("batch_size", "root_priority", "insert_batches", "expected_batches"),
    [
        pytest.param(
            2,
            1.0,
            [],
            [[1.0]],
            id="only-root",
        ),
        pytest.param(
            2,
            0.1,
            [jnp.array([0.9, 0.8])],
            [[0.9, 0.8], [0.1]],
            id="full-batch-then-root",
        ),
        pytest.param(
            3,
            0.1,
            [jnp.array([0.9, 0.8])],
            [[0.9, 0.8, 0.1], []],
            id="partial-batch",
        ),
        pytest.param(
            2,
            0.3,
            [jnp.array([0.5]), jnp.array([0.4, 0.2])],
            [[0.5, 0.4], [0.3, 0.2]],
            id="interleaved-inserts",
        ),
        pytest.param(
            2,
            0.1,
            [jnp.array([0.5, 0.5, 0.2])],
            [[5.0, 4.0], [2.0, 0.1]],
            id="tied-priorities-distinct-data",
        ),
        pytest.param(
            2,
            -1.0,
            [jnp.array([0.9]), jnp.array([0.8])],
            [[0.9, 0.8], [-1.0]],
            id="exact-batch-then-root",
        ),
        pytest.param(
            3,
            0.0,
            [jnp.array([0.4, 0.9, 0.1, 0.8, 0.6, 0.2])],
            [[0.9, 0.8, 0.6], [0.4, 0.2, 0.1], [0.0]],
            id="two-batches-partition",
        ),
        pytest.param(
            2,
            -jnp.inf,
            [jnp.array([jnp.inf, 1.0]), jnp.array([-jnp.inf])],
            [[jnp.inf, 1.0], [-jnp.inf, -jnp.inf]],
            id="infinite-priorities",
        ),
    ],
)
def test_scalar_sequences_per_batch(
    batch_size: int,
    root_priority: float,
    insert_batches: list[jnp.ndarray],
    expected_batches: list[list[float]],
):
    store = make_store(batch_size=batch_size, root_priority=root_priority)

    for batch in insert_batches:
        data = batch
        # Use distinct data for tied-priority case to check alignment
        if batch.shape[0] == 3 and jnp.allclose(batch, jnp.array([0.5, 0.5, 0.2])):
            data = jnp.array([5.0, 4.0, 2.0])
        store.insert(batch, data)

    outputs = [store.extract_max() for _ in range(len(expected_batches))]
    for out, expected in zip(outputs, expected_batches, strict=True):
        assert jnp.allclose(out, jnp.array(expected))


# Complex pytree and multi-dimensional data tests
def test_nested_dict_pytree_roundtrip():
    root = {"a": {"x": jnp.array([[1.0, 2.0]])}, "b": (jnp.array([[3.0]]),)}
    store = PriorityBranchStore(jnp.array([0.5]), root, batch_size=2)
    priorities = jnp.array([0.9, 0.8])
    data = {
        "a": {"x": jnp.array([[9.0, 8.0], [7.0, 6.0]])},
        "b": (jnp.array([[5.0], [4.0]]),),
    }
    store.insert(priorities, data)
    extracted = store.extract_max()
    assert jnp.allclose(extracted["a"]["x"], jnp.array([[9.0, 8.0], [7.0, 6.0]]))
    assert jnp.allclose(extracted["b"][0], jnp.array([[5.0], [4.0]]))


def test_list_tuple_dict_mixed_pytree():
    root = (
        [jnp.array([[1.0, 1.0]])],
        {"k": jnp.array([[2.0]])},
        (jnp.array([[3.0, 4.0]]),),
    )
    store = PriorityBranchStore(jnp.array([0.5]), root, batch_size=2)
    priorities = jnp.array([1.2, 1.1])
    data = (
        [jnp.array([[10.0, 11.0], [12.0, 13.0]])],
        {"k": jnp.array([[20.0], [21.0]])},
        (jnp.array([[30.0, 31.0], [32.0, 33.0]]),),
    )
    store.insert(priorities, data)
    out = store.extract_max()
    assert jnp.allclose(out[0][0], jnp.array([[10.0, 11.0], [12.0, 13.0]]))
    assert jnp.allclose(out[1]["k"], jnp.array([[20.0], [21.0]]))
    assert jnp.allclose(out[2][0], jnp.array([[30.0, 31.0], [32.0, 33.0]]))


@pytest.mark.parametrize("shape", [(2, 2), (3, 4, 1)])
def test_multidimensional_arrays_preserve_shapes(shape):
    batch_size = 3
    root = jnp.zeros((1, *shape))
    store = PriorityBranchStore(jnp.array([0.5]), root, batch_size=batch_size)
    priorities = jnp.array([0.9, 0.8, 0.7])
    data = jnp.arange(jnp.prod(jnp.array(shape)) * 3.0).reshape((3, *shape))
    store.insert(priorities, data)
    extracted = store.extract_max()
    assert extracted.shape == (batch_size, *shape)
    assert jnp.allclose(extracted, data)


def test_multidimensional_arrays_sorted_by_priority():
    batch_size = 2
    root = jnp.zeros((1, 2, 2))
    store = PriorityBranchStore(jnp.array([0.1]), root, batch_size=batch_size)
    priorities = jnp.array([0.4, 0.9, 0.2])
    data = jnp.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ]
    )
    store.insert(priorities, data)
    out = store.extract_max()
    assert jnp.allclose(out[0], data[1])  # priority 0.9
    assert jnp.allclose(out[1], data[0])  # priority 0.4


def test_extract_batches_multidimensional_data_until_empty():
    batch_size = 2
    root = jnp.zeros((1, 2, 1))
    store = PriorityBranchStore(jnp.array([0.05]), root, batch_size=batch_size)
    priorities = jnp.linspace(0.1, 1.0, num=7)
    data = jnp.arange(7 * 2.0).reshape((7, 2, 1))
    for i in range(0, priorities.shape[0], 4):  # respect 2B=4
        store.insert(priorities[i : i + 4], data[i : i + 4])
    seen = []
    while True:
        batch = store.extract_max()
        if batch.shape[0] == 0:
            break
        seen.append(batch)
    concatenated = jnp.concatenate(seen, axis=0)
    expected = jnp.concatenate([data[priorities.argsort()[::-1]], root], axis=0)
    assert jnp.allclose(concatenated, expected[: concatenated.shape[0]])


def test_nested_pytree_partial_batches_extract_in_order():
    root = {"l": jnp.array([[0.1, 0.2]])}
    store = PriorityBranchStore(jnp.array([0.05]), root, batch_size=2)
    priorities = jnp.array([0.9, 0.6, 0.4])
    data = {"l": jnp.array([[9.0, 9.1], [6.0, 6.1], [4.0, 4.1]])}
    store.insert(priorities, data)
    first = store.extract_max()
    second = store.extract_max()
    assert jnp.allclose(first["l"], jnp.array([[9.0, 9.1], [6.0, 6.1]]))
    assert jnp.allclose(second["l"], jnp.array([[4.0, 4.1], [0.1, 0.2]]))


def test_deep_heap_four_levels_orders_descending():
    batch_size = 2
    store = make_store(batch_size=batch_size, root_priority=0.0)
    priorities = jnp.arange(1, 31, dtype=float)  # enough to create multiple levels
    for i in range(0, priorities.shape[0], 4):
        store.insert(priorities[i : i + 4], priorities[i : i + 4])
    gathered = []
    while True:
        batch = store.extract_max()
        if batch.shape[0] == 0:
            break
        gathered.extend(batch.tolist())
    expected = sorted(priorities.tolist() + [0.0], reverse=True)
    assert jnp.allclose(jnp.array(gathered), jnp.array(expected))


def test_large_heap_up_to_hundred_elements_orders_descending():
    batch_size = 5
    store = make_store(batch_size=batch_size, root_priority=-1.0)
    priorities = jnp.arange(1, 101, dtype=float)
    for i in range(0, priorities.shape[0], 10):  # insert at 2B chunks
        store.insert(priorities[i : i + 10], priorities[i : i + 10])
    collected = []
    while True:
        batch = store.extract_max()
        if batch.shape[0] == 0:
            break
        collected.extend(batch.tolist())
    expected = sorted(priorities.tolist() + [-1.0], reverse=True)
    assert jnp.allclose(jnp.array(collected), jnp.array(expected))


def test_large_heap_multidimensional_entries():
    batch_size = 4
    root = jnp.zeros((1, 3, 3))
    store = PriorityBranchStore(jnp.array([0.0]), root, batch_size=batch_size)
    priorities = jnp.linspace(0.1, 2.0, num=40)
    data = jnp.arange(40 * 3 * 3.0).reshape((40, 3, 3))
    for i in range(0, priorities.shape[0], 8):  # 2B = 8
        store.insert(priorities[i : i + 8], data[i : i + 8])
    outputs = []
    while True:
        batch = store.extract_max()
        if batch.shape[0] == 0:
            break
        outputs.append(batch)
    concatenated = jnp.concatenate(outputs, axis=0)
    expected_indices = jnp.argsort(priorities)[::-1]
    expected = data[expected_indices]
    # Only compare available entries (root may appear last)
    assert jnp.allclose(concatenated[: expected.shape[0]], expected)


def test_extract_reduces_length_by_batch():
    store = make_store(batch_size=2, root_priority=0.1)
    store.insert(jnp.array([0.9, 0.8]), jnp.array([0.9, 0.8]))
    assert len(store) == 3
    store.extract_max()
    assert len(store) == 1  # expected to drop by extracted batch size


def test_extract_to_empty_sets_length_zero():
    store = make_store(batch_size=3, root_priority=0.5)
    assert len(store) == 1
    store.extract_max()
    assert len(store) == 0


@pytest.mark.parametrize(
    ("batch_size", "root_priority", "prios"),
    [
        pytest.param(2, -1.0, [5.0, 4.0, 3.0, 2.0], id="descending"),
        pytest.param(2, 10.0, [1.0, 2.0, 3.0, 4.0], id="ascending"),
        pytest.param(3, 0.0, [1.0, 1.0, 1.0, 1.0, 1.0], id="all-equal"),
        pytest.param(2, -5.0, [-1.0, -2.0, -3.0], id="all-negative"),
        pytest.param(4, 0.0, [0.3, 0.9, 0.1, 0.8, 0.5, 0.7, 0.2, 0.6], id="randomized"),
        pytest.param(2, 0.0, list(range(1, 31)), id="four-levels"),
        pytest.param(5, -1.0, list(range(1, 101)), id="hundred-elements"),
    ],
)
def test_scalar_global_drain_orders_descending(
    batch_size: int, root_priority: float, prios: list[float]
):
    store = make_store(batch_size=batch_size, root_priority=root_priority)
    prios_arr = jnp.array(prios, dtype=float)
    for i in range(0, prios_arr.shape[0], batch_size * 2):
        store.insert(
            prios_arr[i : i + batch_size * 2], prios_arr[i : i + batch_size * 2]
        )

    collected = []
    while True:
        batch = store.extract_max()
        if batch.shape[0] == 0:
            break
        collected.extend(batch.tolist())

    expected = sorted(prios + [root_priority], reverse=True)
    assert jnp.allclose(jnp.array(collected), jnp.array(expected))


@pytest.mark.parametrize(
    ("batch_size", "root_priority", "insert_batches"),
    [
        pytest.param(
            3,
            0.5,
            [jnp.array([0.6]), jnp.array([0.7]), jnp.array([0.8])],
            id="small-inserts-fill-buffer",
        ),
        pytest.param(2, 0.1, [jnp.array([], dtype=float)], id="empty-insert-noop"),
        pytest.param(
            3,
            0.0,
            [jnp.array([0.5, 0.4, 0.3, 0.2])],
            id="more-than-batch-less-than-two",
        ),
        pytest.param(
            3,
            0.0,
            [jnp.array([0.3, 0.4, 0.5]), jnp.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1])],
            id="capacity-boundary",
        ),
        pytest.param(
            2,
            -jnp.inf,
            [jnp.array([jnp.inf, 1.0]), jnp.array([-jnp.inf])],
            id="infinite-priorities-again",
        ),
    ],
)
def test_misc_insert_edge_cases(batch_size: int, root_priority: float, insert_batches):
    store = make_store(batch_size=batch_size, root_priority=root_priority)
    for batch in insert_batches:
        store.insert(batch, batch)

    collected = []
    while True:
        batch = store.extract_max()
        if batch.shape[0] == 0:
            break
        collected.extend(batch.tolist())

    expected = sorted(
        [*jnp.concatenate(insert_batches).tolist(), root_priority]
        if insert_batches and insert_batches[0].size > 0
        else [root_priority],
        reverse=True,
    )
    assert jnp.allclose(jnp.array(collected), jnp.array(expected))
