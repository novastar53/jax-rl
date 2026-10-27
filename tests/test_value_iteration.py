import jax.numpy as jnp
import pytest

from jax_rl.dp.value_iteration import value_iteration


def test_value_iteration_simple_chain():
    """
    Two-state MDP.

    State 0:
        action 0 -> state 0, reward 0
        action 1 -> state 1, reward 1

    State 1:
        both actions -> state 1, reward 0

    Optimal behavior from state 0 is action 1.
    """

    P = jnp.array(
        [
            # state 0
            [
                [1.0, 0.0],  # action 0
                [0.0, 1.0],  # action 1
            ],
            # state 1
            [
                [0.0, 1.0],  # action 0
                [0.0, 1.0],  # action 1
            ],
        ]
    )

    R = jnp.array(
        [
            # state 0
            [
                [0.0, 0.0],  # action 0
                [0.0, 1.0],  # action 1
            ],
            # state 1
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        ]
    )

    V, policy = value_iteration(
        P,
        R,
        gamma=0.9,
        tolerance=1e-8,
    )

    expected_V = jnp.array([1.0, 0.0])

    expected_policy = jnp.array(
        [
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )

    assert jnp.allclose(V, expected_V, atol=1e-6)
    assert jnp.allclose(policy, expected_policy, atol=1e-6)


def test_value_iteration_propagates_future_value():
    """
    Three-state chain.

    State 0 -> State 1 -> State 2

    The reward is only received when moving from state 1 to state 2.

    This verifies that value iteration propagates future reward backward.
    """

    P = jnp.array(
        [
            [[0.0, 1.0, 0.0]],  # state 0 -> state 1
            [[0.0, 0.0, 1.0]],  # state 1 -> state 2
            [[0.0, 0.0, 1.0]],  # state 2 -> state 2
        ]
    )

    R = jnp.array(
        [
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0]],
        ]
    )

    V, policy = value_iteration(
        P,
        R,
        gamma=0.9,
        tolerance=1e-8,
    )

    expected_V = jnp.array(
        [
            0.9,
            1.0,
            0.0,
        ]
    )

    expected_policy = jnp.ones((3, 1))

    assert jnp.allclose(V, expected_V, atol=1e-6)
    assert jnp.allclose(policy, expected_policy, atol=1e-6)


def test_value_iteration_splits_probability_across_tied_actions():
    """
    Both actions have identical value.

    The returned greedy policy should split probability equally.
    """

    P = jnp.array(
        [
            [
                [1.0],
                [1.0],
            ]
        ]
    )

    R = jnp.array(
        [
            [
                [0.0],
                [0.0],
            ]
        ]
    )

    V, policy = value_iteration(
        P,
        R,
        gamma=0.9,
        tolerance=1e-8,
    )

    expected_V = jnp.array([0.0])
    expected_policy = jnp.array([[0.5, 0.5]])

    assert jnp.allclose(V, expected_V, atol=1e-6)
    assert jnp.allclose(policy, expected_policy, atol=1e-6)