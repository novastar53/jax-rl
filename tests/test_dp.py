import jax
import jax.numpy as jnp
import pytest

from jax_rl.envs.gridworld import Gridworld
from jax_rl.dp.finite_mdp import build_deterministic_model


@pytest.fixture
def model():
    env = Gridworld(size=4)
    key = jax.random.key(42)
    return build_deterministic_model(env, key)


def test_model_shapes(model):
    P, R = model

    assert P.shape == (16, 4, 16)
    assert R.shape == (16, 4, 16)


def test_transition_probabilities(model):
    P, _ = model

    # Every state-action pair has exactly one successor.
    assert jnp.allclose(P.sum(axis=2), 1.0)

    # Transitions are deterministic.
    assert jnp.all((P == 0.0) | (P == 1.0))


def test_expected_rewards(model):
    P, R = model

    # Expected immediate reward for each state-action pair.
    expected_rewards = jnp.sum(P * R, axis=2)

    for state in range(16):
        if state in (0, 15):
            assert jnp.allclose(expected_rewards[state], 0.0)
        else:
            assert jnp.allclose(expected_rewards[state], -1.0)


@pytest.mark.parametrize(
    "state, action, expected_next_state",
    [
        (5, 0, 1),   # Up
        (5, 1, 6),   # Right
        (5, 2, 9),   # Down
        (5, 3, 4),   # Left
        (1, 0, 1),   # Upper boundary
        (4, 3, 4),   # Left boundary
        (14, 2, 14), # Lower boundary
        (7, 1, 7),   # Right boundary
        (1, 3, 0),   # Enter terminal state 0
        (14, 1, 15), # Enter terminal state 15
        (0, 1, 0),   # Terminal state is absorbing
        (15, 3, 15), # Terminal state is absorbing
    ],
)
def test_transitions(model, state, action, expected_next_state):
    P, _ = model

    assert int(jnp.argmax(P[state, action])) == expected_next_state
    assert jnp.isclose(
        P[state, action, expected_next_state], 1.0
    )


def test_terminal_states_are_absorbing(model):
    P, R = model

    for state in (0, 15):
        for action in range(4):
            assert jnp.isclose(P[state, action, state], 1.0)
            assert jnp.isclose(R[state, action, state], 0.0)



from jax_rl.dp.policy_evaluation import (
    bellman_sweep,
    iterative_policy_evaluation,
)


@pytest.fixture
def random_policy():
    # Uniform random policy for the 4x4 Gridworld
    return jnp.full((16, 4), 0.25)


def test_bellman_sweep(model, random_policy):
    P, R = model
    V = jnp.zeros(16)

    V_new = bellman_sweep(
        P, R, random_policy, V, gamma=1.0
    )

    expected = jnp.array([
        [ 0, -1, -1, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1,  0],
    ]).reshape(-1)

    assert jnp.allclose(V_new, expected)


def test_iterative_policy_evaluation(model, random_policy):
    P, R = model

    V, iterations = iterative_policy_evaluation(
        P,
        R,
        random_policy,
        gamma=1.0,
        theta=1e-6,
        max_iterations=10000,
    )

    # Sutton & Barto, Example 4.1
    expected = jnp.array([
        [  0, -14, -20, -22],
        [-14, -18, -20, -20],
        [-20, -20, -18, -14],
        [-22, -20, -14,   0],
    ]).reshape(-1)

    assert jnp.allclose(V, expected, atol=1e-3)
    assert 1 < iterations < 10000


def test_max_iterations(model, random_policy):
    P, R = model

    V, iterations = iterative_policy_evaluation(
        P,
        R,
        random_policy,
        gamma=1.0,
        theta=1e-6,
        max_iterations=1,
    )

    expected = jnp.array([
        [ 0, -1, -1, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1,  0],
    ]).reshape(-1)

    assert iterations == 1
    assert jnp.allclose(V, expected)


def test_discount_factor(model, random_policy):
    P, R = model

    V, iterations = iterative_policy_evaluation(
        P,
        R,
        random_policy,
        gamma=0.9,
        theta=1e-6,
        max_iterations=10000,
    )

    # A discounted value function should have smaller
    # absolute values than the undiscounted solution.
    undiscounted_V, _ = iterative_policy_evaluation(
        P, R, random_policy, gamma=1.0
    )

    assert jnp.all(V >= undiscounted_V - 1e-4)
    assert jnp.all(V <= 0)
    assert iterations < 10000
