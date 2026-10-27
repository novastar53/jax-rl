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



from jax_rl.dp.policy_improvement import greedy_policy_improvement


def test_greedy_policy_improvement():
    # Two states, three actions.
    # Every action transitions back to the current state.
    P = jnp.zeros((2, 3, 2))
    P = P.at[0, :, 0].set(1.0)
    P = P.at[1, :, 1].set(1.0)

    # Immediate rewards determine the greedy actions.
    R = jnp.zeros((2, 3, 2))
    R = R.at[0, :, 0].set(jnp.array([1.0, 5.0, 2.0]))
    R = R.at[1, :, 1].set(jnp.array([4.0, 1.0, 3.0]))

    V = jnp.zeros(2)

    policy = greedy_policy_improvement(P, R, V)

    expected = jnp.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    assert jnp.allclose(policy, expected)


def test_greedy_policy_ties():
    # One state, four actions, all self-transitions.
    P = jnp.ones((1, 4, 1))

    R = jnp.array([[
        [2.0],
        [5.0],
        [5.0],
        [1.0],
    ]])

    V = jnp.zeros(1)

    policy = greedy_policy_improvement(P, R, V)

    expected = jnp.array([
        [0.0, 0.5, 0.5, 0.0]
    ])

    assert jnp.allclose(policy, expected)


def test_gridworld_greedy_policy(model, random_policy):
    P, R = model

    # First evaluate the uniform random policy.
    V, _ = iterative_policy_evaluation(
        P,
        R,
        random_policy,
        gamma=1.0,
        theta=1e-6,
    )

    # Improve the policy with respect to V.
    improved_policy = greedy_policy_improvement(
        P, R, V, gamma=1.0
    )

    assert improved_policy.shape == (16, 4)

    # Every row must define a valid probability distribution.
    assert jnp.allclose(
        improved_policy.sum(axis=1),
        1.0,
    )

    assert jnp.all(improved_policy >= 0.0)
    assert jnp.all(improved_policy <= 1.0)

    # State 1: LEFT leads directly to terminal state 0.
    # Action order: UP=0, RIGHT=1, DOWN=2, LEFT=3.
    assert jnp.allclose(
        improved_policy[1],
        jnp.array([0.0, 0.0, 0.0, 1.0]),
    )

    # State 4: UP leads directly to terminal state 0.
    assert jnp.allclose(
        improved_policy[4],
        jnp.array([1.0, 0.0, 0.0, 0.0]),
    )

    # Terminal states: every action is equally valuable.
    for state in (0, 15):
        assert jnp.allclose(
            improved_policy[state],
            jnp.full(4, 0.25),
        )


def test_greedy_policy_jit(model):
    P, R = model
    V = jnp.zeros(16)

    jit_improvement = jax.jit(greedy_policy_improvement)

    policy = jit_improvement(P, R, V)

    expected = greedy_policy_improvement(P, R, V)

    assert jnp.allclose(policy, expected)



from jax_rl.dp.policy_iteration import policy_iteration


def test_policy_iteration_optimal_values(model):
    P, R = model

    V, policy, iterations = policy_iteration(
        P,
        R,
        gamma=1.0,
        theta=1e-6,
        max_iterations=100,
    )

    # Sutton & Barto 4x4 Gridworld optimal values.
    expected = jnp.array([
        [ 0, -1, -2, -3],
        [-1, -2, -3, -2],
        [-2, -3, -2, -1],
        [-3, -2, -1,  0],
    ]).reshape(-1)

    assert jnp.allclose(V, expected, atol=1e-3)
    assert 1 <= iterations < 100


def test_policy_iteration_valid_policy(model):
    P, R = model

    V, policy, _ = policy_iteration(P, R, gamma=1.0)

    assert V.shape == (16,)
    assert policy.shape == (16, 4)

    # Every state has a valid action distribution.
    assert jnp.allclose(policy.sum(axis=1), 1.0)
    assert jnp.all(policy >= 0.0)
    assert jnp.all(policy <= 1.0)

    # All actions in terminal states are equivalent.
    assert jnp.allclose(policy[0], jnp.full(4, 0.25))
    assert jnp.allclose(policy[15], jnp.full(4, 0.25))


def test_policy_iteration_greedy_actions(model):
    P, R = model

    V, policy, _ = policy_iteration(P, R, gamma=1.0)

    # Q-values under the final value function.
    Q = jnp.sum(
        P * (R + V[None, None, :]),
        axis=2,
    )

    max_Q = jnp.max(Q, axis=1, keepdims=True)

    # The learned policy should assign probability only
    # to optimal actions (allowing numerical tolerance).
    suboptimal = Q < max_Q - 1e-4

    assert jnp.all(policy[suboptimal] == 0.0)

    # State 1 must move LEFT toward terminal state 0.
    assert jnp.allclose(
        policy[1],
        jnp.array([0.0, 0.0, 0.0, 1.0]),
    )

    # State 4 must move UP toward terminal state 0.
    assert jnp.allclose(
        policy[4],
        jnp.array([1.0, 0.0, 0.0, 0.0]),
    )


def test_policy_iteration_max_iterations(model):
    P, R = model

    V, policy, iterations = policy_iteration(
        P,
        R,
        gamma=1.0,
        theta=1e-6,
        max_iterations=1,
    )

    assert iterations == 1

    # One policy improvement step should produce
    # a valid policy and an evaluated value function.
    assert V.shape == (16,)
    assert policy.shape == (16, 4)
    assert jnp.allclose(policy.sum(axis=1), 1.0)

    # The returned value function should correspond
    # to the returned policy.
    evaluated_V, _ = iterative_policy_evaluation(
        P,
        R,
        policy,
        gamma=1.0,
        theta=1e-6,
    )

    assert jnp.allclose(V, evaluated_V, atol=1e-3)


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

def test_value_iteration_gridworld(model):
    P, R = model

    V, policy = value_iteration(
        P,
        R,
        gamma=1.0,
        tolerance=1e-6,
    )

    expected = jnp.array([
        [ 0, -1, -2, -3],
        [-1, -2, -3, -2],
        [-2, -3, -2, -1],
        [-3, -2, -1,  0],
    ]).reshape(-1)

    assert jnp.allclose(V, expected, atol=1e-6)

    assert policy.shape == (16, 4)
    assert jnp.allclose(policy.sum(axis=1), 1.0)