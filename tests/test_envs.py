
import jax
import jax.numpy as jnp
import pytest

from jax_rl.envs.gridworld import Gridworld, Action


@pytest.mark.parametrize(
    "action, expected_state",
    [
        (Action.UP, 1),
        (Action.RIGHT, 6),
        (Action.DOWN, 9),
        (Action.LEFT, 4),
    ],
)
def test_gridworld_movement(action, expected_state):
    env = Gridworld(size=4)
    key = jax.random.key(42)

    new_state, observation, reward, terminated, info = env.step(
        key, jnp.array(5), action
    )

    assert int(new_state) == expected_state
    assert int(observation) == expected_state
    assert float(reward) == -1.0
    assert not bool(terminated)


@pytest.mark.parametrize(
    "state, action",
    [
        (1, Action.UP),      # Top boundary
        (4, Action.LEFT),    # Left boundary
        (11, Action.RIGHT),  # Right boundary
        (14, Action.DOWN),   # Bottom boundary
    ],
)
def test_gridworld_boundaries(state, action):
    env = Gridworld(size=4)
    key = jax.random.key(42)

    new_state, _, reward, terminated, _ = env.step(
        key, jnp.array(state), action
    )

    assert int(new_state) == state
    assert float(reward) == -1.0
    assert not bool(terminated)


@pytest.mark.parametrize(
    "state, action, expected_state",
    [
        (1, Action.LEFT, 0),
        (14, Action.RIGHT, 15),
    ],
)
def test_gridworld_enter_terminal(state, action, expected_state):
    env = Gridworld(size=4)
    key = jax.random.key(42)

    new_state, _, reward, terminated, _ = env.step(
        key, jnp.array(state), action
    )

    assert int(new_state) == expected_state
    assert float(reward) == -1.0
    assert bool(terminated)


@pytest.mark.parametrize("terminal_state", [0, 15])
def test_gridworld_terminal_absorbing(terminal_state):
    env = Gridworld(size=4)
    key = jax.random.key(42)

    for action in Action:
        new_state, _, reward, terminated, _ = env.step(
            key, jnp.array(terminal_state), action
        )

        assert int(new_state) == terminal_state
        assert float(reward) == 0.0
        assert bool(terminated)


def test_gridworld_jit():
    env = Gridworld(size=4)
    key = jax.random.key(42)

    jitted_step = jax.jit(env.step)

    new_state, _, reward, terminated, _ = jitted_step(
        key,
        jnp.array(5),
        jnp.array(Action.RIGHT),
    )

    assert int(new_state) == 6
    assert float(reward) == -1.0
    assert not bool(terminated)


from jax_rl.envs.bernoulli_bandit import BernoulliBandit


def test_bernoulli_bandit_reset():
    env = BernoulliBandit(k=10)
    key = jax.random.key(42)

    env_state, observation = env.reset(key)

    assert env_state.shape == (10,)
    assert jnp.all(env_state >= 0.0)
    assert jnp.all(env_state <= 1.0)
    assert observation == 0


def test_bernoulli_bandit_step():
    env = BernoulliBandit(k=10)
    key = jax.random.key(42)

    key, reset_key, step_key = jax.random.split(key, 3)

    env_state, _ = env.reset(reset_key)

    new_state, observation, reward, terminated, info = env.step(
        step_key, env_state, action=3
    )

    # Reward must be a scalar Bernoulli outcome
    assert reward.shape == ()
    assert reward in (0, 1)

    # Environment remains stationary
    assert jnp.array_equal(new_state, env_state)

    assert observation == 0
    assert not terminated


@pytest.mark.parametrize(
    "action, expected_reward",
    [
        (0, 0),
        (1, 1),
    ],
)
def test_bernoulli_bandit_deterministic_rewards(
    action, expected_reward
):
    env = BernoulliBandit(k=2)
    key = jax.random.key(42)

    # Arm 0 always fails; arm 1 always succeeds
    env_state = jnp.array([0.0, 1.0])

    _, _, reward, _, _ = env.step(
        key, env_state, action
    )

    assert int(reward) == expected_reward


def test_bernoulli_bandit_jit():
    env = BernoulliBandit(k=2)

    env_state = jnp.array([0.0, 1.0])
    key = jax.random.key(42)

    jitted_step = jax.jit(env.step)

    new_state, _, reward, terminated, _ = jitted_step(
        key,
        env_state,
        jnp.array(1),
    )

    assert int(reward) == 1
    assert jnp.array_equal(new_state, env_state)
    assert not bool(terminated)