
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
