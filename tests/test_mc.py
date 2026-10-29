import jax
import jax.numpy as jnp
import pytest

from jax_rl.envs.gridworld import Gridworld
from jax_rl.mc.episode import generate_episode


@pytest.fixture
def env():
    return Gridworld(size=4)


@pytest.fixture
def random_policy():
    return jnp.full((16, 4), 0.25)


def test_episode_structure(env, random_policy):
    states, actions, rewards = generate_episode(
        env,
        random_policy,
        jax.random.key(42),
    )

    T = len(states)

    assert T > 0
    assert states.shape == (T,)
    assert actions.shape == (T,)
    assert rewards.shape == (T,)

    # Every recorded state is nonterminal.
    assert jnp.all((states > 0) & (states < 15))

    # Actions are valid.
    assert jnp.all((actions >= 0) & (actions < 4))

    # Every transition from a nonterminal
    # Gridworld state produces reward -1.
    assert jnp.all(rewards == -1)


def test_episode_reproducibility(env, random_policy):
    key = jax.random.key(42)

    episode1 = generate_episode(env, random_policy, key)
    episode2 = generate_episode(env, random_policy, key)

    for array1, array2 in zip(episode1, episode2):
        assert jnp.array_equal(array1, array2)


def test_episode_follows_policy():
    class FixedStartGridworld(Gridworld):
        def reset(self, key):
            state = jnp.array(1, dtype=jnp.int32)
            return state, state

    env = FixedStartGridworld(size=4)

    policy = jnp.zeros((16, 4))
    policy = policy.at[:, 1].set(1.0)

    with pytest.raises(RuntimeError):
        generate_episode(
            env,
            policy,
            jax.random.key(42),
            max_steps=10,
        )


def test_episode_reaches_terminal(env):
    # Greedy policy that moves toward the nearest terminal.
    # Action order: UP=0, RIGHT=1, DOWN=2, LEFT=3.
    policy = jnp.zeros((16, 4))

    actions = jnp.array([
        0, 3, 3, 3,
        0, 0, 0, 2,
        0, 0, 1, 2,
        1, 1, 1, 0,
    ])

    policy = policy.at[jnp.arange(16), actions].set(1.0)

    states, chosen_actions, rewards = generate_episode(
        env,
        policy,
        jax.random.key(42),
        max_steps=100,
    )

    assert len(states) > 0
    assert jnp.all(
        chosen_actions == actions[states]
    )
    assert jnp.all(rewards == -1)


def test_episode_max_steps(env, random_policy):
    with pytest.raises(RuntimeError):
        generate_episode(
            env,
            random_policy,
            jax.random.key(42),
            max_steps=1,
        )
