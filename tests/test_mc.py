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



import jax
import jax.numpy as jnp
import pytest

from jax_rl.mc.mc_prediction import mc_prediction
from jax_rl.envs.gridworld import Gridworld


class FixedEpisodeEnv:
    """
    A deterministic environment producing:

        states  = [0, 1, 0, 2]
        rewards = [1, 2, 3, 4]

    The episode terminates after four transitions.
    """

    num_states = 4

    def reset(self, key):
        env_state = 0
        observation = jnp.array(0, dtype=jnp.int32)
        return env_state, observation

    def step(self, key, env_state, action):
        next_states = jnp.array([1, 0, 2, 3])
        rewards = jnp.array([1.0, 2.0, 3.0, 4.0])

        next_state = next_states[env_state]
        reward = rewards[env_state]

        new_env_state = env_state + 1
        terminated = new_env_state == 4

        return new_env_state, next_state, reward, terminated, {}


def test_first_visit_mc_single_episode():
    env = FixedEpisodeEnv()
    policy = jnp.ones((4, 1))

    V, N = mc_prediction(
        env,
        policy,
        jax.random.key(42),
        num_episodes=1,
        gamma=1.0,
    )

    # Returns for the episode:
    #
    # State 0 (first visit): 1 + 2 + 3 + 4 = 10
    # State 1:               2 + 3 + 4 = 9
    # State 0 (second visit): 3 + 4 = 7 (ignored)
    # State 2:               4
    expected_V = jnp.array([10.0, 9.0, 4.0, 0.0])
    expected_N = jnp.array([1, 1, 1, 0])

    assert jnp.allclose(V, expected_V)
    assert jnp.array_equal(N, expected_N)


def test_first_visit_mc_discounting():
    env = FixedEpisodeEnv()
    policy = jnp.ones((4, 1))

    V, N = mc_prediction(
        env,
        policy,
        jax.random.key(42),
        num_episodes=1,
        gamma=0.5,
    )

    # Discounted returns:
    #
    # State 0: 1 + 0.5*2 + 0.25*3 + 0.125*4 = 3.25
    # State 1: 2 + 0.5*3 + 0.25*4 = 4.5
    # State 2: 4
    expected_V = jnp.array([3.25, 4.5, 4.0, 0.0])
    expected_N = jnp.array([1, 1, 1, 0])

    assert jnp.allclose(V, expected_V)
    assert jnp.array_equal(N, expected_N)


def test_first_visit_mc_multiple_episodes():
    env = FixedEpisodeEnv()
    policy = jnp.ones((4, 1))

    V, N = mc_prediction(
        env,
        policy,
        jax.random.key(42),
        num_episodes=3,
        gamma=1.0,
    )

    # Every episode is identical, so estimates stay unchanged.
    expected_V = jnp.array([10.0, 9.0, 4.0, 0.0])
    expected_N = jnp.array([3, 3, 3, 0])

    assert jnp.allclose(V, expected_V)
    assert jnp.array_equal(N, expected_N)


@pytest.mark.slow
def test_first_visit_mc_gridworld():
    env = Gridworld(size=4)
    policy = jnp.full((16, 4), 0.25)

    V, N = mc_prediction(
        env,
        policy,
        jax.random.key(42),
        num_episodes=5000,
        gamma=1.0,
    )

    # Sutton & Barto Example 4.1:
    # Exact random-policy values from dynamic programming.
    expected = jnp.array([
        [  0, -14, -20, -22],
        [-14, -18, -20, -20],
        [-20, -20, -18, -14],
        [-22, -20, -14,   0],
    ]).reshape(-1)

    # Monte Carlo estimates fluctuate due to sampling.
    # Allow an absolute error of 3 per nonterminal state.
    assert jnp.allclose(V, expected, atol=3.0)

    # Both terminal states should remain unvisited.
    assert N[0] == 0
    assert N[15] == 0

    # All nonterminal states should have been visited.
    assert jnp.all(N[1:15] > 0)



def test_mc_prediction_without_callback():
    env = FixedEpisodeEnv()
    policy = jnp.ones((4, 1))

    V, N = mc_prediction(
        env,
        policy,
        jax.random.key(42),
        num_episodes=3,
    )

    assert jnp.allclose(
        V,
        jnp.array([10.0, 9.0, 4.0, 0.0]),
    )
    assert jnp.array_equal(
        N,
        jnp.array([3, 3, 3, 0]),
    )


def test_mc_prediction_callback():
    env = FixedEpisodeEnv()
    policy = jnp.ones((4, 1))

    calls = []

    def callback(episode, V, N):
        calls.append(
            (episode, jnp.array(V), jnp.array(N))
        )

    mc_prediction(
        env,
        policy,
        jax.random.key(42),
        num_episodes=5,
        callback=callback,
        callback_every=2,
    )

    # Every two episodes, plus the final episode.
    assert [episode for episode, _, _ in calls] == [2, 4, 5]

    assert jnp.array_equal(
        calls[-1][2],
        jnp.array([5, 5, 5, 0]),
    )


def test_mc_prediction_every_visit():
    env = FixedEpisodeEnv()
    policy = jnp.ones((4, 1))

    V, N = mc_prediction(
        env,
        policy,
        jax.random.key(42),
        num_episodes=1,
        visit_mode="every",
    )

    # State 0 appears twice, with returns 10 and 7.
    expected_V = jnp.array([8.5, 9.0, 4.0, 0.0])
    expected_N = jnp.array([2, 1, 1, 0])

    assert jnp.allclose(V, expected_V)
    assert jnp.array_equal(N, expected_N)


from jax_rl.envs.gridworld import Gridworld
from jax_rl.mc.mc_prediction import mc_prediction, mc_prediction_jit


@pytest.mark.parametrize("visit_mode", ["first", "every"])
@pytest.mark.parametrize("gamma", [1.0, 0.5])
def test_mc_jit_matches_reference(visit_mode, gamma):
    env = Gridworld(size=4)
    policy = jnp.full((16, 4), 0.25)
    key = jax.random.key(42)

    reference_V, reference_N = mc_prediction(
        env,
        policy,
        key,
        num_episodes=100,
        gamma=gamma,
        visit_mode=visit_mode,
    )

    compiled_V, compiled_N = mc_prediction_jit(
        env,
        policy,
        key,
        num_episodes=100,
        gamma=gamma,
        visit_mode=visit_mode,
    )

    assert jnp.array_equal(reference_N, compiled_N)
    assert jnp.allclose(
        reference_V,
        compiled_V,
        atol=1e-4,
        rtol=1e-5,
    )


@pytest.mark.parametrize("visit_mode", ["first", "every"])
def test_mc_jit_zero_episodes(visit_mode):
    env = Gridworld(size=4)
    policy = jnp.full((16, 4), 0.25)

    V, N = mc_prediction_jit(
        env,
        policy,
        jax.random.key(42),
        num_episodes=0,
        visit_mode=visit_mode,
    )

    assert jnp.array_equal(V, jnp.zeros(16))
    assert jnp.array_equal(N, jnp.zeros(16))


def test_mc_jit_max_steps():
    env = Gridworld(size=4)
    policy = jnp.full((16, 4), 0.25)

    # A one-step episode is not guaranteed to terminate.
    # Use a reproducible batch and a strict step limit.
    with pytest.raises(RuntimeError, match="did not terminate"):
        mc_prediction_jit(
            env,
            policy,
            jax.random.key(42),
            num_episodes=100,
            max_steps=1,
        )


def test_mc_jit_invalid_visit_mode():
    env = Gridworld(size=4)
    policy = jnp.full((16, 4), 0.25)

    with pytest.raises(ValueError, match="visit_mode"):
        mc_prediction_jit(
            env,
            policy,
            jax.random.key(42),
            visit_mode="invalid",
        )



@pytest.mark.parametrize("visit_mode", ["first", "every"])
def test_mc_jit_single_episode_matches_reference(visit_mode):
    env = Gridworld(size=4)
    policy = jnp.full((16, 4), 0.25)
    key = jax.random.key(42)

    V_ref, N_ref = mc_prediction(
        env,
        policy,
        key,
        num_episodes=1,
        visit_mode=visit_mode,
    )

    V_jit, N_jit = mc_prediction_jit(
        env,
        policy,
        key,
        num_episodes=1,
        visit_mode=visit_mode,
    )

    assert jnp.array_equal(N_ref, N_jit)
    assert jnp.allclose(V_ref, V_jit, atol=1e-5)
