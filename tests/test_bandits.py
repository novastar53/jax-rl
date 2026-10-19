import jax
import jax.numpy as jnp
from jax_rl.envs.bandit_env import KArmedBandit
from jax_rl.bandits.sample_average import SampleAverageAgent


def test_reset():
    key = jax.random.key(42)

    env = KArmedBandit(k=10)
    env_state, observation = env.reset(key)

    assert env_state.shape == (10,)
    assert observation == 0


def test_bandit_step():
    key = jax.random.key(42)
    env = KArmedBandit(k=10)
    env_state, observation = env.reset(key)
    action = 3
    key = jax.random.key(43)
    env_state_new, observation, reward, terminated, info = env.step(key, env_state, action)
    assert(env_state_new.all() == env_state.all())
    assert(observation == 0)
    assert(terminated == False)



def test_sample_average_agent():
    agent = SampleAverageAgent(k=10)
    state = agent.init()

    action = 3
    rewards = [2.0, 4.0, 6.0]
    expected_q = [2.0, 3.0, 4.0]

    for i, reward in enumerate(rewards):
        state = agent.update(state, action, reward)

        assert jnp.isclose(state.Q[action], expected_q[i])
        assert state.N[action] == i + 1

    # Verify final estimate and count
    assert jnp.isclose(state.Q[3], 4.0)
    assert state.N[3] == 3

    # Verify other actions remain unchanged
    other_actions = [a for a in range(10) if a != action]

    assert jnp.all(state.Q[jnp.array(other_actions)] == 0)
    assert jnp.all(state.N[jnp.array(other_actions)] == 0)


from jax_rl.bandits.epsilon_greedy import EpsilonGreedyPolicy


def test_epsilon_greedy():
    policy = EpsilonGreedyPolicy(epsilon=0.2)
    Q = jnp.array([1.0, 2.0, 3.0, 0.0])

    key = jax.random.key(42)
    keys = jax.random.split(key, 10_000)

    # Sample 10,000 actions in parallel
    actions = jax.vmap(policy.select_action, in_axes=(0, None))(keys, Q)

    # Calculate empirical selection probabilities
    counts = jnp.bincount(actions, length=4)
    frequencies = counts / 10_000

    expected = jnp.array([0.05, 0.05, 0.85, 0.05])

    assert jnp.allclose(frequencies, expected, atol=0.02)


def test_epsilon_greedy_uniform():
    policy = EpsilonGreedyPolicy(epsilon=0.2)
    Q = jnp.zeros(4)

    key = jax.random.key(42)
    keys = jax.random.split(key, 10_000)

    # Sample 10,000 actions in parallel
    actions = jax.vmap(policy.select_action, in_axes=(0, None))(keys, Q)

    # Calculate empirical selection probabilities
    counts = jnp.bincount(actions, length=4)
    frequencies = counts / 10_000

    expected = jnp.array([0.25, 0.25, 0.25, 0.25])

    assert jnp.allclose(frequencies, expected, atol=0.02)


def test_optimistic_initialization():
    agent = SampleAverageAgent(k=10, initial_q=5.0)
    state = agent.init()

    # All action-value estimates should initially be 5
    assert jnp.all(state.Q == 5.0)

    # No actions should have been selected yet
    assert jnp.all(state.N == 0)

    # Verify shapes
    assert state.Q.shape == (10,)
    assert state.N.shape == (10,)