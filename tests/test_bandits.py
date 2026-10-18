import jax
from jax_rl.envs.bandit_env import KArmedBandit


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
    action = 2
    env_state, observation, reward, terminated, info = env.step(key, env_state, action)