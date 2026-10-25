import jax
import jax.numpy as jnp

from . import Environment

class BernoulliBandit(Environment):
    def __init__(self, k=10):
        self.k = k

    def reset(self, key):
        p = jax.random.uniform(key, shape=(self.k,), minval=0.0, maxval=1.0)
        observation = 0
        env_state = p
        return env_state, observation

    def step(self, key, env_state, action):
        reward = jax.random.bernoulli(key, p=env_state[action]).astype(jnp.float32)
        observation = 0
        terminated = False

        info = {}

        return env_state, observation, reward, terminated, info



