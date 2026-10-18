import jax
import jax.numpy as jnp

from . import Environment

class KArmedBandit(Environment):
    def __init__(self, k=10):
        self.k = k


    def reset(self, key):
        q_star = jax.random.normal(key, shape=(self.k,))
        observation = 0
        env_state = q_star
        return env_state, observation


    def step(self, key, env_state, action):
        reward = env_state[action] + jax.random.normal(key) 
        observation = 0
        terminated = False

        info = {}

        return env_state, observation, reward, terminated, info




    