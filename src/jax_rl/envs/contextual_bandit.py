from typing import NamedTuple

import jax
import jax.numpy as jnp

from .base import Environment


class ContextualBanditState(NamedTuple):
    q_star: jax.Array
    context: jax.Array


class ContextualBandit(Environment):
    def __init__(self, num_contexts=3, k=4):
        self.num_contexts = num_contexts
        self.k = k

    def reset(self, key):
        q_key, context_key = jax.random.split(key)
        q_star = jax.random.normal(q_key, shape=(self.num_contexts, self.k))
        context = jax.random.choice(context_key, a=jnp.arange(self.num_contexts))
        env_state = ContextualBanditState(
            q_star=q_star,
            context=context
        )
        observation = context
        return env_state, observation

    def step(self, key, env_state, action):
        reward_key, context_key = jax.random.split(key)
        q = env_state.q_star[env_state.context, action]
        reward = q + jax.random.normal(reward_key)
        context = jax.random.choice(context_key, a=jnp.arange(self.num_contexts))
        new_env_state = ContextualBanditState(q_star=env_state.q_star, context=context)
        observation = context
        info = {}
        terminated = False
        return new_env_state, observation, reward, terminated, info
        
