from typing import NamedTuple

import jax
import jax.numpy as jnp


class AgentState(NamedTuple):
    Q: jax.Array
    N: jax.Array


class SampleAverageAgent:
    def __init__(self, k=10, alpha=None, initial_q=0.0):
        self.k = k
        self.alpha = alpha
        self.initial_q = initial_q

    def init(self):
        state = AgentState(Q=jnp.ones(self.k) * self.initial_q, N=jnp.zeros(self.k, dtype=jnp.int32))
        return state


    def update(self, agent_state, action, reward):
        q_old = agent_state.Q[action]
        n_old = agent_state.N[action]
        lr = self.alpha if self.alpha is not None else 1/(n_old + 1)
        q_update = q_old + lr * (reward - q_old)
        q_new = agent_state.Q.at[action].set(q_update)
        n_new = agent_state.N.at[action].set(n_old + 1)
        return AgentState(Q=q_new, N=n_new)