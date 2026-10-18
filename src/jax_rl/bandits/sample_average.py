from typing import NamedTuple

import jax
import jax.numpy as jnp


class AgentState(NamedTuple):
    Q: jax.Array
    N: jax.Array


class SampleAverageAgent:
    def __init__(self, k=10):
        self.k = k

    def init(self):
        state = AgentState(Q=jnp.zeros(self.k), N=jnp.zeros(self.k, dtype=jnp.int32))
        return state


    def update(self, agent_state, action, reward):
        q_old = agent_state.Q[action]
        n_old = agent_state.N[action]
        q_new = agent_state.Q.at[action].set(q_old + (reward - q_old)/(n_old + 1))
        n_new = agent_state.N.at[action].set(n_old + 1)
        return AgentState(Q=q_new, N=n_new)