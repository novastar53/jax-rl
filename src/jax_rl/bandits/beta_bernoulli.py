from typing import NamedTuple

import jax
import jax.numpy as jnp


class BetaBernoulliState(NamedTuple):
    alpha: jax.Array
    beta: jax.Array


class BetaBernoulliAgent:
    def __init__(self, k=10):
        self.k = k

    def init(self):
        state = BetaBernoulliState(alpha=jnp.ones(self.k, dtype=jnp.float32), beta=jnp.ones(self.k, dtype=jnp.float32))
        return state

    def update(self, agent_state, action, reward):
        alpha_new = agent_state.alpha.at[action].add(reward)
        beta_new = agent_state.beta.at[action].add(1 - reward)

        return BetaBernoulliState(
            alpha=alpha_new,
            beta=beta_new,
        )

    def select_action(self, key, agent_state):
        # Sample one success probability from
        # the posterior distribution of each arm
        probs = jax.random.beta(key, agent_state.alpha, agent_state.beta)

        # Select the arm with the highest sample
        action = jnp.argmax(probs)
        return action