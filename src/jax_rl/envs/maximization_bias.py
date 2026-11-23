from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class MaximizationBias:
    """
    Sutton & Barto's maximization-bias MDP.

    State 0 (A):
        Action 0: Left  -> B, reward 0
        Action 1: Right -> Terminal, reward 0

    State 1 (B):
        Actions 0-9 -> Terminal
        Reward ~ Normal(-0.1, 1.0)

    State 2: Terminal
    """

    num_states: int = 3
    num_actions: int = 10

    def reset(self, key):
        state = jnp.int32(0)
        return state, state

    def step(self, key, env_state, action):
        at_start = env_state == 0
        go_left = action == 0

        next_state = jnp.where(
            at_start,
            jnp.where(go_left, 1, 2),
            2,
        ).astype(jnp.int32)

        noisy_reward = (
            -0.1 + jax.random.normal(key)
        )

        reward = jnp.where(
            at_start | (env_state == 2),
            0.0,
            noisy_reward,
        )

        terminated = next_state == 2

        return (
            next_state,
            next_state,
            reward,
            terminated,
            {},
        )

    def action_mask(self, state):
        actions = jnp.arange(self.num_actions)

        return jnp.where(
            state == 0,
            actions < 2,
            actions < self.num_actions,
        )
