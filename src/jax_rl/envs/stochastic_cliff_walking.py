"""Cliff walking with configurable action slippage and reward variance."""

import jax
import jax.numpy as jnp

from .cliff_walking import CliffWalking
from ._stochastic_grid import action_probabilities, build_transition_model, sample_action


class StochasticCliffWalking(CliffWalking):
    def __init__(self, slip_prob=0.2, reward_std=1.0):
        super().__init__()
        if reward_std < 0:
            raise ValueError("reward_std must be nonnegative")
        self.slip_prob = float(slip_prob)
        self.reward_std = float(reward_std)
        self._action_probs = action_probabilities(self.slip_prob)

    def step(self, key, env_state, action):
        move_key, reward_key = jax.random.split(key)
        actual_action = sample_action(move_key, action, self._action_probs)
        next_state, observation, base_reward, terminated, info = super().step(
            move_key, env_state, actual_action
        )
        was_terminal = env_state == self.goal_state
        reward = base_reward + jnp.where(
            was_terminal, 0.0,
            self.reward_std * jax.random.normal(reward_key),
        )
        return next_state, observation, reward, terminated, info

    def _deterministic_step(self, state, action):
        next_state, _, reward, _, _ = CliffWalking.step(self, None, state, action)
        return next_state, reward

    def transition_model(self):
        """P[s,a,s'], conditional expected reward R[s,a,s']."""
        return build_transition_model(
            self.num_states,
            self.num_actions,
            self._action_probs,
            self._deterministic_step,
        )
