"""Pure-JAX FrozenLake using the repository's UP/RIGHT/DOWN/LEFT action order.

The predefined layouts and slippery-action probabilities match Gymnasium.
The action IDs do NOT match Gymnasium: here 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT.
"""

import jax
import jax.numpy as jnp

from .base import Environment
from .gridworld import Action
from ._stochastic_grid import action_probabilities, build_transition_model, sample_action


MAPS = {
    "4x4": (
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    ),
    "8x8": (
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFHFFFF",
        "FFFHFFFG",
    ),
}


class FrozenLake(Environment):
    def __init__(self, map_name="4x4", desc=None, is_slippery=True, success_rate=1 / 3):
        rows = MAPS[map_name] if desc is None else tuple(desc)
        if not rows or not all(len(row) == len(rows[0]) for row in rows):
            raise ValueError("desc must be a nonempty rectangular layout")
        flattened = "".join(rows)
        if flattened.count("S") != 1 or flattened.count("G") != 1:
            raise ValueError("desc must contain exactly one start and one goal")
        if any(cell not in "SFGH" for cell in flattened):
            raise ValueError("desc contains invalid tiles")
        if not 0 <= success_rate <= 1:
            raise ValueError("success_rate must be between 0 and 1")

        self.desc = rows
        self.rows = len(rows)
        self.cols = len(rows[0])
        self.num_states = self.rows * self.cols
        self.num_actions = len(Action)
        self.start_state = flattened.index("S")
        self.goal_state = flattened.index("G")
        self._terminal_mask = jnp.asarray([tile in "GH" for tile in flattened])
        self._goal_mask = jnp.asarray([tile == "G" for tile in flattened])
        self.is_slippery = bool(is_slippery)
        self.success_rate = float(success_rate)
        self._action_probs = action_probabilities(
            1.0 - self.success_rate if self.is_slippery else 0.0
        )

    def reset(self, key):
        state = jnp.int32(self.start_state)
        return state, state

    def _deterministic_step(self, state, action):
        row, col = state // self.cols, state % self.cols
        new_row = jnp.clip(
            row + jnp.where(action == Action.UP, -1,
                            jnp.where(action == Action.DOWN, 1, 0)),
            0, self.rows - 1,
        )
        new_col = jnp.clip(
            col + jnp.where(action == Action.LEFT, -1,
                            jnp.where(action == Action.RIGHT, 1, 0)),
            0, self.cols - 1,
        )
        next_state = new_row * self.cols + new_col
        was_terminal = self._terminal_mask[state]
        next_state = jnp.where(was_terminal, state, next_state)
        reward = jnp.where(
            was_terminal, 0.0, self._goal_mask[next_state].astype(jnp.float32)
        )
        return next_state.astype(jnp.int32), reward

    def step(self, key, env_state, action):
        actual_action = sample_action(key, action, self._action_probs)
        next_state, reward = self._deterministic_step(env_state, actual_action)
        terminated = self._terminal_mask[next_state]
        return next_state, next_state, reward, terminated, {}

    def transition_model(self):
        """P[s,a,s'], conditional expected reward R[s,a,s']."""
        return build_transition_model(
            self.num_states, self.num_actions, self._action_probs,
            self._deterministic_step,
        )
