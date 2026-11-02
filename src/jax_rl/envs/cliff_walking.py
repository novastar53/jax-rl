import jax.numpy as jnp

from . import Environment
from .gridworld import Action


class CliffWalking(Environment):
    def __init__(self):
        self.rows = 4
        self.cols = 12

        self.num_states = self.rows * self.cols
        self.num_actions = len(Action)

        self.start_state = 3 * self.cols
        self.goal_state = self.num_states - 1

    def reset(self, key):
        state = jnp.int32(self.start_state)
        return state, state

    def step(self, key, env_state, action):
        row = env_state // self.cols
        col = env_state % self.cols

        new_row = row + jnp.where(
            action == Action.UP, -1,
            jnp.where(action == Action.DOWN, 1, 0),
        )

        new_col = col + jnp.where(
            action == Action.LEFT, -1,
            jnp.where(action == Action.RIGHT, 1, 0),
        )

        new_row = jnp.clip(new_row, 0, self.rows - 1)
        new_col = jnp.clip(new_col, 0, self.cols - 1)

        next_state = new_row * self.cols + new_col

        # The cliff occupies the bottom row except
        # for the start and goal positions.
        fell = (
            (new_row == self.rows - 1)
            & (new_col > 0)
            & (new_col < self.cols - 1)
        )

        # Falling returns the agent to the start.
        next_state = jnp.where(
            fell,
            self.start_state,
            next_state,
        )

        reward = jnp.where(fell, -100.0, -1.0)
        terminated = next_state == self.goal_state

        # Keep the goal absorbing if stepped after termination.
        was_terminal = env_state == self.goal_state
        next_state = jnp.where(
            was_terminal, self.goal_state, next_state
        )
        reward = jnp.where(was_terminal, 0.0, reward)
        terminated = terminated | was_terminal

        return (
            next_state,
            next_state,
            reward,
            terminated,
            {},
        )
