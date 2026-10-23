from enum import IntEnum

import jax
import jax.numpy as jnp

from . import Environment


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Gridworld(Environment):
    def __init__(self, size=4):
        self.size = size
        self.num_states = size ** 2
        self.num_actions = len(Action)

    def reset(self, key):
        env_state = jnp.array(5, dtype=jnp.int32)
        observation = env_state
        return env_state, observation

    def step(self, key, env_state, action):
        row = env_state // self.size
        col = env_state % self.size

        # Calculate movement for each action
        new_row = row + jnp.where(
            action == Action.UP, -1,
            jnp.where(action == Action.DOWN, 1, 0)
        )

        new_col = col + jnp.where(
            action == Action.LEFT, -1,
            jnp.where(action == Action.RIGHT, 1, 0)
        )

        # Prevent movement outside the grid
        new_row = jnp.clip(new_row, 0, self.size - 1)
        new_col = jnp.clip(new_col, 0, self.size - 1)

        new_state = new_row * self.size + new_col

        # Terminal states are absorbing
        was_terminal = (env_state == 0) | (
            env_state == self.num_states - 1
        )

        new_state = jnp.where(was_terminal, env_state, new_state)

        # Reward is -1 for every step from a nonterminal state
        reward = jnp.where(was_terminal, 0.0, -1.0)

        # Check whether the new state is terminal
        terminated = (new_state == 0) | (
            new_state == self.num_states - 1
        )

        observation = new_state
        info = {}

        return new_state, observation, reward, terminated, info

