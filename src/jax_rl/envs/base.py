from abc import ABC, abstractmethod
from typing import Any


class Environment(ABC):
    @abstractmethod
    def reset(self, key) -> Any:
        """
        Reset the environment and return the initial observation/state.

        Returns:
            env_state
            observation
        """
        ...

    @abstractmethod
    def step(self, key, env_state, action):
        """
        Apply an action to the environment.

        Returns:
            new_env_state
            observation
            reward
            terminated
            info
        """
        ...