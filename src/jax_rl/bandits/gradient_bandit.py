from typing import NamedTuple
import jax
import jax.numpy as jnp


class GradientBanditState(NamedTuple):
    H: jax.Array
    avg_reward: jax.Array
    N: jax.Array


class GradientBanditAgent:
    def __init__(self, k=10, alpha=0.1, use_baseline=True):
        self.k = k
        self.alpha = alpha
        self.use_baseline = use_baseline
        

    def init(self):
        # Initialize action preferences H to zeros
        return GradientBanditState(
                H=jnp.zeros(self.k),
                avg_reward=jnp.array(0.0),
                N=jnp.array(0, dtype=jnp.int32)
        )

    def select_action(self, key, agent_state: GradientBanditState):
        # Sample an action from softmax(H)
        H, _, _ = agent_state
        probs = jax.nn.softmax(H)
        action = jax.random.choice(key, a=self.k, p=probs)
        return action

    def update(self, agent_state, action, reward):
        H, avg_reward, N = agent_state
        probs = jax.nn.softmax(H)
        mask = jnp.zeros(self.k).at[action].set(1)
        baseline = avg_reward if self.use_baseline else 0.0
        H = H + self.alpha * (reward - avg_reward) * (mask - probs)
        avg_reward = avg_reward + (reward - baseline) / (N + 1)
        return GradientBanditState(
            H=H,
            avg_reward=avg_reward,
            N=N+1
        )