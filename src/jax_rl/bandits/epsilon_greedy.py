import jax
import jax.numpy as jnp

class EpsilonGreedyPolicy:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon


    def select_action(self, key, Q):
        k = Q.shape[0]
        greedy_mask = (Q == jnp.max(Q))
        greedy_probs = greedy_mask / jnp.sum(greedy_mask)
        probs = self.epsilon / k + (1 - self.epsilon) * greedy_probs
        action = jax.random.choice(key, a=k, p=probs)
        return action

