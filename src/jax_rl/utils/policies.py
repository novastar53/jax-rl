import jax
import jax.numpy as jnp


class EpsilonGreedyPolicy:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def action_probs(self, q_values):
        greedy = q_values == jnp.max(q_values)
        greedy_probs = greedy / jnp.sum(greedy)

        num_actions = q_values.shape[0]

        return (
            self.epsilon / num_actions
            + (1.0 - self.epsilon) * greedy_probs
        )

    def select_action(self, key, q_values):
        probs = self.action_probs(q_values)

        return jax.random.choice(
            key,
            a=q_values.shape[0],
            p=probs,
        )
