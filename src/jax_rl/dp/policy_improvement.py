import jax
import jax.numpy as jnp


def greedy_policy_improvement(P, R, V, gamma=1.0):
    """
    Return a greedy policy with respect to V.

    P: (S, A, S) transition probabilities
    R: (S, A, S) transition-conditioned rewards
    V: (S,) state values

    Returns:
        policy: (S, A) action probabilities
    """

    Q = jnp.sum(P * (R + gamma * V[None, None, :]), axis=2) # (S, A)
    max_values = jnp.max(Q, axis=1, keepdims=True)
    greedy_mask = Q == max_values
    policy = greedy_mask / jnp.sum(greedy_mask, axis=1, keepdims=True)
    return policy