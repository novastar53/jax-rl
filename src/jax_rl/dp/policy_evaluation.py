import jax
import jax.numpy as jnp


def bellman_sweep(P, R, policy, V, gamma=1.0):
    """
    P: (S, A, S) transition probabilities
    R: (S, A, S) expected transition rewards
    policy: (S, A) action probabilities
    V: (S,) current value estimates 
    """

    Q = jnp.sum(
        P * (R + gamma * V[None, None, :]),
        axis=2,
    )

    V_new = jnp.sum(policy * Q, axis=1)
    return V_new