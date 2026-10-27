import jax
import jax.numpy as jnp

from .policy_improvement import greedy_policy_improvement


def value_iteration(
    P,
    R,
    gamma=1.0,
    tolerance=1e-8,
    max_iterations=10_000,
):
    """
    Compute the optimal value function using value iteration.

    Returns:
        V:      (S,) optimal state values
        policy: (S, A) greedy policy with respect to V
    """
    num_states, num_actions, _ = P.shape
    V = jnp.zeros(num_states,) # (S,)
    for _ in range(max_iterations):
        Q = jnp.sum(P * (R + gamma * V[None, None, :]), axis=2) # (S,A)
        V_new = jnp.max(Q, axis=1)
        if jnp.max(jnp.abs(V_new - V)) < tolerance:
            V = V_new
            break
        V = V_new
    policy = greedy_policy_improvement(
        P, R, V, gamma=gamma
    ) 
    return V, policy