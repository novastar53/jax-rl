import jax
import jax.numpy as jnp


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
    Q = jnp.sum(P * (R + gamma * V[None, None, :]), axis=2) # (S,A)
    for _ in range(max_iterations):
        V_new = jnp.max(Q, axis=1)
        if jnp.max(jnp.abs(V_new - V)) < tolerance:
            V = V_new
            break
        V = V_new
        Q = jnp.sum(P * (R + gamma * V[None, None, :]), axis=2) # (S,A)
    Q = jnp.sum(P * (R + gamma * V[None, None, :]), axis=2) # (S,A)
    max_values = jnp.max(Q, axis=1, keepdims=True)
    greedy_mask = Q == max_values
    policy = greedy_mask / jnp.sum(greedy_mask, axis=1, keepdims=True)
    return V, policy