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



def iterative_policy_evaluation(
    P,
    R,
    policy,
    gamma=1.0,
    theta=1e-6,
    max_iterations=10000,
):
    """
    Repeatedly apply the Bellman expectation operator.

    Stop when max(abs(V_new - V)) < theta,
    or when max_iterations is reached.

    Returns:
        V: Converged value estimates, shape (S,).
        iterations: Number of Bellman sweeps performed.
    """
    V = jnp.zeros((policy.shape[0]))
    V_new = bellman_sweep(P, R, policy, V, gamma)
    iters = 1
    while jnp.max(jnp.abs(V_new - V)) >= theta and iters < max_iterations:
        V = V_new
        V_new = bellman_sweep(P, R, policy, V_new, gamma)
        iters += 1

    return V_new, iters