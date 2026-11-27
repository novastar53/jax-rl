import jax
import jax.numpy as jnp


def exact_discounted_policy_evaluation(P, R, policy, gamma=0.99):
    """Solve the finite discounted policy Bellman equations.

    P and R have shape (states, actions, states), with R the conditional
    expected transition reward; policy has shape (states, actions). For
    0 <= gamma < 1, return V = (I - gamma P_pi)^-1 r_pi. Absorbing terminal
    states must have zero reward; alternatively their policy row may be zero,
    in which case their value is fixed at zero. This is a model-based evaluator,
    not a learning update. Results inherit the input's floating-point precision.
    """
    p_pi = jnp.einsum("sa,sat->st", policy, P)
    r_pi = jnp.einsum("sa,sat,sat->s", policy, P, R)
    return jnp.linalg.solve(jnp.eye(P.shape[0], dtype=p_pi.dtype) - gamma * p_pi, r_pi)


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
