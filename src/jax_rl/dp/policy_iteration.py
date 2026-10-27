import jax.numpy as jnp

from .policy_evaluation import iterative_policy_evaluation
from .policy_improvement import greedy_policy_improvement


def policy_iteration(
    P,
    R,
    gamma=1.0,
    theta=1e-6,
    max_iterations=100,
):
    """
    Perform policy iteration on a finite MDP.

    Args:
        P: (S, A, S) transition probabilities.
        R: (S, A, S) transition-conditioned rewards.
        gamma: Discount factor.
        theta: Policy evaluation convergence threshold.
        max_iterations: Maximum policy improvement iterations.

    Returns:
        V: Final state-value estimates, shape (S,).
        policy: Final policy, shape (S, A).
        iterations: Number of policy improvement iterations.
    """
    num_states, num_actions, _ = P.shape

    # Initialize uniform random policy.
    policy = jnp.full(
        (num_states, num_actions),
        1.0 / num_actions,
    )

    for iteration in range(1, max_iterations + 1):

        # 1. Policy evaluation
        V, _ = iterative_policy_evaluation(
            P,
            R,
            policy,
            gamma=gamma,
            theta=theta,
        )

        # 2. Greedy policy improvement
        new_policy = greedy_policy_improvement(
            P,
            R,
            V,
            gamma=gamma,
        )

        # 3. Check policy stability
        if jnp.allclose(new_policy, policy):
            return V, policy, iteration

        policy = new_policy

    # Evaluate the final improved policy before returning.
    V, _ = iterative_policy_evaluation(
        P,
        R,
        policy,
        gamma=gamma,
        theta=theta,
    )

    return V, policy, max_iterations
