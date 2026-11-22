import jax
import jax.numpy as jnp


def q_sigma_update(
    Q,
    state,
    action,
    reward,
    next_state,
    next_action,
    terminated,
    policy,
    sigma=0.5,
    alpha=0.1,
    gamma=1.0,
):
    """
    Perform a one-step Q(sigma) update.

    Args:
        Q: (S, A) action-value estimates.
        state: Current state.
        action: Current action.
        reward: Observed reward.
        next_state: Next state.
        next_action: Action selected at next_state.
        terminated: Whether the transition is terminal.
        policy: Policy exposing action_probs(q_values).
        sigma: Sampling coefficient in [0, 1].
        alpha: Learning rate.
        gamma: Discount factor.

    Returns:
        Updated Q, shape (S, A).
    """
    