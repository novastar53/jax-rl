"""Count-based and uncertainty-based tabular exploration utilities.

These stateless functions accept one state's action values and work with JIT
and vmap. The uncertainty propagation rule is the experiment 8e heuristic; it
is neither a posterior variance nor a theoretical uncertainty Bellman bound.
"""

import jax
import jax.numpy as jnp


def local_ucb_action(key, q_values, counts, coefficient=1.0):
    """Sample uniformly among maximizers of a local UCB score.

    Unvisited actions take precedence, with ties broken uniformly. For visited
    actions the bonus is coefficient * sqrt(log(1 + sum(counts)) / counts[a]).
    Counts refer only to the current state and must be nonnegative.
    """
    counts = jnp.asarray(counts)
    bonus = coefficient * jnp.sqrt(jnp.log1p(jnp.sum(counts)) / jnp.maximum(counts, 1))
    scores = jnp.where(counts == 0, jnp.inf, q_values + bonus)
    winners = scores == jnp.max(scores)
    return jax.random.choice(key, q_values.shape[0], p=winners / jnp.sum(winners))


def gaussian_thompson_action(key, q_values, uncertainty, beta=1.0):
    """Select argmax(Q + beta * sqrt(uncertainty) * Z), with independent Z.

    Uncertainty is a nonnegative variance-like proxy, not necessarily a
    posterior variance. Negative inputs are clipped to zero. Score ties use
    the first maximum, matching the experiment implementation.
    """
    q_values = jnp.asarray(q_values)
    noise = jax.random.normal(key, q_values.shape, dtype=q_values.dtype)
    scores = q_values + beta * jnp.sqrt(jnp.maximum(uncertainty, 0.0)) * noise
    return jnp.argmax(scores)


def count_uncertainty(counts, prior_uncertainty=4.0):
    """Return the experiment's count-based proxy, prior_uncertainty / (N + 1)."""
    return prior_uncertainty / (jnp.asarray(counts) + 1.0)


def propagated_uncertainty(
    count, successor_uncertainty, terminated, *, gamma=0.99, prior_uncertainty=4.0,
    reward_noise_var=1.0, eta=0.05,
):
    """Compute the experiment 8e propagated-uncertainty exploration heuristic.

    `count` is the visit count *after* the current transition. The caller
    chooses the successor action and its uncertainty before applying this
    update. Terminal transitions use zero successor uncertainty. The rule is
    (1-eta) * prior/(count+1) + eta * (reward_noise_var + gamma**2 * child).
    It blends local and successor proxies; it is not a UBE variance bound and
    does not account for epistemic uncertainty in transition probabilities.
    """
    child = jnp.where(terminated, 0.0, successor_uncertainty)
    return (1.0 - eta) * count_uncertainty(count, prior_uncertainty) + eta * (
        reward_noise_var + gamma**2 * child
    )
