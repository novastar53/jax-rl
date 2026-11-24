from typing import NamedTuple

import jax
import jax.numpy as jnp


class ESSState(NamedTuple):
    W: jax.Array
    H: jax.Array


def init_ess(num_states, num_actions):
    zeros = jnp.zeros((num_states, num_actions))
    return ESSState(W=zeros, H=zeros)


def update_ess(state, s, a, alpha):
    W = state.W[s, a]
    H = state.H[s, a]

    new_W = (1 - alpha) * W + alpha
    new_H = (1 - alpha) ** 2 * H + alpha**2

    return ESSState(
        W=state.W.at[s, a].set(new_W),
        H=state.H.at[s, a].set(new_H),
    )


def effective_sample_size(state):
    return jnp.where(
        state.H > 0,
        state.W**2 / jnp.maximum(state.H, 1e-12),
        0.0,
    )



from typing import NamedTuple

import jax
import jax.numpy as jnp


class WeightedVarianceState(NamedTuple):
    mean: jax.Array
    M2: jax.Array


def init_weighted_variance(num_states, num_actions):
    zeros = jnp.zeros((num_states, num_actions))

    return WeightedVarianceState(
        mean=zeros,
        M2=zeros,
    )


def update_weighted_variance(state, ess_state, s, a, alpha, target):
    W = ess_state.W[s, a]
    mean = state.mean[s, a]
    M2 = state.M2[s, a]

    new_W = (1 - alpha) * W + alpha

    delta = target - mean
    new_mean = mean + alpha / jnp.maximum(new_W, 1e-12) * delta
    new_M2 = (1 - alpha) * M2 + alpha * delta * (target - new_mean)

    return WeightedVarianceState(
        mean=state.mean.at[s, a].set(new_mean),
        M2=state.M2.at[s, a].set(new_M2),
    )


def weighted_sample_variance(state, ess_state):
    W = ess_state.W
    H = ess_state.H

    denominator = W - H / jnp.maximum(W, 1e-12)

    variance = state.M2 / jnp.maximum(denominator, 1e-12)

    return jnp.where(
        denominator > 1e-12,
        jnp.maximum(variance, 0.0),
        0.0,
    )



def gaussian_weighted_max(
    key,
    q_values,
    standard_errors,
    num_samples=100,
    action_mask=None,
):
    num_actions = q_values.shape[0]

    if action_mask is None:
        action_mask = jnp.ones(num_actions, dtype=bool)

    noise = jax.random.normal(
        key,
        shape=(num_samples, num_actions),
        dtype=q_values.dtype,
    )

    samples = (
        q_values[None, :]
        + standard_errors[None, :] * noise
    )

    samples = jnp.where(
        action_mask[None, :],
        samples,
        -jnp.inf,
    )

    maxima = jnp.max(samples, axis=1, keepdims=True)
    winners = (samples == maxima) & action_mask[None, :]

    probabilities = jnp.mean(
        winners / jnp.maximum(
            jnp.sum(winners, axis=1, keepdims=True), 1
        ),
        axis=0,
    )

    weighted_value = jnp.sum(probabilities * q_values)

    return weighted_value, probabilities



class GaussianWeightedQState(NamedTuple):
    Q: jax.Array
    N: jax.Array
    ess: ESSState
    variance: WeightedVarianceState


def init_gaussian_weighted_q(num_states, num_actions):
    return GaussianWeightedQState(
        Q=jnp.zeros((num_states, num_actions)),
        N=jnp.zeros((num_states, num_actions), dtype=jnp.int32),
        ess=init_ess(num_states, num_actions),
        variance=init_weighted_variance(num_states, num_actions),
    )


def update_gaussian_weighted_q(
    state,
    key,
    s,
    a,
    reward,
    next_s,
    terminated,
    gamma=0.95,
    num_samples=100,
    initial_standard_error=1.0,
    next_action_mask=None,
    fixed_alpha=None,
):
    variance = weighted_sample_variance(
        state.variance, state.ess
    )

    ess = effective_sample_size(state.ess)

    standard_errors = jnp.sqrt(
        variance[next_s] / jnp.maximum(ess[next_s], 1.0)
    )

    standard_errors = jnp.where(
        state.N[next_s] < 2,
        initial_standard_error,
        standard_errors,
    )

    next_value, _ = gaussian_weighted_max(
        key,
        state.Q[next_s],
        standard_errors,
        num_samples=num_samples,
        action_mask=next_action_mask,
    )

    target = reward + gamma * jnp.where(
        terminated, 0.0, next_value
    )

    count = state.N[s, a] + 1
    if fixed_alpha is None:
        alpha = 1.0 / count.astype(state.Q.dtype) ** 0.8
    else:
        alpha = jnp.asarray(fixed_alpha, dtype=state.Q.dtype)

    new_variance = update_weighted_variance(
        state.variance, state.ess, s, a, alpha, target
    )

    new_ess = update_ess(
        state.ess, s, a, alpha
    )

    new_Q = state.Q.at[s, a].add(
        alpha * (target - state.Q[s, a])
    )

    new_N = state.N.at[s, a].set(count)

    return GaussianWeightedQState(
        Q=new_Q,
        N=new_N,
        ess=new_ess,
        variance=new_variance,
    )
