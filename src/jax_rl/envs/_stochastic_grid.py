"""Shared stochastic cardinal-action dynamics and exact tabular models."""

import jax
import jax.numpy as jnp


def action_probabilities(slip_prob):
    if not 0.0 <= slip_prob <= 1.0:
        raise ValueError("slip_prob must be between 0 and 1")
    return jnp.array([1.0 - slip_prob, slip_prob / 2.0, slip_prob / 2.0])


def possible_actions(action):
    """Intended, left-turn, right-turn in UP/RIGHT/DOWN/LEFT ordering."""
    return jnp.stack([action, (action - 1) % 4, (action + 1) % 4])


def sample_action(key, action, probs):
    index = jax.random.categorical(key, jnp.log(probs))
    return possible_actions(action)[index]


def build_transition_model(num_states, num_actions, probs, deterministic_step):
    """Return P[s,a,s'] and E[R | s,a,s'] (zero if P is zero).

    deterministic_step(state, actual_action) returns (next_state, mean_reward).
    Noise added independently with zero mean does not affect either array.
    """
    states = jnp.arange(num_states, dtype=jnp.int32)
    actions = jnp.arange(num_actions, dtype=jnp.int32)

    def outcomes(state, action):
        next_states, rewards = jax.vmap(
            lambda actual_action: deterministic_step(state, actual_action)
        )(possible_actions(action))
        return next_states, rewards

    next_states, rewards = jax.vmap(
        lambda state: jax.vmap(lambda action: outcomes(state, action))(actions)
    )(states)

    transitions = jax.nn.one_hot(next_states, num_states)
    weighted = transitions * probs[None, None, :, None]
    P = jnp.sum(weighted, axis=2)
    reward_mass = jnp.sum(weighted * rewards[..., None], axis=2)
    R = jnp.where(P > 0, reward_mass / jnp.where(P > 0, P, 1), 0.0)
    return P, R
