import jax
import jax.numpy as jnp


def build_deterministic_model(env, key):
    """
    Convert a deterministic, finite environment into
    transition and reward tensors.

    Returns:
        P: (S, A, S) transition probabilities
        R: (S, A, S) expected transition rewards
    """
    P = jnp.zeros((env.num_states, env.num_actions, env.num_states))
    R = jnp.zeros((env.num_states, env.num_actions, env.num_states))

    for state in range(env.num_states):
        for action in range(env.num_actions):
            _, observation, reward, _, _ = env.step(key, state, action)
            P = P.at[state, action, observation].set(1)
            R = R.at[state, action, observation].set(reward)

    return P, R