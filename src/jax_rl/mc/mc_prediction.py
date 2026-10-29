import jax
import jax.numpy as jnp

from jax_rl.mc.episode import generate_episode


def compute_returns(rewards, gamma=1.0):
    """Compute discounted returns for a complete episode."""
    returns = jnp.zeros_like(rewards, dtype=jnp.float32)
    G = 0.0

    for t in range(len(rewards) - 1, -1, -1):
        G = rewards[t] + gamma * G
        returns = returns.at[t].set(G)

    return returns


def update_mc_values(V, N, states, returns, visit_mode="first"):
    """
    Update MC state-value estimates from one complete episode.

    V: (S,) current state-value estimates
    N: (S,) current update counts
    states: (T,) states visited before each action
    returns: (T,) corresponding returns
    visit_mode: "first" or "every"

    Returns:
        Updated (V, N).
    """
    if visit_mode not in ("first", "every"):
        raise ValueError("visit_mode must be 'first' or 'every'")

    seen_states = set()

    for state, G in zip(states, returns):
        s = int(state)

        if visit_mode == "first":
            if s in seen_states:
                continue
            seen_states.add(s)

        count = N[s] + 1

        V = V.at[s].set(
            V[s] + (G - V[s]) / count
        )
        N = N.at[s].set(count)

    return V, N


def mc_prediction(
    env,
    policy,
    key,
    num_episodes=10000,
    gamma=1.0,
    visit_mode="first",
    callback=None,
    callback_every=50,
):
    """
    Monte Carlo state-value prediction.

    Returns:
        V: (S,) estimated state values
        N: (S,) update counts
    """
    if visit_mode not in ("first", "every"):
        raise ValueError("visit_mode must be 'first' or 'every'")

    if num_episodes < 0:
        raise ValueError("num_episodes must be nonnegative")

    if callback is not None and callback_every < 1:
        raise ValueError("callback_every must be positive")

    num_states = policy.shape[0]

    V = jnp.zeros(num_states, dtype=jnp.float32)
    N = jnp.zeros(num_states, dtype=jnp.int32)

    for episode in range(1, num_episodes + 1):
        key, episode_key = jax.random.split(key)

        states, _, rewards = generate_episode(
            env, policy, episode_key
        )

        returns = compute_returns(rewards, gamma)

        V, N = update_mc_values(
            V, N, states, returns, visit_mode
        )

        if callback is not None and (
            episode % callback_every == 0
            or episode == num_episodes
        ):
            callback(episode, V, N)

    return V, N
