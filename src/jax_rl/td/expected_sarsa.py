import jax
import jax.numpy as jnp


def expected_sarsa_update(
    Q,
    state,
    action,
    reward,
    next_state,
    terminated,
    policy,
    alpha=0.1,
    gamma=1.0,
):
    """
    Perform one Expected SARSA update.

    Args:
        Q: (S, A) action-value table.
        state: Current state.
        action: Action taken.
        reward: Observed reward.
        next_state: Next state.
        terminated: Whether the episode ended.
        policy: Policy exposing action_probs(q_values).
        alpha: Learning rate.
        gamma: Discount factor.

    Returns:
        Updated Q.
    """
    if terminated:
        delta_t = reward - Q[state, action]
    else:
        delta_t = reward + gamma * jnp.dot(policy.action_probs(Q[next_state]), Q[next_state]) - Q[state, action]

    Q = Q.at[state, action].add(alpha * delta_t)
    return Q




def expected_sarsa(
    env,
    policy,
    key,
    num_episodes=10_000,
    alpha=0.1,
    gamma=1.0,
    max_steps=1000,
):
    """
    On-policy Expected SARSA control.

    Args:
        env: Episodic environment.
        policy: Object exposing select_action() and action_probs().
        key: JAX PRNG key.
        num_episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        max_steps: Maximum transitions per episode.

    Returns:
        Q: (S, A) learned action values.
    """
    Q = jnp.zeros((env.num_states, env.num_actions), dtype=jnp.float32)
    for _ in range(num_episodes):
        key, env_key, action_key = jax.random.split(key, 3)
        env_state, observation = env.reset(env_key)
        action = policy.select_action(action_key, Q[observation])
        for _ in range(max_steps):
            key, env_key, action_key = jax.random.split(key, 3)
            new_state, new_observation, reward, terminated, _ = env.step(env_key, env_state, action)
            Q = expected_sarsa_update(Q, observation, action, reward, new_observation, terminated, policy, alpha, gamma)
            if bool(terminated):
                break
            env_state, observation = new_state, new_observation
            action = policy.select_action(action_key, Q[observation])
        else:
            raise RuntimeError(
                    f"Episode did not terminate within {max_steps} steps"
                )
    return Q