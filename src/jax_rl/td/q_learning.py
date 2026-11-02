import jax
import jax.numpy as jnp


def q_learning(
    env,
    policy,
    key,
    num_episodes=10_000,
    alpha=0.1,
    gamma=1.0,
    max_steps=1000,
):
    """
    Off-policy tabular Q-learning control.

    Args:
        env: Episodic environment.
        policy: Behavior policy with select_action(key, q_values).
        key: JAX PRNG key.
        num_episodes: Number of episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        max_steps: Maximum transitions per episode.

    Returns:
        Q: (S, A) learned action values.
    """
    Q = jnp.zeros(shape=(env.num_states, env.num_actions), dtype=jnp.float32)
    for _ in range(num_episodes):
        key, env_key, action_key = jax.random.split(key, 3)
        env_state, observation = env.reset(env_key)
        action = policy.select_action(action_key, Q[observation])
        for _ in range(max_steps):
            key, env_key, action_key = jax.random.split(key, 3)
            new_state, new_observation, reward, terminated, _ = env.step(env_key, env_state, action)
            q_old = Q[observation, action]
            if bool(terminated):
                delta = alpha * (reward - q_old)
                Q = Q.at[observation, action].add(delta) 
                break
            else:
                delta = alpha * (reward + gamma * jnp.max(Q[new_observation]) - q_old)
                Q = Q.at[observation, action].add(delta) 
            env_state, observation = new_state, new_observation
            action = policy.select_action(action_key, Q[observation])
        else:
            raise RuntimeError(
                f"Episode did not terminate within {max_steps} steps"
            )
    return Q




from jax_rl.td.control_jit import train_control_jit


def q_learning_jit(
    env,
    policy,
    key,
    num_episodes=10_000,
    alpha=0.1,
    gamma=1.0,
    max_steps=1000,
    return_history=False,
    truncate=False,
):
    """
    JIT-compiled off-policy Q-learning control.

    Returns:
        Q, or (Q, episode_returns) if return_history=True.
    """
    return train_control_jit(
        env,
        policy,
        key,
        num_episodes=num_episodes,
        alpha=alpha,
        gamma=gamma,
        max_steps=max_steps,
        algorithm="q_learning",
        return_history=return_history,
        truncate=truncate,
    )
