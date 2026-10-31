import jax
import jax.numpy as jnp


def sarsa(
    env,
    policy,
    key,
    num_episodes=10_000,
    alpha=0.1,
    gamma=1.0,
    max_steps = 1000,
):
    """
    On-policy SARSA control.

    Returns:
        Q: (S, A) learned action values.
    """
    Q = jnp.zeros((env.num_states, env.num_actions), dtype=jnp.float32)
    for _ in range(num_episodes):
        key, env_key, action_key = jax.random.split(key, 3)
        env_state, observation = env.reset(env_key)
        action = policy.select_action(action_key, Q[observation])
        for _ in range(max_steps):
            key, action_key, env_key = jax.random.split(key, 3)
            old_q = Q[observation, action]
            new_state, new_observation, reward, terminated, _ = env.step(env_key, env_state, action)
            if bool(terminated):
                delta = reward - old_q
                Q = Q.at[observation, action].add(alpha * delta)
                break
            else:
                new_action = policy.select_action(action_key, Q[new_observation])
                delta = reward + gamma * Q[new_observation, new_action] - old_q
                Q = Q.at[observation, action].add(alpha * delta)
            action, env_state, observation = new_action, new_state, new_observation
        else:
            raise RuntimeError(
                f"Episode did not terminate within {max_steps} steps"
            )
    return Q