import jax
import jax.numpy as jnp


def generate_episode(env, policy, key, max_steps=1000):
    """
    Generate an episode using a fixed policy.

    Returns:
        states:  (T,)
        actions: (T,)
        rewards: (T,)
    """
    states, actions, rewards = [], [], []
    key, reset_key = jax.random.split(key)
    env_state, cur_state = env.reset(reset_key)
    for _ in range(max_steps):
        states.append(cur_state)
        key, action_key, env_key = jax.random.split(key, 3)
        action = jax.random.choice(
                    action_key,
                    a=policy.shape[1],
                    p=policy[cur_state],
        )
        actions.append(action)
        env_state, cur_state, reward, terminated, _ = env.step(env_key, env_state, action)
        rewards.append(reward)
        if terminated:
            break

    if not bool(terminated):
        raise RuntimeError(
        f"Episode did not terminate within {max_steps} steps"
    )

    return (
        jnp.asarray(states),
        jnp.asarray(actions),
        jnp.asarray(rewards),
    )