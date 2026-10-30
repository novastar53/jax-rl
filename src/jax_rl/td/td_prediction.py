import jax
import jax.numpy as jnp


def td0_update(
    V,
    state,
    reward,
    next_state,
    terminated,
    alpha=0.1,
    gamma=1.0,
):
    """
    Perform one TD(0) state-value update.

    Returns:
        Updated V, shape (S,).
    """
    target = reward + gamma * (1.0 - terminated) * V[next_state]
    v_old = V[state]
    delta = target - v_old
    V = V.at[state].set(v_old + alpha * delta)
    return V



def td0_prediction(
    env,
    policy,
    key,
    num_episodes=10_000,
    alpha=0.1,
    gamma=1.0,
    max_steps=1000,
):
    """
    TD(0) state-value prediction under a fixed policy.

    Returns:
        V: (S,) estimated state values.
    """
    num_states, num_actions = policy.shape
    V = jnp.zeros(num_states, dtype=jnp.float32)

    for _ in range(num_episodes):
        key, reset_key = jax.random.split(key)
        env_state, state = env.reset(reset_key)

        for _ in range(max_steps):
            key, action_key, step_key = jax.random.split(key, 3)

            action = jax.random.choice(
                action_key,
                a=num_actions,
                p=policy[state],
            )

            next_env_state, next_state, reward, terminated, _ = (
                env.step(step_key, env_state, action)
            )

            V = td0_update(
                V,
                state,
                reward,
                next_state,
                terminated,
                alpha,
                gamma,
            )

            env_state = next_env_state
            state = next_state

            if bool(terminated):
                break
        else:
            raise RuntimeError(
                f"Episode did not terminate within {max_steps} steps"
            )

    return V
