from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
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



@partial(
    jax.jit,
    static_argnames=("env", "num_episodes", "max_steps"),
)
def _td0_prediction_compiled(
    env,
    policy,
    key,
    num_episodes,
    alpha,
    gamma,
    max_steps,
):
    num_states, num_actions = policy.shape
    V_initial = jnp.zeros(num_states, dtype=jnp.float32)

    def run_episode(carry):
        key, V, failed = carry

        key, reset_key = jax.random.split(key)
        env_state, state = env.reset(reset_key)

        # Loop state: PRNG key, environment state,
        # observation, values, step count, termination.
        loop_state = (
            key,
            env_state,
            state,
            V,
            jnp.array(0),
            jnp.array(False),
        )

        def condition(loop_state):
            _, _, _, _, steps, terminated = loop_state
            return (steps < max_steps) & (~terminated)

        def step(loop_state):
            key, env_state, state, V, steps, _ = loop_state

            key, action_key, step_key = jax.random.split(
                key, 3
            )

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

            return (
                key,
                next_env_state,
                next_state,
                V,
                steps + 1,
                terminated,
            )

        key, _, _, V, _, terminated = jax.lax.while_loop(
            condition,
            step,
            loop_state,
        )

        failed = failed | (~terminated)
        return key, V, failed

    def episode_step(carry, _):
        # Stop training on subsequent episodes if
        # one episode exceeds the step limit.
        carry = jax.lax.cond(
            carry[2],
            lambda c: c,
            run_episode,
            carry,
        )
        return carry, None

    initial_carry = (
        key,
        V_initial,
        jnp.array(False),
    )

    (_, V, failed), _ = jax.lax.scan(
        episode_step,
        initial_carry,
        xs=None,
        length=num_episodes,
    )

    return V, failed


def td0_prediction_jit(
    env,
    policy,
    key,
    num_episodes=10_000,
    alpha=0.1,
    gamma=1.0,
    max_steps=1000,
):
    """
    Fully JIT-compiled TD(0) prediction.

    Same return type as td0_prediction().
    """
    if num_episodes < 0:
        raise ValueError("num_episodes must be nonnegative")

    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    V, failed = _td0_prediction_compiled(
        env,
        policy,
        key,
        num_episodes,
        alpha,
        gamma,
        max_steps,
    )

    if bool(failed):
        raise RuntimeError(
            f"Episode did not terminate within {max_steps} steps"
        )

    return V
