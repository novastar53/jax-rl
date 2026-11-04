from functools import partial

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



@partial(
    jax.jit,
    static_argnames=(
        "env",
        "policy",
        "num_episodes",
        "max_steps",
        "truncate",
    ),
)
def _train_q_learning(
    env,
    policy,
    key,
    num_episodes,
    alpha,
    gamma,
    max_steps,
    truncate,
):
    Q = jnp.zeros(
        (env.num_states, env.num_actions),
        dtype=jnp.float32,
    )

    def run_episode(carry):
        key, Q, failed = carry

        key, reset_key, action_key = jax.random.split(key, 3)
        env_state, state = env.reset(reset_key)

        action = policy.select_action(
            action_key, Q[state]
        )

        initial = (
            key,
            env_state,
            state,
            action,
            Q,
            jnp.int32(0),
            jnp.float32(0.0),
            jnp.bool_(False),
        )

        def condition(c):
            _, _, _, _, _, steps, _, terminated = c
            return (steps < max_steps) & (~terminated)

        def step(c):
            (
                key,
                env_state,
                state,
                action,
                Q,
                steps,
                total_reward,
                _,
            ) = c

            key, step_key, action_key = jax.random.split(
                key, 3
            )

            next_env_state, next_state, reward, terminated, _ = (
                env.step(step_key, env_state, action)
            )


            target = jnp.where(
                terminated,
                reward,
                reward + gamma * jnp.max(Q[next_state]),
            )

            Q = Q.at[state, action].add(
                alpha * (target - Q[state, action])
            )

            # Choose the next behavior action
            # after updating Q.
            next_action = policy.select_action(
                action_key, Q[next_state]
            )

            return (
                key,
                next_env_state,
                next_state,
                next_action,
                Q,
                steps + 1,
                total_reward + reward,
                terminated,
            )

        (
            key,
            _,
            _,
            _,
            Q,
            _,
            total_reward,
            terminated,
        ) = jax.lax.while_loop(
            condition,
            step,
            initial,
        )

        failed = failed | (~terminated)

        return (key, Q, failed), total_reward

    def episode_step(carry, _):
        # Preserve the reference implementation's
        # failure behavior: stop after an episode
        # exceeds max_steps.
        if truncate:
            return run_episode(carry)

        return jax.lax.cond(
            carry[2],
            lambda c: (c, jnp.float32(0.0)),
            run_episode,
            carry,
        )

    initial_carry = (
        key,
        Q,
        jnp.bool_(False),
    )

    (_, Q, failed), episode_returns = jax.lax.scan(
        episode_step,
        initial_carry,
        xs=None,
        length=num_episodes,
    )

    return Q, episode_returns, failed


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
    if num_episodes < 0:
        raise ValueError("num_episodes must be nonnegative")

    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    Q, episode_returns, failed = _train_q_learning(
        env,
        policy,
        key,
        num_episodes,
        alpha,
        gamma,
        max_steps,
        truncate,
    )

    if bool(failed) and not truncate:
        raise RuntimeError(
            f"Episode did not terminate within {max_steps} steps"
        )

    if return_history:
        return Q, episode_returns

    return Q
