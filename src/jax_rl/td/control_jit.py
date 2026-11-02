from functools import partial

import jax
import jax.numpy as jnp


@partial(
    jax.jit,
    static_argnames=(
        "env",
        "policy",
        "num_episodes",
        "max_steps",
        "algorithm",
        "truncate",
    ),
)
def _train_control(
    env,
    policy,
    key,
    num_episodes,
    alpha,
    gamma,
    max_steps,
    algorithm,
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

            if algorithm == "sarsa":
                # SARSA uses the next action chosen by
                # the current behavior policy.
                def nonterminal_target(_):
                    next_action = policy.select_action(
                        action_key, Q[next_state]
                    )
                    target = reward + gamma * Q[
                        next_state, next_action
                    ]
                    return target, next_action

                def terminal_target(_):
                    return reward, jnp.int32(0)

                target, next_action = jax.lax.cond(
                    terminated,
                    terminal_target,
                    nonterminal_target,
                    operand=None,
                )

                Q = Q.at[state, action].add(
                    alpha * (target - Q[state, action])
                )

            else:
                # Q-learning bootstraps from the
                # maximum next-state action value.
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


def train_control_jit(
    env,
    policy,
    key,
    num_episodes=10_000,
    alpha=0.1,
    gamma=1.0,
    max_steps=1000,
    algorithm="sarsa",
    return_history=False,
    truncate=False,
):
    if algorithm not in ("sarsa", "q_learning"):
        raise ValueError("Unknown control algorithm")

    if num_episodes < 0:
        raise ValueError("num_episodes must be nonnegative")

    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    Q, episode_returns, failed = _train_control(
        env,
        policy,
        key,
        num_episodes,
        alpha,
        gamma,
        max_steps,
        algorithm,
        truncate,
    )

    if bool(failed) and not truncate:
        raise RuntimeError(
            f"Episode did not terminate within {max_steps} steps"
        )

    if return_history:
        return Q, episode_returns

    return Q
