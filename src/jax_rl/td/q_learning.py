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
    decay_exponent=None,
):
    Q = jnp.zeros(
        (env.num_states, env.num_actions),
        dtype=jnp.float32,
    )
    N = jnp.zeros_like(Q, dtype=jnp.int32)

    for _ in range(num_episodes):
        key, env_key, action_key = jax.random.split(key, 3)
        env_state, observation = env.reset(env_key)
        action = policy.select_action(action_key, Q[observation])

        for _ in range(max_steps):
            key, env_key, action_key = jax.random.split(key, 3)

            new_state, new_observation, reward, terminated, _ = (
                env.step(env_key, env_state, action)
            )

            count = N[observation, action] + 1
            N = N.at[observation, action].set(count)

            if decay_exponent is None:
                step_alpha = alpha
            else:
                step_alpha = count.astype(Q.dtype) ** (-decay_exponent)

            target = reward
            if not bool(terminated):
                target += gamma * jnp.max(Q[new_observation])

            Q = Q.at[observation, action].add(
                step_alpha * (target - Q[observation, action])
            )

            if bool(terminated):
                break

            env_state, observation = new_state, new_observation
            action = policy.select_action(
                action_key, Q[observation]
            )
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
        "decay_exponent",
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
    decay_exponent,
):
    Q = jnp.zeros(
        (env.num_states, env.num_actions),
        dtype=jnp.float32,
    )
    N = jnp.zeros_like(Q, dtype=jnp.int32)

    def run_episode(carry):
        key, Q, N, failed = carry

        key, reset_key, action_key = jax.random.split(key, 3)
        env_state, state = env.reset(reset_key)

        action = policy.select_action(action_key, Q[state])

        initial = (
            key,
            env_state,
            state,
            action,
            Q,
            N,
            jnp.int32(0),
            jnp.float32(0.0),
            jnp.bool_(False),
        )

        def condition(c):
            return (c[6] < max_steps) & (~c[8])

        def step(c):
            (
                key, env_state, state, action,
                Q, N, steps, total_reward, _,
            ) = c

            key, step_key, action_key = jax.random.split(key, 3)

            next_env_state, next_state, reward, terminated, _ = (
                env.step(step_key, env_state, action)
            )

            target = jnp.where(
                terminated,
                reward,
                reward + gamma * jnp.max(Q[next_state]),
            )

            count = N[state, action] + 1

            if decay_exponent is None:
                step_alpha = alpha
            else:
                step_alpha = count.astype(Q.dtype) ** (-decay_exponent)

            Q = Q.at[state, action].add(
                step_alpha * (target - Q[state, action])
            )
            N = N.at[state, action].set(count)

            next_action = policy.select_action(
                action_key, Q[next_state]
            )

            return (
                key,
                next_env_state,
                next_state,
                next_action,
                Q,
                N,
                steps + 1,
                total_reward + reward,
                terminated,
            )

        (
            key, _, _, _, Q, N, _,
            total_reward, terminated,
        ) = jax.lax.while_loop(condition, step, initial)

        failed = failed | (~terminated)

        return (key, Q, N, failed), total_reward

    def episode_step(carry, _):
        if truncate:
            return run_episode(carry)

        return jax.lax.cond(
            carry[3],
            lambda c: (c, jnp.float32(0.0)),
            run_episode,
            carry,
        )

    (_, Q, N, failed), episode_returns = jax.lax.scan(
        episode_step,
        (key, Q, N, jnp.bool_(False)),
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
    decay_exponent=None,
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
        decay_exponent,
    )

    if bool(failed) and not truncate:
        raise RuntimeError(
            f"Episode did not terminate within {max_steps} steps"
        )

    if return_history:
        return Q, episode_returns

    return Q



def double_q_learning_update(
    Q1,
    Q2,
    state,
    action,
    reward,
    next_state,
    terminated,
    update_first,
    alpha=0.1,
    gamma=1.0,
):
    """
    Perform one Double Q-learning update.

    Args:
        Q1, Q2: (S, A) action-value tables.
        state: Current state.
        action: Action taken.
        reward: Observed reward.
        next_state: Next state.
        terminated: Whether the transition is terminal.
        update_first: If True update Q1, otherwise update Q2.
        alpha: Learning rate.
        gamma: Discount factor.

    Returns:
        Updated (Q1, Q2).
    """
    if update_first:
        Q_update = Q1
        Q_lookup = Q2
    else:
        Q_update = Q2
        Q_lookup = Q1

    if terminated:
        delta = alpha * (reward - Q_update[state, action])
    else:
        next_action = jnp.argmax(Q_update[next_state])
        delta = alpha * (reward + gamma * Q_lookup[next_state, next_action] - Q_update[state, action])

    Q_update = Q_update.at[state, action].add(delta)

    if update_first:
        return Q_update, Q2
    else:
        return Q1, Q_update



def double_update(
    Q1, Q2, N1, N2,
    state, action, reward, next_state,
    terminated, update_first, gamma,
):
    def update(Q, other, N):
        count = N[state, action] + 1
        alpha = 1.0 / count.astype(Q.dtype) ** 0.8

        greedy_action = jnp.argmax(Q[next_state])

        target = reward + gamma * jnp.where(
            terminated,
            0.0,
            other[next_state, greedy_action],
        )

        Q = Q.at[state, action].add(
            alpha * (target - Q[state, action])
        )

        N = N.at[state, action].set(count)

        return Q, N

    def update_q1(_):
        Q1_new, N1_new = update(Q1, Q2, N1)
        return Q1_new, Q2, N1_new, N2

    def update_q2(_):
        Q2_new, N2_new = update(Q2, Q1, N2)
        return Q1, Q2_new, N1, N2_new

    return jax.lax.cond(
        update_first,
        update_q1,
        update_q2,
        operand=None,
    )


def double_q_learning(
    env,
    policy,
    key,
    num_episodes=10_000,
    alpha=0.1,
    gamma=1.0,
    max_steps=1000,
    decay_exponent=None,
):
    Q1 = jnp.zeros((env.num_states, env.num_actions))
    Q2 = jnp.zeros_like(Q1)

    N1 = jnp.zeros_like(Q1, dtype=jnp.int32)
    N2 = jnp.zeros_like(Q2, dtype=jnp.int32)

    for _ in range(num_episodes):
        key, env_key, action_key = jax.random.split(key, 3)
        env_state, observation = env.reset(env_key)

        action = policy.select_action(
            action_key, Q1[observation] + Q2[observation]
        )

        for _ in range(max_steps):
            key, env_key, action_key, update_key = jax.random.split(key, 4)

            new_state, new_observation, reward, terminated, _ = (
                env.step(env_key, env_state, action)
            )

            update_first = bool(jax.random.bernoulli(update_key))

            if update_first:
                N1 = N1.at[observation, action].add(1)
                count = N1[observation, action]
            else:
                N2 = N2.at[observation, action].add(1)
                count = N2[observation, action]

            if decay_exponent is None:
                step_alpha = alpha
            else:
                step_alpha = count.astype(Q1.dtype) ** (-decay_exponent)

            Q1, Q2 = double_q_learning_update(
                Q1, Q2,
                observation, action, reward,
                new_observation, terminated,
                update_first,
                alpha=step_alpha,
                gamma=gamma,
            )

            if bool(terminated):
                break

            env_state, observation = new_state, new_observation

            action = policy.select_action(
                action_key,
                Q1[observation] + Q2[observation],
            )
        else:
            raise RuntimeError(
                f"Episode did not terminate within {max_steps} steps"
            )

    return Q1, Q2



@partial(
    jax.jit,
    static_argnames=(
        "env",
        "policy",
        "num_episodes",
        "max_steps",
        "decay_exponent",
    ),
)
def _double_q_learning_jit(
    env,
    policy,
    key,
    num_episodes,
    alpha,
    gamma,
    max_steps,
    decay_exponent,
):
    Q1 = jnp.zeros(
        (env.num_states, env.num_actions), dtype=jnp.float32
    )
    Q2 = jnp.zeros_like(Q1)

    N1 = jnp.zeros_like(Q1, dtype=jnp.int32)
    N2 = jnp.zeros_like(Q2, dtype=jnp.int32)

    def train_episode(carry, _):
        Q1, Q2, N1, N2, key, failed = carry

        def skip_episode(_):
            return Q1, Q2, N1, N2, key, failed

        def run_episode(_):
            episode_key, reset_key, action_key = (
                jax.random.split(key, 3)
            )

            env_state, observation = env.reset(reset_key)

            action = policy.select_action(
                action_key,
                Q1[observation] + Q2[observation],
            )

            initial = (
                Q1, Q2, N1, N2, episode_key,
                env_state, observation, action,
                jnp.int32(0),
                jnp.bool_(False),
                jnp.bool_(False),
            )

            def condition(carry):
                return ~carry[9]

            def step(carry):
                (
                    Q1, Q2, N1, N2, key,
                    env_state, observation, action,
                    steps, done, failed,
                ) = carry

                (
                    next_key, step_key,
                    action_key, update_key,
                ) = jax.random.split(key, 4)

                (
                    next_env_state, next_observation,
                    reward, terminated, _,
                ) = env.step(step_key, env_state, action)

                reward = jnp.asarray(reward, dtype=Q1.dtype)

                update_first = jax.random.bernoulli(
                    update_key, p=0.5
                )

                def update_one(Q, other, N):
                    count = N[observation, action] + 1

                    if decay_exponent is None:
                        step_alpha = alpha
                    else:
                        step_alpha = (
                            count.astype(Q.dtype) ** (-decay_exponent)
                        )

                    next_action = jnp.argmax(
                        Q[next_observation]
                    )

                    target = reward + gamma * jnp.where(
                        terminated,
                        0.0,
                        other[next_observation, next_action],
                    )

                    Q = Q.at[observation, action].add(
                        step_alpha * (
                            target - Q[observation, action]
                        )
                    )

                    N = N.at[observation, action].set(count)

                    return Q, N

                def update_q1(_):
                    updated_Q1, updated_N1 = update_one(
                        Q1, Q2, N1
                    )
                    return updated_Q1, Q2, updated_N1, N2

                def update_q2(_):
                    updated_Q2, updated_N2 = update_one(
                        Q2, Q1, N2
                    )
                    return Q1, updated_Q2, N1, updated_N2

                Q1, Q2, N1, N2 = jax.lax.cond(
                    update_first,
                    update_q1,
                    update_q2,
                    operand=None,
                )

                next_action = jax.lax.cond(
                    terminated,
                    lambda _: action,
                    lambda _: policy.select_action(
                        action_key,
                        Q1[next_observation] + Q2[next_observation],
                    ),
                    operand=None,
                )

                failed = (
                    ~terminated
                ) & (steps + 1 >= max_steps)

                done = terminated | failed

                return (
                    Q1, Q2, N1, N2, next_key,
                    next_env_state, next_observation,
                    next_action, steps + 1, done, failed,
                )

            final = jax.lax.while_loop(
                condition, step, initial
            )

            return (
                final[0], final[1], final[2],
                final[3], final[4], final[10],
            )

        next_carry = jax.lax.cond(
            failed,
            skip_episode,
            run_episode,
            operand=None,
        )

        return next_carry, None

    (Q1, Q2, _, _, _, failed), _ = jax.lax.scan(
        train_episode,
        (
            Q1, Q2, N1, N2,
            key, jnp.bool_(False),
        ),
        None,
        length=num_episodes,
    )

    return Q1, Q2, failed



def double_q_learning_jit(
    env,
    policy,
    key,
    num_episodes=10_000,
    alpha=0.1,
    gamma=1.0,
    max_steps=1000,
    decay_exponent=None,
):
    """
    JIT-compiled tabular Double Q-learning.

    Returns:
        Q1, Q2: Two learned action-value tables.
    """
    if num_episodes < 0:
        raise ValueError("num_episodes must be nonnegative")

    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    Q1, Q2, failed = _double_q_learning_jit(
        env,
        policy,
        key,
        num_episodes,
        alpha,
        gamma,
        max_steps,
        decay_exponent,
    )

    if bool(failed):
        raise RuntimeError(
            f"Episode did not terminate within {max_steps} steps"
        )

    return Q1, Q2
