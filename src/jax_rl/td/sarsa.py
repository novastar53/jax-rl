from functools import partial

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
            key, env_key, action_key = jax.random.split(key, 3)
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
def _train_sarsa(
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



def sarsa_jit(
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

    Q, episode_returns, failed = _train_sarsa(
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


def n_step_sarsa_return(
    rewards,
    states,
    actions,
    Q,
    tau,
    T,
    n,
    gamma=1.0,
):
    """
    Calculate the n-step SARSA return using circular buffers.

    Args:
        rewards: (n,) circular buffer of recent rewards.
        states: (n+1,) circular buffer of recent states.
        actions: (n+1,) circular buffer of recent actions.
        Q: (S, A) current action-value estimates.
        tau: Time index of the state-action pair being updated.
        T: Episode termination time.
        n: Number of steps to look ahead.
        gamma: Discount factor.

    Returns:
        Scalar n-step SARSA return.
    """
    value = 0.0

    for i in range(tau, min(tau + n, T)):
        value += rewards[i % n] * gamma ** (i - tau)

    if tau + n < T: 
        idx = (tau + n) % (n + 1)
        q_state = states[idx]
        q_action = actions[idx]
        q = Q[q_state, q_action]
        value += q * (gamma ** n)

    return value


def n_step_sarsa(
    env,
    policy,
    key,
    num_episodes=10_000,
    n=1,
    alpha=0.1,
    gamma=1.0,
    max_steps=1000,
):
    """
    Online n-step SARSA control.

    Returns:
        Q: (S, A) estimated action values.
    """
    Q = jnp.zeros((env.num_states, env.num_actions), dtype=jnp.float32)
    for _ in range(num_episodes):
        reward_buffer = [0.0] * n
        state_buffer = [None] * (n + 1)
        action_buffer = [None] * (n + 1)

        terminated = False
        t = 0
        T = float('inf')

        key, env_key, action_key = jax.random.split(key, 3)
        env_state, observation = env.reset(env_key)
        state_buffer[t % (n + 1)] = observation
        action = policy.select_action(action_key, Q[observation])
        action_buffer[t % (n + 1)] = action
        while True:
            if not terminated:
                key, env_key, action_key = jax.random.split(key, 3)
                new_state, new_observation, reward, terminated, _ = env.step(env_key, env_state, action)
                reward_buffer[t % n] = reward
                state_buffer[(t + 1) % (n + 1)] = new_observation
                if bool(terminated):
                    T = t + 1
                else:
                    new_action = policy.select_action(action_key, Q[new_observation])
                    action_buffer[(t + 1) % (n + 1)] = new_action
                    action = new_action
                env_state = new_state
                observation = new_observation

                if not bool(terminated) and t + 1 >= max_steps:
                    raise RuntimeError(
                        f"Episode did not terminate within {max_steps} steps"
                    )

            tau = t - n + 1
            if tau >= 0:
                action_value = n_step_sarsa_return(reward_buffer,
                                                   state_buffer,
                                                   action_buffer,
                                                   Q, tau, T, n, gamma)
                s_tau = state_buffer[tau % (n + 1)]
                a_tau = action_buffer[tau % (n + 1)]
                delta = alpha * (action_value - Q[s_tau, a_tau])
                Q = Q.at[s_tau, a_tau].add(delta)

            if tau == T - 1:
                break

            t += 1
    return Q



@partial(
    jax.jit,
    static_argnames=(
        "env",
        "policy",
        "num_episodes",
        "n",
        "max_steps",
        "truncate"
    ),
)
def _n_step_sarsa_jit(
    env,
    policy,
    key,
    num_episodes,
    n,
    alpha,
    gamma,
    max_steps,
    truncate,
):
    Q = jnp.zeros(
        (env.num_states, env.num_actions),
        dtype=jnp.float32,
    )

    def train_episode(carry, _):
        Q, key, failed = carry

        def skip_episode(_):
            return Q, key, failed

        def run_episode(_):
            episode_key, env_key, action_key = jax.random.split(key, 3)

            env_state, observation = env.reset(env_key)
            action = policy.select_action(action_key, Q[observation])

            states = jnp.zeros(
                n + 1, dtype=jnp.int32
            ).at[0].set(observation)

            actions = jnp.zeros(
                n + 1, dtype=jnp.int32
            ).at[0].set(action)

            rewards = jnp.zeros(n, dtype=jnp.float32)

            initial = (
                Q,
                episode_key,
                env_state,
                observation,
                action,
                states,
                actions,
                rewards,
                jnp.int32(0),              # t
                jnp.int32(max_steps + 1),  # T
                jnp.bool_(False),          # terminated
                jnp.bool_(False),          # truncated
                jnp.bool_(False),          # failed
                jnp.bool_(False),          # done
            )

            def condition(carry):
                return ~carry[-1]

            def step(carry):
                (
                    Q, key, env_state, observation, action,
                    states, actions, rewards, t, T,
                    terminated, truncated, failed, done,
                ) = carry


                def take_step(_):
                    next_key, env_key, action_key = jax.random.split(
                        key, 3
                    )

                    (
                        next_env_state,
                        next_observation,
                        reward,
                        is_terminal,
                        _,
                    ) = env.step(env_key, env_state, action)

                    next_states = states.at[
                        (t + 1) % (n + 1)
                    ].set(next_observation)

                    next_rewards = rewards.at[t % n].set(reward)

                    reached_limit = (t + 1 >= max_steps) & (~is_terminal)

                    is_truncated = reached_limit & truncate
                    step_failed = reached_limit & (not truncate)

                    episode_ended = is_terminal | is_truncated

                    next_T = jnp.where(
                        episode_ended, t + 1, T
                    )

                    # A new action is needed at a time-limit cutoff
                    # because the pending returns must bootstrap.
                    next_action = jax.lax.cond(
                        is_terminal,
                        lambda _: action,
                        lambda _: policy.select_action(
                            action_key, Q[next_observation]
                        ),
                        operand=None,
                    )

                    next_actions = actions.at[
                        (t + 1) % (n + 1)
                    ].set(next_action)

                    return (
                        next_key,
                        next_env_state,
                        next_observation,
                        next_action,
                        next_states,
                        next_actions,
                        next_rewards,
                        next_T,
                        episode_ended,
                        is_truncated,
                        step_failed,
                    )
 
                def flush_pending(_):
                    return (
                        key,
                        env_state,
                        observation,
                        action,
                        states,
                        actions,
                        rewards,
                        T,
                        terminated,
                        truncated,
                        jnp.bool_(False),
                    )

                (
                    key,
                    env_state,
                    observation,
                    action,
                    states,
                    actions,
                    rewards,
                    T,
                    terminated,
                    truncated,
                    step_failed,
                ) = jax.lax.cond(
                    terminated,
                    flush_pending,
                    take_step,
                    operand=None,
                )

                tau = t - n + 1

                def update_q(Q):
                    end = jnp.minimum(tau + n, T)

                    def accumulate_reward(i, total):
                        return (
                            total
                            + gamma ** (i - tau)
                            * rewards[i % n]
                        )

                    G = jax.lax.fori_loop(
                        tau,
                        end,
                        accumulate_reward,
                        jnp.float32(0.0),
                    )

                    bootstrap_time = jnp.minimum(tau + n, T)

                    should_bootstrap = (tau + n < T) | truncated

                    def add_bootstrap(G):
                        index = bootstrap_time % (n + 1)

                        bootstrap_state = states[index]
                        bootstrap_action = actions[index]

                        return (
                            G
                            + gamma ** (bootstrap_time - tau)
                            * Q[bootstrap_state, bootstrap_action]
                        )

                    G = jax.lax.cond(
                        should_bootstrap,
                        add_bootstrap,
                        lambda G: G,
                        G,
                    )

                    index = tau % (n + 1)

                    s_tau = states[index]
                    a_tau = actions[index]

                    return Q.at[s_tau, a_tau].add(
                        alpha * (G - Q[s_tau, a_tau])
                    )


                Q = jax.lax.cond(
                    (tau >= 0) & (~step_failed),
                    update_q,
                    lambda Q: Q,
                    Q,
                )

                done = step_failed | (tau == T - 1)

                return (
                    Q,
                    key,
                    env_state,
                    observation,
                    action,
                    states,
                    actions,
                    rewards,
                    t + 1,
                    T,
                    terminated,
                    truncated,
                    step_failed,
                    done,
                )

            final = jax.lax.while_loop(
                condition,
                step,
                initial,
            )

            return final[0], final[1], final[12]

        next_carry = jax.lax.cond(
            failed,
            skip_episode,
            run_episode,
            operand=None,
        )

        return next_carry, None

    (Q, _, failed), _ = jax.lax.scan(
        train_episode,
        (Q, key, jnp.bool_(False)),
        None,
        length=num_episodes,
    )

    return Q, failed


def n_step_sarsa_jit(
    env,
    policy,
    key,
    num_episodes=10_000,
    n=1,
    alpha=0.1,
    gamma=1.0,
    max_steps=1000,
    truncate=False,
):
    """
    JIT-compiled online n-step SARSA control.

    Returns:
        Q: (S, A) estimated action values.
    """
    if n < 1:
        raise ValueError("n must be positive")

    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    Q, failed = _n_step_sarsa_jit(
        env,
        policy,
        key,
        num_episodes,
        n,
        alpha,
        gamma,
        max_steps,
        truncate,
    )

    if bool(failed):
        raise RuntimeError(
            f"Episode did not terminate within {max_steps} steps"
        )

    return Q



def n_step_importance_sampling_ratio(
    states,
    actions,
    behavior_prob_buffer,
    target_policy,
    Q,
    tau,
    T,
    n,
):
    """
    Calculate the importance-sampling ratio for
    off-policy n-step SARSA.

    Args:
        states: (n+1,) circular buffer of state indices.
        actions: (n+1,) circular buffer of action indices.
        target_probs: (S, A) target policy probabilities.
        behavior_probs: (S, A) behavior policy probabilities.
        tau: Time index of the state-action pair being updated.
        T: Episode termination time.
        n: Number of steps.
        
    Returns:
        Scalar importance-sampling ratio.
    """
    rho = 1.0

    for k in range(tau + 1, min(tau + n, T - 1) + 1):
        index = k % (n + 1)

        state = states[index]
        action = actions[index]
        q_values = Q[state]
        pi = target_policy.action_probs(q_values)[action]
        b = behavior_prob_buffer[index]

        rho *= pi / b

    return rho



def off_policy_n_step_sarsa(
    env,
    behavior_policy,
    target_policy,
    key,
    num_episodes=10_000,
    n=1,
    alpha=0.1,
    gamma=1.0,
    max_steps=1000,
):
    """
    Online n-step SARSA control.

    Returns:
        Q: (S, A) estimated action values.
    """
    if n < 1:
        raise ValueError("n must be positive")

    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    Q = jnp.zeros((env.num_states, env.num_actions), dtype=jnp.float32)
    for _ in range(num_episodes):
        reward_buffer = [0.0] * n
        state_buffer = [None] * (n + 1)
        action_buffer = [None] * (n + 1)
        behavior_prob_buffer = [0.0] * (n + 1)

        terminated = False
        t = 0
        T = float('inf')

        key, env_key, action_key = jax.random.split(key, 3)
        env_state, observation = env.reset(env_key)
        state_buffer[t % (n + 1)] = observation
        action = behavior_policy.select_action(action_key, Q[observation])
        behavior_prob_buffer[t % (n + 1)] = (
            behavior_policy.action_probs(Q[observation])[action]
        )
        action_buffer[t % (n + 1)] = action
        while True:
            if not terminated:
                key, env_key, action_key = jax.random.split(key, 3)
                new_state, new_observation, reward, terminated, _ = env.step(env_key, env_state, action)
                reward_buffer[t % n] = reward
                state_buffer[(t + 1) % (n + 1)] = new_observation
                if bool(terminated):
                    T = t + 1
                else:
                    q_values = Q[new_observation]
                    new_action = behavior_policy.select_action(
                        action_key, q_values
                    )

                    index = (t + 1) % (n + 1)

                    action_buffer[index] = new_action
                    behavior_prob_buffer[index] = (
                        behavior_policy.action_probs(q_values)[new_action]
                    )

                    action = new_action

                env_state = new_state
                observation = new_observation


                if not bool(terminated) and t + 1 >= max_steps:
                    raise RuntimeError(
                        f"Episode did not terminate within {max_steps} steps"
                    )

            tau = t - n + 1
            if tau >= 0:
                action_value = n_step_sarsa_return(reward_buffer,
                                                   state_buffer,
                                                   action_buffer,
                                                   Q, tau, T, n, gamma)
                rho = n_step_importance_sampling_ratio(state_buffer, 
                                                                    action_buffer, 
                                                                    behavior_prob_buffer,
                                                                    target_policy, 
                                                                    Q, tau, T, n)
                s_tau = state_buffer[tau % (n + 1)]
                a_tau = action_buffer[tau % (n + 1)]
                delta = alpha * rho * (action_value - Q[s_tau, a_tau])
                Q = Q.at[s_tau, a_tau].add(delta)

            if tau == T - 1:
                break

            t += 1
    return Q



@partial(
    jax.jit,
    static_argnames=(
        "env",
        "behavior_policy",
        "target_policy",
        "num_episodes",
        "n",
        "max_steps",
    ),
)
def _off_policy_n_step_sarsa_jit(
    env,
    behavior_policy,
    target_policy,
    key,
    num_episodes,
    n,
    alpha,
    gamma,
    max_steps,
):
    Q = jnp.zeros(
        (env.num_states, env.num_actions),
        dtype=jnp.float32,
    )

    def train_episode(carry, _):
        Q, key, failed = carry

        def skip_episode(_):
            return Q, key, failed

        def run_episode(_):
            episode_key, reset_key, action_key = jax.random.split(
                key, 3
            )

            env_state, observation = env.reset(reset_key)

            action = behavior_policy.select_action(
                action_key, Q[observation]
            )
            action_prob = behavior_policy.action_probs(
                Q[observation]
            )[action]

            states = jnp.zeros(
                (n + 1,), dtype=jnp.int32
            ).at[0].set(observation)

            actions = jnp.zeros(
                (n + 1,), dtype=jnp.int32
            ).at[0].set(action)

            behavior_probs = jnp.zeros(
                (n + 1,), dtype=jnp.float32
            ).at[0].set(action_prob)

            rewards = jnp.zeros(
                (n,), dtype=jnp.float32
            )

            initial = (
                Q,
                episode_key,
                env_state,
                observation,
                action,
                states,
                actions,
                behavior_probs,
                rewards,
                jnp.int32(0),              # t
                jnp.int32(max_steps + 1),  # T
                jnp.bool_(False),          # terminated
                jnp.bool_(False),          # failed
                jnp.bool_(False),          # done
            )

            def condition(carry):
                return ~carry[-1]

            def step(carry):
                (
                    Q, key, env_state, observation, action,
                    states, actions, behavior_probs, rewards,
                    t, T, terminated, failed, done,
                ) = carry

                def take_step(_):
                    next_key, step_key, action_key = (
                        jax.random.split(key, 3)
                    )

                    (
                        next_env_state,
                        next_observation,
                        reward,
                        is_terminal,
                        _,
                    ) = env.step(step_key, env_state, action)

                    next_states = states.at[
                        (t + 1) % (n + 1)
                    ].set(next_observation)

                    next_rewards = rewards.at[
                        t % n
                    ].set(reward)

                    next_T = jnp.where(
                        is_terminal, t + 1, T
                    )

                    def select_next_action(_):
                        q_values = Q[next_observation]

                        next_action = (
                            behavior_policy.select_action(
                                action_key, q_values
                            )
                        )

                        next_prob = (
                            behavior_policy.action_probs(
                                q_values
                            )[next_action]
                        )

                        return next_action, next_prob

                    next_action, next_prob = jax.lax.cond(
                        is_terminal,
                        lambda _: (
                            action,
                            jnp.float32(0.0),
                        ),
                        select_next_action,
                        operand=None,
                    )

                    next_actions = actions.at[
                        (t + 1) % (n + 1)
                    ].set(next_action)

                    next_behavior_probs = behavior_probs.at[
                        (t + 1) % (n + 1)
                    ].set(next_prob)

                    step_failed = (
                        ~is_terminal
                    ) & (t + 1 >= max_steps)

                    return (
                        next_key,
                        next_env_state,
                        next_observation,
                        next_action,
                        next_states,
                        next_actions,
                        next_behavior_probs,
                        next_rewards,
                        next_T,
                        is_terminal,
                        step_failed,
                    )

                def flush_pending(_):
                    return (
                        key,
                        env_state,
                        observation,
                        action,
                        states,
                        actions,
                        behavior_probs,
                        rewards,
                        T,
                        terminated,
                        jnp.bool_(False),
                    )

                (
                    key,
                    env_state,
                    observation,
                    action,
                    states,
                    actions,
                    behavior_probs,
                    rewards,
                    T,
                    terminated,
                    step_failed,
                ) = jax.lax.cond(
                    terminated,
                    flush_pending,
                    take_step,
                    operand=None,
                )

                tau = t - n + 1

                def update_q(Q):
                    end = jnp.minimum(tau + n, T)

                    def accumulate_reward(i, total):
                        return (
                            total
                            + gamma ** (i - tau)
                            * rewards[i % n]
                        )

                    G = jax.lax.fori_loop(
                        tau,
                        end,
                        accumulate_reward,
                        jnp.float32(0.0),
                    )

                    def add_bootstrap(G):
                        index = (tau + n) % (n + 1)

                        return (
                            G
                            + gamma ** n
                            * Q[states[index], actions[index]]
                        )

                    G = jax.lax.cond(
                        tau + n < T,
                        add_bootstrap,
                        lambda G: G,
                        G,
                    )

                    last_action = jnp.minimum(
                        tau + n, T - 1
                    )

                    def accumulate_ratio(k, rho):
                        index = k % (n + 1)

                        state = states[index]
                        action = actions[index]

                        pi = target_policy.action_probs(
                            Q[state]
                        )[action]

                        b = behavior_probs[index]

                        return rho * (pi / b)

                    rho = jax.lax.fori_loop(
                        tau + 1,
                        last_action + 1,
                        accumulate_ratio,
                        jnp.float32(1.0),
                    )

                    index = tau % (n + 1)

                    s_tau = states[index]
                    a_tau = actions[index]

                    return Q.at[s_tau, a_tau].add(
                        alpha
                        * rho
                        * (G - Q[s_tau, a_tau])
                    )

                Q = jax.lax.cond(
                    (tau >= 0) & (~step_failed),
                    update_q,
                    lambda Q: Q,
                    Q,
                )

                done = step_failed | (tau == T - 1)

                return (
                    Q,
                    key,
                    env_state,
                    observation,
                    action,
                    states,
                    actions,
                    behavior_probs,
                    rewards,
                    t + 1,
                    T,
                    terminated,
                    step_failed,
                    done,
                )

            final = jax.lax.while_loop(
                condition,
                step,
                initial,
            )

            return final[0], final[1], final[12]

        next_carry = jax.lax.cond(
            failed,
            skip_episode,
            run_episode,
            operand=None,
        )

        return next_carry, None

    (Q, _, failed), _ = jax.lax.scan(
        train_episode,
        (Q, key, jnp.bool_(False)),
        None,
        length=num_episodes,
    )

    return Q, failed


def off_policy_n_step_sarsa_jit(
    env,
    behavior_policy,
    target_policy,
    key,
    num_episodes=10_000,
    n=1,
    alpha=0.1,
    gamma=1.0,
    max_steps=1000,
):
    """
    JIT-compiled off-policy n-step SARSA.

    Uses recorded behavior probabilities for
    importance sampling.

    Returns:
        Q: (S, A) estimated action values.
    """
    if n < 1:
        raise ValueError("n must be positive")

    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    Q, failed = _off_policy_n_step_sarsa_jit(
        env,
        behavior_policy,
        target_policy,
        key,
        num_episodes,
        n,
        alpha,
        gamma,
        max_steps,
    )

    if bool(failed):
        raise RuntimeError(
            f"Episode did not terminate within {max_steps} steps"
        )

    return Q


def n_step_tree_backup_return(
    states,
    actions,
    rewards,
    Q,
    target_policy,
    tau,
    T,
    n,
    gamma=1.0,
):
    """
    Compute the n-step Tree Backup return for
    the state-action pair at time tau.
    """
    value = 0.0

    if tau + n < T: 
        idx = (tau + n) % (n + 1)
        state = states[idx]
        p_actions = target_policy.action_probs[state]
        v = jnp.sum(p_actions * Q[state])
        value += v * (gamma ** n)

    for i in range(tau, min(tau + n, T)):
        value += rewards[i % n] * gamma ** (i - tau)

    return value

