"""Maximization-bias comparison with capped per-entry decaying learning rates."""

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax_rl.envs.maximization_bias import MaximizationBias
from jax_rl.td.gaussian_weighted_q import (
    init_gaussian_weighted_q,
    update_gaussian_weighted_q,
)

def select_action(key, q_values, state, epsilon, env):
    valid = env.action_mask(state)
    masked_q = jnp.where(valid, q_values, -jnp.inf)
    greedy = valid & (q_values == jnp.max(masked_q))

    probs = (
        epsilon * valid / jnp.sum(valid)
        + (1.0 - epsilon) * greedy / jnp.sum(greedy)
    )

    return jax.random.choice(key, env.num_actions, p=probs)


def learning_rate(count, dtype, decay_exponent, alpha_cap):
    return jnp.minimum(
        jnp.asarray(alpha_cap, dtype=dtype),
        count.astype(dtype) ** (-decay_exponent),
    )


def double_update(
    Q1, Q2, N1, N2,
    state, action, reward, next_state,
    terminated, update_first, gamma, decay_exponent, alpha_cap,
):
    def update(Q, other, N):
        count = N[state, action] + 1

        step_alpha = learning_rate(count, Q.dtype, decay_exponent, alpha_cap)

        greedy_action = jnp.argmax(Q[next_state])

        target = reward + gamma * jnp.where(
            terminated,
            0.0,
            other[next_state, greedy_action],
        )

        Q = Q.at[state, action].add(
            step_alpha * (target - Q[state, action])
        )

        N = N.at[state, action].set(count)

        return Q, N

    def update_q1(_):
        Q, N = update(Q1, Q2, N1)
        return Q, Q2, N, N2

    def update_q2(_):
        Q, N = update(Q2, Q1, N2)
        return Q1, Q, N1, N

    return jax.lax.cond(
        update_first,
        update_q1,
        update_q2,
        operand=None,
    )


def q_learning_episode(carry, env, epsilon, gamma, decay_exponent, alpha_cap):
    Q, N, key = carry

    key, reset_key, action_key, step_key, b_action_key, b_step_key = (
        jax.random.split(key, 6)
    )

    def update(Q, N, s, a, target):
        N = N.at[s, a].add(1)
        alpha = learning_rate(N[s, a], Q.dtype, decay_exponent, alpha_cap)

        Q = Q.at[s, a].add(
            alpha * (target - Q[s, a])
        )

        return Q, N

    env_state, state = env.reset(reset_key)

    action = select_action(
        action_key, Q[state], state, epsilon, env
    )

    next_env_state, next_state, reward, terminated, _ = env.step(
        step_key, env_state, action
    )

    target = reward + gamma * jnp.where(
        terminated, 0.0, jnp.max(Q[next_state])
    )

    Q, N = update(Q, N, state, action, target)

    def visit_b(tables):
        Q, N = tables

        action_b = select_action(
            b_action_key, Q[next_state], next_state, epsilon, env
        )

        _, _, reward_b, _, _ = env.step(
            b_step_key, next_env_state, action_b
        )

        Q, N = update(Q, N, next_state, action_b, reward_b)

        return (Q, N), reward + reward_b

    def end_at_a(tables):
        return tables, reward

    (Q, N), episode_return = jax.lax.cond(
        action == 0,
        visit_b,
        end_at_a,
        (Q, N),
    )

    return (Q, N, key), (action == 0, episode_return)


def double_q_learning_episode(
    carry, env, epsilon, gamma, decay_exponent, alpha_cap,
):
    Q1, Q2, N1, N2, key = carry

    (
        key, reset_key, action_key, step_key, update_key,
        b_action_key, b_step_key, b_update_key,
    ) = jax.random.split(key, 8)

    env_state, state = env.reset(reset_key)

    action = select_action(
        action_key,
        Q1[state] + Q2[state],
        state,
        epsilon,
        env,
    )

    next_env_state, next_state, reward, terminated, _ = env.step(
        step_key, env_state, action
    )

    Q1, Q2, N1, N2 = double_update(
        Q1, Q2, N1, N2,
        state, action, reward, next_state,
        terminated,
        jax.random.bernoulli(update_key),
        gamma, decay_exponent, alpha_cap,
    )

    def visit_b(tables):
        Q1, Q2, N1, N2 = tables

        action_b = select_action(
            b_action_key,
            Q1[next_state] + Q2[next_state],
            next_state,
            epsilon,
            env,
        )

        _, terminal_state, reward_b, terminated_b, _ = env.step(
            b_step_key, next_env_state, action_b
        )

        tables = double_update(
            Q1, Q2, N1, N2,
            next_state, action_b, reward_b,
            terminal_state, terminated_b,
            jax.random.bernoulli(b_update_key),
            gamma, decay_exponent, alpha_cap,
        )

        return tables, reward + reward_b

    def end_at_a(tables):
        return tables, reward

    (Q1, Q2, N1, N2), episode_return = jax.lax.cond(
        action == 0,
        visit_b,
        end_at_a,
        (Q1, Q2, N1, N2),
    )

    return (
        (Q1, Q2, N1, N2, key),
        (action == 0, episode_return),
    )


def gaussian_weighted_q_episode(carry, env, epsilon, gamma, decay_exponent, alpha_cap):
    learner, key = carry

    (
        key, reset_key, action_key, step_key,
        update_key, b_action_key, b_step_key, b_update_key,
    ) = jax.random.split(key, 8)

    env_state, state = env.reset(reset_key)

    action = select_action(
        action_key, learner.Q[state], state, epsilon, env
    )

    next_env_state, next_state, reward, terminated, _ = env.step(
        step_key, env_state, action
    )

    learner = update_gaussian_weighted_q(
        learner,
        update_key,
        state,
        action,
        reward,
        next_state,
        terminated,
        gamma=gamma,
        fixed_alpha=learning_rate(
            learner.N[state, action] + 1, learner.Q.dtype,
            decay_exponent, alpha_cap,
        ),
        next_action_mask=env.action_mask(next_state),
    )

    def visit_b(learner):
        action_b = select_action(
            b_action_key,
            learner.Q[next_state],
            next_state,
            epsilon,
            env,
        )

        _, terminal_state, reward_b, terminated_b, _ = env.step(
            b_step_key, next_env_state, action_b
        )

        learner = update_gaussian_weighted_q(
            learner,
            b_update_key,
            next_state,
            action_b,
            reward_b,
            terminal_state,
            terminated_b,
            gamma=gamma,
            fixed_alpha=learning_rate(
                learner.N[next_state, action_b] + 1, learner.Q.dtype,
                decay_exponent, alpha_cap,
            ),
            next_action_mask=env.action_mask(terminal_state),
        )

        return learner, reward + reward_b

    def end_at_a(learner):
        return learner, reward

    learner, episode_return = jax.lax.cond(
        action == 0,
        visit_b,
        end_at_a,
        learner,
    )

    return (learner, key), (action == 0, episode_return)


@partial(jax.jit, static_argnames=("num_runs", "num_episodes"))
def run_experiment(
    key,
    num_runs=200,
    num_episodes=5000,
    epsilon=0.1,
    gamma=1.0,
    decay_exponent=0.8,
    alpha_cap=0.1,
):
    env = MaximizationBias()

    q_keys, double_keys, gaussian_keys = jax.random.split(key, 3)

    q_keys = jax.random.split(q_keys, num_runs)
    double_keys = jax.random.split(double_keys, num_runs)
    gaussian_keys = jax.random.split(gaussian_keys, num_runs)


    def run_q(key):
        Q = jnp.zeros(
            (env.num_states, env.num_actions),
            dtype=jnp.float32,
        )
        N = jnp.zeros_like(Q, dtype=jnp.int32)

        _, results = jax.lax.scan(
            lambda carry, _: q_learning_episode(
                carry, env, epsilon, gamma=gamma,
                decay_exponent=decay_exponent,
                alpha_cap=alpha_cap,
            ),
            (Q, N, key),
            None,
            length=num_episodes,
        )

        return results


    def run_double(key):
        Q1 = jnp.zeros(
            (env.num_states, env.num_actions),
            dtype=jnp.float32,
        )
        Q2 = jnp.zeros_like(Q1)

        N1 = jnp.zeros_like(Q1, dtype=jnp.int32)
        N2 = jnp.zeros_like(Q2, dtype=jnp.int32)

        _, results = jax.lax.scan(
            lambda carry, _: double_q_learning_episode(
                carry,
                env,
                epsilon,
                gamma=gamma,
                decay_exponent=decay_exponent,
                alpha_cap=alpha_cap,
            ),
            (Q1, Q2, N1, N2, key),
            None,
            length=num_episodes,
        )

        return results

    def run_gaussian(key):
        learner = init_gaussian_weighted_q(
            env.num_states, env.num_actions
        )

        _, results = jax.lax.scan(
            lambda carry, _: gaussian_weighted_q_episode(
                carry, env, epsilon, gamma=gamma,
                decay_exponent=decay_exponent,
                alpha_cap=alpha_cap,
            ),
            (learner, key),
            None,
            length=num_episodes,
        )

        return results


    q_left, q_reward = jax.vmap(run_q)(q_keys)
    double_left, double_reward = jax.vmap(run_double)(double_keys)
    gaussian_left, gaussian_reward = jax.vmap(run_gaussian)(
        gaussian_keys
    )


    regret_by_run = 0.1 * jnp.stack([
        q_left.sum(axis=1),
        double_left.sum(axis=1),
        gaussian_left.sum(axis=1),
    ])


    return (
        q_left.mean(axis=0) * 100,
        double_left.mean(axis=0) * 100,
        gaussian_left.mean(axis=0) * 100,
        q_reward.mean(axis=0),
        double_reward.mean(axis=0),
        gaussian_reward.mean(axis=0),
        regret_by_run,
    )


def main():
    num_runs = 200
    num_episodes = 5000
    epsilon = 0.1
    decay_exponent = 0.8
    alpha_cap = 0.1

    q_left, double_left, gaussian_left, q_reward, double_reward, gaussian_reward, regret_by_run = (
        run_experiment(
            jax.random.key(42),
            num_runs=num_runs,
            num_episodes=num_episodes,
            epsilon=epsilon,
            decay_exponent=decay_exponent,
            alpha_cap=alpha_cap,
        )
    )

    episodes = jnp.arange(1, num_episodes + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(episodes, q_left, label="Q-learning")
    plt.plot(episodes, double_left, label="Double Q-learning")
    plt.plot(episodes, gaussian_left, label="Gaussian-weighted Q-learning")
    plt.axhline(
        epsilon / 2 * 100,
        linestyle="--",
        label="Optimal ε-greedy baseline",
    )
    plt.xlabel("Episodes")
    plt.ylabel("Left actions (%)")
    plt.title(f"Maximization Bias: Capped decaying α (cap={alpha_cap:g}, p={decay_exponent:g})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    window = min(20, num_episodes)
    kernel = jnp.ones(window) / window

    q_reward_smooth = jnp.convolve(
        q_reward, kernel, mode="valid"
    )
    double_reward_smooth = jnp.convolve(
        double_reward, kernel, mode="valid"
    )

    gaussian_reward_smooth = jnp.convolve(
        gaussian_reward, kernel, mode="valid"
    )

    plt.figure(figsize=(9, 5))
    plt.plot(
        episodes[window - 1:],
        q_reward_smooth,
        label="Q-learning",
    )
    plt.plot(
        episodes[window - 1:],
        double_reward_smooth,
        label="Double Q-learning",
    )
    plt.plot(
        episodes[window - 1:],
        gaussian_reward_smooth,
        label="Gaussian-weighted Q-learning",
    )
    plt.axhline(0, linestyle="--", label="Optimal greedy return")
    plt.axhline(
        -0.1 * epsilon / 2,
        linestyle=":",
        label="Optimal ε-greedy return",
    )
    plt.xlabel("Episodes")
    plt.ylabel("Average episode return")
    plt.title(f"Episode Returns: Capped decaying α (cap={alpha_cap:g}, p={decay_exponent:g})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


    print(f"Learning-rate cap: {alpha_cap:g}")
    print(f"Decay exponent: {decay_exponent:g}")
    tail = min(50, num_episodes)
    print(f"Left selection rate, last {tail} episodes:")
    print("Q-learning:", q_left[-tail:].mean())
    print("Double Q-learning:", double_left[-tail:].mean())
    print("Gaussian-weighted Q-learning:", gaussian_left[-tail:].mean())

    print("Regret:")
    for name, left in [
        ("Q-learning", q_left),
        ("Double Q-learning", double_left),
        ("Gaussian-weighted Q-learning", gaussian_left),
    ]:
        print(f"\n{name}")

        horizons = [50, 100, 200, 300, 1000, 5000, num_episodes]
        for n in sorted({n for n in horizons if n <= num_episodes}):
            regret = 0.1 * jnp.sum(left[:n] / 100)
            print(f"  First {n} episodes: {regret:.4f}")


    double_regret = regret_by_run[1]
    gaussian_regret = regret_by_run[2]

    improvement = double_regret.mean() - gaussian_regret.mean()

    n = double_regret.shape[0]
    se = jnp.sqrt(
        double_regret.var(ddof=1) / n
        + gaussian_regret.var(ddof=1) / n
    )

    print("Regret reduction:", improvement)
    print("95% CI:", improvement - 1.96 * se,
        improvement + 1.96 * se)


if __name__ == "__main__":
    main()