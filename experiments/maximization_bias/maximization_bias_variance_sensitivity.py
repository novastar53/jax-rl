"""Sensitivity of maximization-bias algorithms to reward standard deviation."""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax_rl.envs.maximization_bias import MaximizationBias
from jax_rl.td.gaussian_weighted_q import (
    init_gaussian_weighted_q,
    update_gaussian_weighted_q,
)

@dataclass(frozen=True)
class VariableRewardMaximizationBias(MaximizationBias):
    reward_std: float = 1.0

    def step(self, key, env_state, action):
        next_env_state, observation, reward, terminated, info = super().step(
            key, env_state, action
        )
        reward = jnp.where(
            env_state == 1,
            -0.1 + self.reward_std * jax.random.normal(key),
            reward,
        )
        return next_env_state, observation, reward, terminated, info


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
    reward_std=1.0,
    q_alpha_cap=1.0,
    double_alpha_cap=0.5,
    gaussian_alpha_cap=0.1,
):
    env = VariableRewardMaximizationBias(reward_std=reward_std)

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
                alpha_cap=q_alpha_cap,
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
                alpha_cap=double_alpha_cap,
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
                alpha_cap=gaussian_alpha_cap,
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


    left = jnp.stack([q_left, double_left, gaussian_left])
    early = min(300, num_episodes)
    regret_by_run = 0.1 * jnp.stack([
        left[..., :early].sum(axis=-1),
        left.sum(axis=-1),
    ])
    tail = min(50, num_episodes)
    last_left = 100 * left[..., -tail:].mean(axis=(-1, -2))

    return regret_by_run, last_left


@partial(jax.jit, static_argnames=("num_runs", "num_episodes"))
def run_sweep(
    key,
    reward_stds,
    num_runs=200,
    num_episodes=5000,
    epsilon=0.1,
    gamma=1.0,
    decay_exponent=0.8,
    q_alpha_cap=1.0,
    double_alpha_cap=0.5,
    gaussian_alpha_cap=0.1,
):
    def run_std(reward_std):
        return run_experiment(
            key,
            num_runs=num_runs,
            num_episodes=num_episodes,
            epsilon=epsilon,
            gamma=gamma,
            decay_exponent=decay_exponent,
            reward_std=reward_std,
            q_alpha_cap=q_alpha_cap,
            double_alpha_cap=double_alpha_cap,
            gaussian_alpha_cap=gaussian_alpha_cap,
        )

    return jax.vmap(run_std)(reward_stds)


def main():
    num_runs = 200
    num_episodes = 5000
    epsilon = 0.1
    decay_exponent = 0.8
    reward_stds = jnp.array([0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
    q_alpha_cap = 1.0
    double_alpha_cap = 0.5
    gaussian_alpha_cap = 0.1

    names = ["Q-learning", "Double Q-learning", "Gaussian-weighted Q-learning"]
    horizons = [min(300, num_episodes), num_episodes]

    # Shape: (reward stds, horizons, algorithms, runs).
    # Reuse the same run keys across reward standard deviations.
    regret, last_left = run_sweep(
        jax.random.key(43),
        reward_stds,
        num_runs=num_runs,
        num_episodes=num_episodes,
        epsilon=epsilon,
        decay_exponent=decay_exponent,
        q_alpha_cap=q_alpha_cap,
        double_alpha_cap=double_alpha_cap,
        gaussian_alpha_cap=gaussian_alpha_cap,
    )

    baseline = 0.1 * epsilon / 2 * jnp.array(horizons)
    excess = regret - baseline[None, :, None, None]
    mean = excess.mean(axis=-1)
    se = excess.std(axis=-1, ddof=1) / jnp.sqrt(num_runs)

    # Algorithm streams are independent; noise comparisons share run keys.
    improvement = regret[:, :, 1].mean(axis=-1) - regret[:, :, 2].mean(axis=-1)
    improvement_se = jnp.sqrt(se[:, :, 1] ** 2 + se[:, :, 2] ** 2)

    print(f"Decay exponent: {decay_exponent:g}")
    print(f"Runs per reward standard deviation: {num_runs}")
    print(f"Caps (Q, Double Q, Gaussian): "
          f"{q_alpha_cap:g}, {double_alpha_cap:g}, {gaussian_alpha_cap:g}")
    for h, horizon in enumerate(horizons):
        print(f"\nExcess regret after {horizon} episodes:")
        print(f"{'Std':>6} {'Q-learning':>12} {'Double Q':>12} "
              f"{'Gaussian':>12} {'D - G':>12} {'95% CI (D - G)':>24}")
        for i, std in enumerate(reward_stds):
            delta = improvement[i, h]
            margin = 1.96 * improvement_se[i, h]
            print(f"{std:6.2f} {mean[i, h, 0]:12.4f} "
                  f"{mean[i, h, 1]:12.4f} {mean[i, h, 2]:12.4f} "
                  f"{delta:12.4f} [{delta - margin:8.4f}, {delta + margin:8.4f}]")

        plt.figure(figsize=(9, 5))
        for a, name in enumerate(names):
            plt.errorbar(
                reward_stds, mean[:, h, a], yerr=1.96 * se[:, h, a],
                marker="o", capsize=4, label=name,
            )
        plt.axhline(0, linestyle="--", label="Optimal ε-greedy baseline")
        plt.xscale("log")
        plt.xticks(reward_stds, [f"{std:g}" for std in reward_stds])
        plt.xlabel("Reward standard deviation (σ)")
        plt.ylabel("Cumulative excess regret")
        plt.title(f"Reward-noise sensitivity: {horizon} episodes (p={decay_exponent:g})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

    plt.figure(figsize=(9, 5))
    for h, horizon in enumerate(horizons):
        plt.errorbar(
            reward_stds, improvement[:, h], yerr=1.96 * improvement_se[:, h],
            marker="o", capsize=4, label=f"{horizon} episodes",
        )
    plt.axhline(0, linestyle="--")
    plt.xscale("log")
    plt.xticks(reward_stds, [f"{std:g}" for std in reward_stds])
    plt.xlabel("Reward standard deviation (σ)")
    plt.ylabel("Double Q regret − Gaussian-weighted regret")
    plt.title("Relative performance (positive favors Gaussian weighting)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    print(f"\nLeft selection rate over last {min(50, num_episodes)} episodes (%):")
    print(f"{'Std':>6} {'Q-learning':>12} {'Double Q':>12} {'Gaussian':>12}")
    for i, std in enumerate(reward_stds):
        print(f"{std:6.2f} {last_left[i, 0]:12.2f} "
              f"{last_left[i, 1]:12.2f} {last_left[i, 2]:12.2f}")

    plt.show()


if __name__ == "__main__":
    main()
