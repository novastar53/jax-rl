"""Track bootstrap estimator bias, variance and MSE in the maximization-bias MDP.

Measurements are made before each episode, whether or not state B is visited.
The Double Q direction and Gaussian Monte Carlo key are the ones that would
be used for the first update of that episode. Training keys are unchanged.
"""

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
    estimate = jnp.max(jnp.where(env.action_mask(1), Q[1], -jnp.inf))

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

    return (Q, N, key), (action == 0, estimate)


def double_q_learning_episode(
    carry, env, epsilon, gamma, decay_exponent, alpha_cap,
):
    Q1, Q2, N1, N2, key = carry

    (
        key, reset_key, action_key, step_key, update_key,
        b_action_key, b_step_key, b_update_key,
    ) = jax.random.split(key, 8)

    update_first = jax.random.bernoulli(update_key)
    valid = env.action_mask(1)
    a1 = jnp.argmax(jnp.where(valid, Q1[1], -jnp.inf))
    a2 = jnp.argmax(jnp.where(valid, Q2[1], -jnp.inf))
    estimate = jnp.where(update_first, Q2[1, a1], Q1[1, a2])

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
        update_first,
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
        (action == 0, estimate),
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

    # Probe a hypothetical A -> B transition with the actual scheduled step
    # size and update key. The library returns a new immutable learner state.
    # Recover the bootstrap from the ordinary TD update, without reimplementing
    # Gaussian weighting or its uncertainty estimates.
    left = jnp.int32(0)
    b = jnp.int32(1)
    alpha_left = learning_rate(
        learner.N[state, left] + 1, learner.Q.dtype,
        decay_exponent, alpha_cap,
    )
    probe = update_gaussian_weighted_q(
        learner, update_key, state, left,
        jnp.zeros((), dtype=learner.Q.dtype), b, jnp.bool_(False),
        gamma=gamma, fixed_alpha=alpha_left,
        next_action_mask=env.action_mask(b),
    )
    q_old = learner.Q[state, left]
    q_new = probe.Q[state, left]
    estimate = (q_old + (q_new - q_old) / alpha_left) / gamma

    def update_right(original):
        return update_gaussian_weighted_q(
            original, update_key, state, action, reward, next_state, terminated,
            gamma=gamma,
            fixed_alpha=learning_rate(
                original.N[state, action] + 1, original.Q.dtype,
                decay_exponent, alpha_cap,
            ),
            next_action_mask=env.action_mask(next_state),
        )

    # Reuse the probe only when Left was actually selected; otherwise discard it.
    learner = jax.lax.cond(action == left, lambda _: probe, update_right, learner)

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

    return (learner, key), (action == 0, estimate)


@partial(jax.jit, static_argnames=("num_runs", "num_episodes"))
def run_experiment(
    key,
    num_runs=200,
    num_episodes=5000,
    epsilon=0.1,
    gamma=1.0,  # Must be positive to recover the bootstrap from the TD update.
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


    q_left, q_estimate = jax.vmap(run_q)(q_keys)
    double_left, double_estimate = jax.vmap(run_double)(double_keys)
    gaussian_left, gaussian_estimate = jax.vmap(run_gaussian)(gaussian_keys)

    # Shape: (algorithms, runs, episodes).
    estimates = jnp.stack([q_estimate, double_estimate, gaussian_estimate])
    left = jnp.stack([q_left, double_left, gaussian_left])
    errors = estimates + 0.1
    bias = errors.mean(axis=1)
    variance = errors.var(axis=1, ddof=0)
    mse = (errors ** 2).mean(axis=1)

    # Number of visits to B before each measurement (not including this episode).
    visits = jnp.concatenate(
        [jnp.zeros_like(left[..., :1], dtype=jnp.int32),
         jnp.cumsum(left[..., :-1], axis=-1, dtype=jnp.int32)],
        axis=-1,
    ).mean(axis=1)

    # Retain regret as a regression check against the prior experiment.
    early = min(300, num_episodes)
    regret = 0.1 * jnp.stack([
        left[..., :early].sum(axis=-1), left.sum(axis=-1)
    ])
    return bias, variance, mse, visits, regret


@partial(jax.jit, static_argnames=("num_runs", "num_episodes"))
def run_sweep(
    key,
    reward_stds,
    num_runs=200,
    num_episodes=5000,
    epsilon=0.1,
    gamma=1.0,  # Must be positive to recover the bootstrap from the TD update.
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
    reward_stds = jnp.array([0.5, 1.0, 4.0])
    q_alpha_cap = 1.0
    double_alpha_cap = 0.5
    gaussian_alpha_cap = 0.1

    names = ["Q-learning", "Double Q-learning", "Gaussian-weighted Q-learning"]
    bias, variance, mse, visits, regret = run_sweep(
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

    # Shape of each diagnostic: (reward stds, algorithms, episodes).
    assert bool(jnp.all(jnp.isfinite(bias)))
    assert bool(jnp.all(jnp.isfinite(variance)))
    assert bool(jnp.all(jnp.isfinite(mse)))
    assert bool(jnp.allclose(mse, bias ** 2 + variance, rtol=1e-4, atol=1e-6))

    episodes = jnp.arange(1, num_episodes + 1)
    checkpoints = [n for n in [1, 10, 50, 100, 300, 1000, 5000]
                   if n <= num_episodes]
    if num_episodes not in checkpoints:
        checkpoints.append(num_episodes)

    for i, std in enumerate(reward_stds):
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        for ax, metric, label in zip(
            axes, [bias, variance, mse],
            ["Bias", "Variance", "Mean squared error"],
        ):
            for a, name in enumerate(names):
                ax.plot(episodes, metric[i, a], label=name)
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
        axes[0].axhline(0, linestyle="--", color="gray", linewidth=1)
        axes[0].legend()
        axes[-1].set_xscale("log")
        axes[-1].set_xlabel("Episode (before update)")
        fig.suptitle(f"Bootstrap estimator diagnostics (σ={std:g})")
        fig.tight_layout()

        print(f"\nReward standard deviation: {std:g}")
        print(f"{'Episode':>8} {'Algorithm':<28} {'Bias':>10} "
              f"{'Variance':>10} {'MSE':>10} {'B visits':>10}")
        for n in checkpoints:
            for a, name in enumerate(names):
                print(f"{n:8d} {name:<28} {bias[i, a, n-1]:10.5f} "
                      f"{variance[i, a, n-1]:10.5f} "
                      f"{mse[i, a, n-1]:10.5f} {visits[i, a, n-1]:10.2f}")

    horizons = jnp.array([min(300, num_episodes), num_episodes])
    baseline = 0.1 * epsilon / 2 * horizons
    excess = regret.mean(axis=-1) - baseline[None, :, None]
    print("\nExcess regret (check against variance-sensitivity experiment):")
    for i, std in enumerate(reward_stds):
        for h, horizon in enumerate(horizons):
            print(f"σ={std:g}, episodes={horizon}: " + ", ".join(
                f"{name}={excess[i, h, a]:.4f}" for a, name in enumerate(names)
            ))

    plt.show()


if __name__ == "__main__":
    main()
