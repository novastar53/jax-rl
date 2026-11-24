"""Compare maximum-expectation estimators as the best arm separates from the others."""

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def update_mean(mean, count, action, reward):
    count = count.at[action].add(1)
    mean = mean.at[action].add(
        (reward - mean[action]) / count[action]
    )
    return mean, count


def gaussian_weighted_estimate(key, mean, count, m2, num_samples, initial_se):
    variance = jnp.maximum(m2, 0.0) / jnp.maximum(count - 1, 1)
    se = jnp.sqrt(variance / jnp.maximum(count, 1))
    se = jnp.where(count >= 2, se, initial_se)

    draws = mean + se * jax.random.normal(
        key, (num_samples, mean.size), dtype=mean.dtype
    )
    winners = jnp.argmax(draws, axis=1)
    return jnp.mean(mean[winners]), jnp.mean(winners == 0)


def run_one(key, reward_std, optimality_gap, checkpoints, num_arms, num_observations,
            reward_mean, num_gaussian_samples, initial_se):
    action_key, reward_key, split_key, diagnostic_key = jax.random.split(key, 4)
    actions = jax.random.randint(action_key, (num_observations,), 0, num_arms)
    rewards = reward_mean + optimality_gap * (actions == 0) + reward_std * jax.random.normal(
        reward_key, (num_observations,), dtype=jnp.float32
    )
    split = jax.random.bernoulli(split_key, 0.5, (num_observations,))

    zero = jnp.zeros(num_arms, dtype=jnp.float32)
    counts = jnp.zeros(num_arms, dtype=jnp.int32)
    initial = (zero, counts, zero, zero, counts, zero, counts)

    def observe(t, carry):
        mean, count, m2, mean1, count1, mean2, count2 = carry
        action, reward = actions[t], rewards[t]

        old_mean = mean[action]
        mean, count = update_mean(mean, count, action, reward)
        m2 = m2.at[action].add(
            (reward - old_mean) * (reward - mean[action])
        )

        def first(_):
            new_mean, new_count = update_mean(mean1, count1, action, reward)
            return new_mean, new_count, mean2, count2

        def second(_):
            new_mean, new_count = update_mean(mean2, count2, action, reward)
            return mean1, count1, new_mean, new_count

        mean1, count1, mean2, count2 = jax.lax.cond(
            split[t], first, second, operand=None
        )
        return mean, count, m2, mean1, count1, mean2, count2

    def measure(carry, stop):
        start, stats = carry
        stats = jax.lax.fori_loop(start, stop, observe, stats)
        mean, count, m2, mean1, count1, mean2, count2 = stats

        direction_key, gaussian_key = jax.random.split(
            jax.random.fold_in(diagnostic_key, stop)
        )
        maximum = jnp.max(mean)
        select_first = jax.random.bernoulli(direction_key)
        selected1 = jnp.argmax(mean1)
        selected2 = jnp.argmax(mean2)
        double = jnp.where(select_first, mean2[selected1], mean1[selected2])
        selected = jnp.where(select_first, selected1, selected2)
        gaussian, optimal_weight = gaussian_weighted_estimate(
            gaussian_key, mean, count, m2, num_gaussian_samples, initial_se
        )
        estimates = jnp.stack([maximum, double, gaussian])
        coverage = jnp.stack([
            jnp.all(count > 0),
            jnp.all((count1 > 0) & (count2 > 0)),
        ])
        return (stop, stats), (estimates, coverage, selected == 0, optimal_weight)

    _, results = jax.lax.scan(
        measure, (jnp.int32(0), initial), checkpoints
    )
    return results


@partial(jax.jit, static_argnames=(
    "num_runs", "num_arms", "num_observations", "num_gaussian_samples",
))
def run_experiment(key, reward_std, optimality_gap, checkpoints, num_runs=1000,
                   num_arms=10, num_observations=5000, reward_mean=-0.1,
                   num_gaussian_samples=100, initial_se=1.0):
    keys = jax.random.split(key, num_runs)
    estimates, coverage, selected_best, best_weight = jax.vmap(
        lambda k: run_one(
            k, reward_std, optimality_gap, checkpoints, num_arms, num_observations,
            reward_mean, num_gaussian_samples, initial_se,
        )
    )(keys)

    # Shapes: (runs, checkpoints, estimators).
    errors = estimates - (reward_mean + optimality_gap)
    bias = errors.mean(axis=0)
    variance = estimates.var(axis=0)
    mse = (errors ** 2).mean(axis=0)
    bias_se = errors.std(axis=0, ddof=1) / jnp.sqrt(num_runs)
    mse_se = (errors ** 2).std(axis=0, ddof=1) / jnp.sqrt(num_runs)
    mse_diff = errors[:, :, 1] ** 2 - errors[:, :, 2] ** 2
    diff_mean = mse_diff.mean(axis=0)
    diff_se = mse_diff.std(axis=0, ddof=1) / jnp.sqrt(num_runs)
    return (bias, variance, mse, bias_se, mse_se, coverage.mean(axis=0),
            selected_best.mean(axis=0), best_weight.mean(axis=0), diff_mean, diff_se)


def main():
    num_runs = 1000
    num_arms = 10
    num_observations = 5000
    reward_mean = -0.1
    num_gaussian_samples = 100
    initial_se = 1.0
    reward_std = 1.0
    optimality_gaps = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    report_at = [1, 10, 50, 100, 300, 1000, 5000]
    checkpoints = jnp.unique(jnp.concatenate([
        jnp.array(report_at, dtype=jnp.int32),
        jnp.rint(jnp.geomspace(1, num_observations, 100)).astype(jnp.int32),
    ]))
    labels = ["Maximum", "Double", "Gaussian-weighted"]
    colors = ["C0", "C1", "C2"]

    summary = []
    for gap in optimality_gaps:
        (bias, variance, mse, bias_se, mse_se, coverage,
         selected_best, best_weight, diff_mean, diff_se) = run_experiment(
            jax.random.key(44), reward_std, gap, checkpoints,
            num_runs=num_runs, num_arms=num_arms,
            num_observations=num_observations, reward_mean=reward_mean,
            num_gaussian_samples=num_gaussian_samples, initial_se=initial_se,
        )
        assert bool(jnp.allclose(mse, bias ** 2 + variance, rtol=1e-5, atol=1e-6))

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        for i, (label, color) in enumerate(zip(labels, colors)):
            for ax, metric, se in zip(
                axes, (bias, variance, mse), (bias_se, None, mse_se)
            ):
                y = metric[:, i]
                ax.plot(checkpoints, y, label=label, color=color)
                if se is not None:
                    low = y - 1.96 * se[:, i]
                    if ax is axes[2]:
                        low = jnp.maximum(low, 0)
                    ax.fill_between(checkpoints, low, y + 1.96 * se[:, i],
                                    color=color, alpha=0.12)

        axes[0].axhline(0, color="gray", linestyle="--")
        axes[0].set_ylabel("Bias")
        axes[1].set_ylabel("Variance")
        axes[2].set_ylabel("Mean squared error")
        axes[2].set_xlabel("Total reward observations")
        for ax in axes:
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
        axes[0].legend()
        fig.suptitle(f"Maximum-estimator comparison (σ={reward_std:g}, Δ={gap:g})")
        fig.tight_layout()

        print(f"\nOptimality gap: {gap:g}; true maximum: {reward_mean + gap:g}")
        print(f"{'Samples':>8} {'Estimator':<20} {'Bias':>11} {'Variance':>11} {'MSE':>11}")
        for n in report_at:
            i = int(jnp.searchsorted(checkpoints, n))
            for j, label in enumerate(labels):
                print(f"{n:8d} {label:<20} {float(bias[i,j]):11.5f} "
                      f"{float(variance[i,j]):11.5f} {float(mse[i,j]):11.5f}")
        print("\nFraction of runs with every arm sampled:")
        print(f"{'Samples':>8} {'Full data':>12} {'Both splits':>12}")
        for n in report_at:
            i = int(jnp.searchsorted(checkpoints, n))
            print(f"{n:8d} {float(coverage[i,0]):12.3f} {float(coverage[i,1]):12.3f}")

        print("\nSelection diagnostics (arm 0; uniquely optimal only when gap > 0):")
        print(f"{'Samples':>8} {'P(Double selects 0)':>20} {'Gaussian weight 0':>18} "
              f"{'Predicted D bias*':>18} {'Observed D bias':>16}")
        for n in report_at:
            i = int(jnp.searchsorted(checkpoints, n))
            predicted = -gap * (1 - selected_best[i])
            print(f"{n:8d} {float(selected_best[i]):20.4f} {float(best_weight[i]):18.4f} "
                  f"{float(predicted):18.5f} {float(bias[i,1]):16.5f}")
        print("*Prediction assumes unbiased evaluation, so incomplete coverage can invalidate it.")
        summary.append((mse, diff_mean, diff_se))

    print("\nPaired MSE comparison (positive favors Gaussian weighting):")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for row, n in enumerate([300, num_observations]):
        i = int(jnp.searchsorted(checkpoints, n))
        mses = jnp.stack([item[0][i] for item in summary])
        diffs = jnp.stack([item[1][i] for item in summary])
        ses = jnp.stack([item[2][i] for item in summary])
        print(f"\nAfter {n} observations:")
        print(f"{'Gap':>6} {'Maximum':>12} {'Double':>12} {'Gaussian':>12} {'D - G':>12} {'95% CI (D - G)':>27}")
        for j, gap in enumerate(optimality_gaps):
            d, se = float(diffs[j]), float(ses[j])
            print(f"{gap:6.2f} {float(mses[j,0]):12.6f} {float(mses[j,1]):12.6f} "
                  f"{float(mses[j,2]):12.6f} {d:12.6f} [{d-1.96*se:11.6f}, {d+1.96*se:11.6f}]")
        for j, (label, color) in enumerate(zip(labels, colors)):
            axes[row, 0].plot(optimality_gaps, mses[:, j], "o-", label=label, color=color)
        axes[row, 0].set_ylabel("MSE")
        axes[row, 0].set_title(f"{n} observations")
        axes[row, 1].errorbar(optimality_gaps, diffs, yerr=1.96 * ses, fmt="o-", capsize=4)
        axes[row, 1].axhline(0, color="gray", linestyle="--")
        axes[row, 1].set_ylabel("Double MSE − Gaussian MSE")
        axes[row, 1].set_title(f"{n} observations (pointwise 95% CI)")
    axes[0, 0].legend()
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Optimality gap Δ")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
