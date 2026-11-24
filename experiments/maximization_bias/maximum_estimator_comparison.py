"""Compare maximum-expectation estimators on identical, exogenous observations."""

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
    return jnp.mean(mean[winners])


def run_one(key, reward_std, checkpoints, num_arms, num_observations,
            reward_mean, num_gaussian_samples, initial_se):
    action_key, reward_key, split_key, diagnostic_key = jax.random.split(key, 4)
    actions = jax.random.randint(action_key, (num_observations,), 0, num_arms)
    rewards = reward_mean + reward_std * jax.random.normal(
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
        double = jnp.where(
            jax.random.bernoulli(direction_key),
            mean2[jnp.argmax(mean1)],
            mean1[jnp.argmax(mean2)],
        )
        gaussian = gaussian_weighted_estimate(
            gaussian_key, mean, count, m2, num_gaussian_samples, initial_se
        )
        estimates = jnp.stack([maximum, double, gaussian])
        coverage = jnp.stack([
            jnp.all(count > 0),
            jnp.all((count1 > 0) & (count2 > 0)),
        ])
        return (stop, stats), (estimates, coverage)

    _, results = jax.lax.scan(
        measure, (jnp.int32(0), initial), checkpoints
    )
    return results


@partial(jax.jit, static_argnames=(
    "num_runs", "num_arms", "num_observations", "num_gaussian_samples",
))
def run_experiment(key, reward_std, checkpoints, num_runs=1000,
                   num_arms=10, num_observations=5000, reward_mean=-0.1,
                   num_gaussian_samples=100, initial_se=1.0):
    keys = jax.random.split(key, num_runs)
    estimates, coverage = jax.vmap(
        lambda k: run_one(
            k, reward_std, checkpoints, num_arms, num_observations,
            reward_mean, num_gaussian_samples, initial_se,
        )
    )(keys)

    # Shapes: (runs, checkpoints, estimators).
    errors = estimates - reward_mean
    bias = errors.mean(axis=0)
    variance = estimates.var(axis=0)
    mse = (errors ** 2).mean(axis=0)
    bias_se = errors.std(axis=0, ddof=1) / jnp.sqrt(num_runs)
    mse_se = (errors ** 2).std(axis=0, ddof=1) / jnp.sqrt(num_runs)
    return bias, variance, mse, bias_se, mse_se, coverage.mean(axis=0)


def main():
    num_runs = 1000
    num_arms = 10
    num_observations = 5000
    reward_mean = -0.1
    num_gaussian_samples = 100
    initial_se = 1.0
    reward_stds = [0.5, 1.0, 4.0]
    report_at = [1, 10, 50, 100, 300, 1000, 5000]
    checkpoints = jnp.unique(jnp.concatenate([
        jnp.array(report_at, dtype=jnp.int32),
        jnp.rint(jnp.geomspace(1, num_observations, 100)).astype(jnp.int32),
    ]))
    labels = ["Maximum", "Double", "Gaussian-weighted"]
    colors = ["C0", "C1", "C2"]

    for reward_std in reward_stds:
        # Common random streams across noise settings as well as estimators.
        bias, variance, mse, bias_se, mse_se, coverage = run_experiment(
            jax.random.key(44), reward_std, checkpoints,
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
        fig.suptitle(f"Controlled maximum-estimator comparison (σ={reward_std:g})")
        fig.tight_layout()

        print(f"\nReward standard deviation: {reward_std:g}")
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

    plt.show()


if __name__ == "__main__":
    main()
