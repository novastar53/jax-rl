"""Compare maximum-expectation estimators with K-fold cross-fitting."""

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


def cross_fitted_estimate(sums, counts):
    train_counts = counts.sum(axis=0) - counts
    train_sums = sums.sum(axis=0) - sums
    train_means = train_sums / jnp.maximum(train_counts, 1)
    selected = jnp.argmax(train_means, axis=1)
    folds = jnp.arange(counts.shape[0])
    eval_counts = counts[folds, selected]
    eval_means = sums[folds, selected] / jnp.maximum(eval_counts, 1)
    return (
        eval_means.mean(),
        (selected == 0).mean(),
        (eval_counts > 0).mean(),
        jnp.all(eval_counts > 0),
    )


def run_one(key, reward_std, optimality_gap, checkpoints, num_arms, num_observations,
            reward_mean, num_gaussian_samples, initial_se, extra_folds=(3, 5, 10)):
    action_key, reward_key, split_key, diagnostic_key = jax.random.split(key, 4)
    actions = jax.random.randint(action_key, (num_observations,), 0, num_arms)
    rewards = reward_mean + optimality_gap * (actions == 0) + reward_std * jax.random.normal(
        reward_key, (num_observations,), dtype=jnp.float32
    )
    split = jax.random.bernoulli(split_key, 0.5, (num_observations,))

    fold_ids = tuple(
        jax.random.randint(jax.random.fold_in(split_key, k),
                           (num_observations,), 0, k)
        for k in extra_folds
    )
    fold_stats = tuple(
        (jnp.zeros((k, num_arms), dtype=jnp.float32),
         jnp.zeros((k, num_arms), dtype=jnp.int32))
        for k in extra_folds
    )

    zero = jnp.zeros(num_arms, dtype=jnp.float32)
    counts = jnp.zeros(num_arms, dtype=jnp.int32)
    initial = (zero, counts, zero, zero, counts, zero, counts, fold_stats)

    def observe(t, carry):
        mean, count, m2, mean1, count1, mean2, count2, fold_stats = carry
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
        fold_stats = tuple(
            (sums.at[ids[t], action].add(reward),
             counts.at[ids[t], action].add(1))
            for (sums, counts), ids in zip(fold_stats, fold_ids)
        )
        return mean, count, m2, mean1, count1, mean2, count2, fold_stats

    def measure(carry, stop):
        start, stats = carry
        stats = jax.lax.fori_loop(start, stop, observe, stats)
        mean, count, m2, mean1, count1, mean2, count2, fold_stats = stats

        direction_key, gaussian_key = jax.random.split(
            jax.random.fold_in(diagnostic_key, stop)
        )
        maximum = jnp.max(mean)
        select_first = jax.random.bernoulli(direction_key)
        selected1 = jnp.argmax(mean1)
        selected2 = jnp.argmax(mean2)
        cross12 = mean2[selected1]
        cross21 = mean1[selected2]
        double = jnp.where(select_first, cross12, cross21)
        symmetric_double = 0.5 * (cross12 + cross21)
        selected = jnp.where(select_first, selected1, selected2)
        gaussian, optimal_weight = gaussian_weighted_estimate(
            gaussian_key, mean, count, m2, num_gaussian_samples, initial_se
        )
        cf = tuple(cross_fitted_estimate(sums, counts)
                   for sums, counts in fold_stats)
        estimates = jnp.stack([maximum, double, gaussian, symmetric_double,
                               *(x[0] for x in cf)])
        eval_available = jnp.stack([count2[selected1] > 0, count1[selected2] > 0])
        cf_accuracy = jnp.stack([
            0.5 * ((selected1 == 0).astype(jnp.float32)
                   + (selected2 == 0).astype(jnp.float32)),
            *(x[1] for x in cf),
        ])
        cf_eval_fraction = jnp.stack([eval_available.mean(), *(x[2] for x in cf)])
        cf_complete = jnp.stack([eval_available.all(), *(x[3] for x in cf)])
        coverage = jnp.stack([
            jnp.all(count > 0),
            jnp.all((count1 > 0) & (count2 > 0)),
        ])
        selection = jnp.stack([
            (selected == 0).astype(jnp.float32),
            0.5 * ((selected1 == 0).astype(jnp.float32)
                   + (selected2 == 0).astype(jnp.float32)),
            ((selected1 == 0) & (selected2 == 0)).astype(jnp.float32),
        ])
        return (stop, stats), (estimates, coverage, selection, optimal_weight,
                                cf_accuracy, cf_eval_fraction, cf_complete)

    _, results = jax.lax.scan(
        measure, (jnp.int32(0), initial), checkpoints
    )
    return results


@partial(jax.jit, static_argnames=(
    "num_runs", "num_arms", "num_observations", "num_gaussian_samples", "extra_folds",
))
def run_experiment(key, reward_std, optimality_gap, checkpoints, num_runs=1000,
                   num_arms=10, num_observations=5000, reward_mean=-0.1,
                   num_gaussian_samples=100, initial_se=1.0, extra_folds=(3, 5, 10)):
    keys = jax.random.split(key, num_runs)
    (estimates, coverage, selection, best_weight, accuracy,
     eval_fraction, complete) = jax.vmap(
        lambda k: run_one(
            k, reward_std, optimality_gap, checkpoints, num_arms, num_observations,
            reward_mean, num_gaussian_samples, initial_se, extra_folds,
        )
    )(keys)
    errors = estimates - (reward_mean + optimality_gap)
    squared_errors = errors ** 2
    bias = errors.mean(axis=0)
    variance = estimates.var(axis=0)
    mse = squared_errors.mean(axis=0)
    mse_se = squared_errors.std(axis=0, ddof=1) / jnp.sqrt(num_runs)
    # Positive differences favor the cross-fitted estimator.
    diff_gaussian = squared_errors[:, :, 2:3] - squared_errors[:, :, 3:]
    diff_symmetric = squared_errors[:, :, 3:4] - squared_errors[:, :, 3:]

    def summarize(diff):
        return diff.mean(axis=0), diff.std(axis=0, ddof=1) / jnp.sqrt(num_runs)

    return (bias, variance, mse, mse_se, accuracy.mean(axis=0),
            eval_fraction.mean(axis=0), complete.mean(axis=0),
            *summarize(diff_gaussian), *summarize(diff_symmetric))


def main():
    num_runs = 1000
    num_arms = 10
    num_observations = 5000
    reward_mean = -0.1
    reward_std = 1.0
    num_gaussian_samples = 100
    initial_se = 1.0
    extra_folds = (3, 5, 10)
    folds = (2,) + extra_folds
    optimality_gaps = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    report_at = sorted(set([n for n in (1, 10, 50, 100, 300, 1000)
                            if n <= num_observations] + [num_observations]))
    checkpoints = jnp.unique(jnp.concatenate([
        jnp.array(report_at, dtype=jnp.int32),
        jnp.rint(jnp.geomspace(1, num_observations, 100)).astype(jnp.int32),
    ]))
    labels = ["Maximum", "Double", "Gaussian-weighted", "CF 2 (Symmetric Double)"] + [
        f"CF {k}" for k in extra_folds
    ]
    summary = []
    for gap in optimality_gaps:
        (bias, variance, mse, mse_se, accuracy, eval_fraction, complete,
         dg, dg_se, ds, ds_se) = run_experiment(
            jax.random.key(44), reward_std, gap, checkpoints,
            num_runs=num_runs, num_arms=num_arms, num_observations=num_observations,
            reward_mean=reward_mean, num_gaussian_samples=num_gaussian_samples,
            initial_se=initial_se, extra_folds=extra_folds,
        )
        assert bool(jnp.allclose(mse, bias ** 2 + variance, rtol=1e-5, atol=1e-6))
        summary.append((mse, dg, dg_se, ds, ds_se))

        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        for j, label in enumerate(labels):
            for ax, metric in zip(axes, (bias, variance, mse)):
                ax.plot(checkpoints, metric[:, j], label=label, color=f"C{j}")
        axes[0].axhline(0, color="gray", linestyle="--")
        for ax, ylabel in zip(axes, ("Bias", "Variance", "MSE")):
            ax.set_ylabel(ylabel)
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
        axes[2].set_xlabel("Total reward observations")
        axes[0].legend(ncol=2)
        fig.suptitle(f"Cross-fitted estimators (σ={reward_std:g}, Δ={gap:g})")
        fig.tight_layout()

        print(f"\nOptimality gap: {gap:g}; true maximum: {reward_mean + gap:g}")
        print(f"{'Samples':>8} {'Estimator':<24} {'Bias':>11} {'Variance':>11} {'MSE':>11}")
        for n in report_at:
            i = int(jnp.searchsorted(checkpoints, n))
            for j, label in enumerate(labels):
                print(f"{n:8d} {label:<24} {float(bias[i,j]):11.5f} "
                      f"{float(variance[i,j]):11.5f} {float(mse[i,j]):11.5f}")
        print("\nCross-fitting diagnostics (selection of arm 0; uniquely optimal only for Δ > 0):")
        print(f"{'Samples':>8} {'K':>3} {'Select 0':>10} {'Eval covered':>13} "
              f"{'All covered':>12} {'Pred. bias':>12} {'Obs. bias':>12}")
        for n in report_at:
            i = int(jnp.searchsorted(checkpoints, n))
            for j, k in enumerate(folds):
                pred = -gap * (1 - accuracy[i, j])
                print(f"{n:8d} {k:3d} {float(accuracy[i,j]):10.4f} "
                      f"{float(eval_fraction[i,j]):13.4f} {float(complete[i,j]):12.4f} "
                      f"{float(pred):12.5f} {float(bias[i,j+3]):12.5f}")
        print("Unobserved means are zero. Bias prediction assumes unbiased held-out means.")

    horizons = sorted(set([min(300, num_observations), num_observations]))
    fig, axes = plt.subplots(len(horizons), 3, figsize=(15, 4 * len(horizons)),
                             squeeze=False, sharex=True)
    for row, n in enumerate(horizons):
        i = int(jnp.searchsorted(checkpoints, n))
        mses, dg, dg_se, ds, ds_se = [
            jnp.stack([item[j][i] for item in summary]) for j in range(5)
        ]
        print(f"\nMSE after {n} observations:")
        print(f"{'Gap':>6} " + " ".join(f"{x:>12}" for x in
              ["Maximum", "Double", "Gaussian"] + [f"CF {k}" for k in folds]))
        for j, gap in enumerate(optimality_gaps):
            print(f"{gap:6.2f} " + " ".join(f"{float(v):12.6f}" for v in mses[j]))
        for j, label in enumerate(labels):
            axes[row, 0].plot(optimality_gaps, mses[:, j], "o-", label=label, color=f"C{j}")
        axes[row, 0].set_title(f"MSE: {n} observations")
        axes[row, 0].set_ylabel("MSE")
        for col, diffs, ses, baseline, start in [
            (1, dg, dg_se, "Gaussian", 0), (2, ds, ds_se, "CF 2", 1)
        ]:
            print(f"\n{baseline} MSE - CF MSE (paired pointwise 95% CI; positive favors CF):")
            print(f"{'Gap':>6} {'K':>3} {'Difference':>12} {'95% CI':>28}")
            for j, k in enumerate(folds):
                if j < start:
                    continue
                axes[row, col].errorbar(optimality_gaps, diffs[:, j], yerr=1.96 * ses[:, j],
                                        fmt="o-", capsize=3, label=f"CF {k}", color=f"C{j+3}")
                for g, gap in enumerate(optimality_gaps):
                    d, se = float(diffs[g,j]), float(ses[g,j])
                    print(f"{gap:6.2f} {k:3d} {d:12.6f} [{d-1.96*se:11.6f}, {d+1.96*se:11.6f}]")
            axes[row, col].axhline(0, color="gray", linestyle="--")
            axes[row, col].set_title(f"{baseline} − CF: {n} observations")
            axes[row, col].set_ylabel("Paired MSE difference")
    for ax in axes[0]:
        ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Optimality gap Δ")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
