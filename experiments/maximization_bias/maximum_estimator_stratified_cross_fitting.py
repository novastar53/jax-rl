"""Compare random and per-arm stratified cross-fitting on shared observations."""

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

    folds = (2,) + extra_folds
    stratified_stats = tuple(
        (jnp.zeros((k, num_arms), dtype=jnp.float32),
         jnp.zeros((k, num_arms), dtype=jnp.int32))
        for k in folds
    )
    # Random starting offsets avoid systematically filling fold zero first.
    # They depend only on the split key, never on rewards.
    offsets = tuple(
        jax.random.randint(
            jax.random.fold_in(split_key, 1000 + k), (num_arms,), 0, k
        )
        for k in folds
    )

    zero = jnp.zeros(num_arms, dtype=jnp.float32)
    counts = jnp.zeros(num_arms, dtype=jnp.int32)
    initial = (zero, counts, zero, zero, counts, zero, counts, fold_stats, stratified_stats)

    def observe(t, carry):
        mean, count, m2, mean1, count1, mean2, count2, fold_stats, stratified_stats = carry
        action, reward = actions[t], rewards[t]

        arm_count = count[action]
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
        def update_stratified(stats, k, offset):
            sums, counts = stats
            fold = (arm_count + offset[action]) % k
            return sums.at[fold, action].add(reward), counts.at[fold, action].add(1)

        stratified_stats = tuple(
            update_stratified(stats, k, offset)
            for stats, k, offset in zip(stratified_stats, folds, offsets)
        )
        return mean, count, m2, mean1, count1, mean2, count2, fold_stats, stratified_stats

    def measure(carry, stop):
        start, stats = carry
        stats = jax.lax.fori_loop(start, stop, observe, stats)
        mean, count, m2, mean1, count1, mean2, count2, fold_stats, stratified_stats = stats

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
        stratified = tuple(cross_fitted_estimate(sums, counts)
                           for sums, counts in stratified_stats)
        estimates = jnp.stack([maximum, double, gaussian, symmetric_double,
                               *(x[0] for x in cf), *(x[0] for x in stratified)])
        eval_available = jnp.stack([count2[selected1] > 0, count1[selected2] > 0])
        cf_accuracy = jnp.stack([
            0.5 * ((selected1 == 0).astype(jnp.float32)
                   + (selected2 == 0).astype(jnp.float32)),
            *(x[1] for x in cf), *(x[1] for x in stratified),
        ])
        cf_eval_fraction = jnp.stack([eval_available.mean(), *(x[2] for x in cf),
                                      *(x[2] for x in stratified)])
        cf_complete = jnp.stack([eval_available.all(), *(x[3] for x in cf),
                                 *(x[3] for x in stratified)])
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
    num_folds = 1 + len(extra_folds)
    random_errors = squared_errors[:, :, 3:3 + num_folds]
    stratified_errors = squared_errors[:, :, 3 + num_folds:]
    diff = random_errors - stratified_errors
    diff_gaussian = squared_errors[:, :, 2:3] - stratified_errors

    def summarize(diff):
        return diff.mean(axis=0), diff.std(axis=0, ddof=1) / jnp.sqrt(num_runs)

    return (bias, variance, mse, mse_se, accuracy.mean(axis=0),
            eval_fraction.mean(axis=0), complete.mean(axis=0),
            *summarize(diff), *summarize(diff_gaussian))


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
    k_count = len(folds)
    random_slice = slice(3, 3 + k_count)
    stratified_slice = slice(3 + k_count, 3 + 2 * k_count)
    summary = []

    for gap in optimality_gaps:
        (bias, variance, mse, mse_se, accuracy, eval_fraction, complete,
         diff, diff_se, dg, dg_se) = run_experiment(
            jax.random.key(44), reward_std, gap, checkpoints,
            num_runs=num_runs, num_arms=num_arms, num_observations=num_observations,
            reward_mean=reward_mean, num_gaussian_samples=num_gaussian_samples,
            initial_se=initial_se, extra_folds=extra_folds,
        )
        assert bool(jnp.allclose(mse, bias ** 2 + variance, rtol=1e-5, atol=1e-6))
        summary.append((mse, diff, diff_se, dg, dg_se))

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for ax, metric, ylabel in zip(
            axes.flat, (bias, variance, mse, eval_fraction),
            ("Bias", "Variance", "MSE", "Selected-action evaluation coverage"),
        ):
            for j, k in enumerate(folds):
                offset = 0 if ylabel.startswith("Selected") else 3
                ax.plot(checkpoints, metric[:, offset+j], "--",
                        color=f"C{j}", label=f"Random K={k}")
                ax.plot(checkpoints, metric[:, offset+k_count+j], "-",
                        color=f"C{j}", label=f"Stratified K={k}")
            if not ylabel.startswith("Selected"):
                ax.plot(checkpoints, metric[:, 2], color="black", label="Gaussian")
                ax.plot(checkpoints, metric[:, 0], color="gray", linestyle=":", label="Maximum")
            ax.set_xscale("log")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
        axes[0, 0].axhline(0, color="gray", linewidth=0.8)
        axes[1, 1].set_ylim(-0.02, 1.02)
        for ax in axes[1]:
            ax.set_xlabel("Total reward observations")
        axes[0, 0].legend(ncol=2, fontsize=8)
        fig.suptitle(f"Random vs stratified cross-fitting (σ={reward_std:g}, Δ={gap:g})")
        fig.tight_layout()

        print(f"\nOptimality gap: {gap:g}; true maximum: {reward_mean + gap:g}")
        print(f"{'Samples':>8} {'K':>3} {'Assignment':>12} {'Bias':>10} {'Variance':>10} "
              f"{'MSE':>10} {'Select 0':>10} {'Eval covered':>13} {'All covered':>12}")
        for n in report_at:
            i = int(jnp.searchsorted(checkpoints, n))
            for j, k in enumerate(folds):
                for assignment, d in (("Random", j), ("Stratified", k_count+j)):
                    e = d + 3
                    print(f"{n:8d} {k:3d} {assignment:>12} {float(bias[i,e]):10.5f} "
                          f"{float(variance[i,e]):10.5f} {float(mse[i,e]):10.5f} "
                          f"{float(accuracy[i,d]):10.4f} {float(eval_fraction[i,d]):13.4f} "
                          f"{float(complete[i,d]):12.4f}")
        print("Arm 0 is uniquely optimal only for Δ > 0; unobserved means are zero.")

    horizons = sorted(set([min(300, num_observations), num_observations]))
    fig, axes = plt.subplots(len(horizons), 2, figsize=(12, 4.5 * len(horizons)),
                             squeeze=False, sharex=True)
    for row, n in enumerate(horizons):
        i = int(jnp.searchsorted(checkpoints, n))
        mses, diff, diff_se, dg, dg_se = [
            jnp.stack([item[j][i] for item in summary]) for j in range(5)
        ]
        print(f"\nMSE after {n} observations (random vs stratified):")
        print(f"{'Gap':>6} {'K':>3} {'Maximum':>11} {'Gaussian':>11} {'Random':>11} "
              f"{'Stratified':>11} {'R - S':>11} {'95% CI (R - S)':>27}")
        for g, gap in enumerate(optimality_gaps):
            for j, k in enumerate(folds):
                d, se = float(diff[g,j]), float(diff_se[g,j])
                print(f"{gap:6.2f} {k:3d} {float(mses[g,0]):11.6f} {float(mses[g,2]):11.6f} "
                      f"{float(mses[g,3+j]):11.6f} {float(mses[g,3+k_count+j]):11.6f} "
                      f"{d:11.6f} [{d-1.96*se:11.6f}, {d+1.96*se:11.6f}]")
        print("\nGaussian MSE - stratified MSE (positive favors stratification):")
        for j, k in enumerate(folds):
            for g, gap in enumerate(optimality_gaps):
                d, se = float(dg[g,j]), float(dg_se[g,j])
                print(f"Δ={gap:g}, K={k}: {d:.6f} [{d-1.96*se:.6f}, {d+1.96*se:.6f}]")
            axes[row, 0].plot(optimality_gaps, mses[:, random_slice][:,j], "o--",
                              color=f"C{j}", label=f"Random K={k}")
            axes[row, 0].plot(optimality_gaps, mses[:, stratified_slice][:,j], "o-",
                              color=f"C{j}", label=f"Stratified K={k}")
            axes[row, 1].errorbar(optimality_gaps, diff[:,j], yerr=1.96*diff_se[:,j],
                                  fmt="o-", capsize=3, color=f"C{j}", label=f"K={k}")
        axes[row, 0].plot(optimality_gaps, mses[:,2], "k-", label="Gaussian")
        axes[row, 0].plot(optimality_gaps, mses[:,0], color="gray", linestyle=":", label="Maximum")
        axes[row, 0].set_title(f"MSE after {n} observations")
        axes[row, 0].set_ylabel("MSE")
        axes[row, 1].axhline(0, color="gray", linestyle="--")
        axes[row, 1].set_title(f"Random − stratified after {n} observations")
        axes[row, 1].set_ylabel("Paired MSE difference (95% CI)")
    for ax in axes[0]:
        ax.legend(ncol=2, fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Optimality gap Δ")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
