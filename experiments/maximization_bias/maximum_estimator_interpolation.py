"""Compare fixed, oracle and plug-in interpolation of maximum and symmetric Double."""

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


def optimal_coefficient(max_error, sd_error, axis=0):
    difference = max_error - sd_error
    denominator = jnp.mean(difference ** 2, axis=axis)
    numerator = -jnp.mean(sd_error * difference, axis=axis)
    return jnp.where(denominator > 0,
                     jnp.clip(numerator / jnp.maximum(denominator, 1e-30), 0., 1.),
                     0.5)


def bootstrap_coefficient(key, mean, count1, count2, m2, num_draws):
    count = count1 + count2
    degrees = jnp.maximum(count - 1, 0).sum()
    variance = jnp.where(degrees > 0,
                         jnp.maximum(m2.sum(), 0.) / jnp.maximum(degrees, 1), 1.)
    noise = jax.random.normal(key, (2, num_draws, mean.size), dtype=mean.dtype)
    def split_mean(n, z):
        return jnp.where(n > 0, mean + jnp.sqrt(variance / jnp.maximum(n, 1)) * z, 0.)
    first = split_mean(count1, noise[0])
    second = split_mean(count2, noise[1])
    pooled = (first * count1 + second * count2) / jnp.maximum(count, 1)
    selected1 = jnp.argmax(first, axis=-1)
    selected2 = jnp.argmax(second, axis=-1)
    sd = 0.5 * (jnp.take_along_axis(second, selected1[:, None], axis=1)[:, 0]
                + jnp.take_along_axis(first, selected2[:, None], axis=1)[:, 0])
    target = jnp.max(mean)
    coefficient = optimal_coefficient(jnp.max(pooled, axis=1) - target, sd - target)
    # No data-driven estimate of the unseen arms or noise is available yet.
    return jnp.where(jnp.all(count > 0) & (degrees > 0), coefficient, 0.5)


def run_one(key, reward_std, optimality_gap, checkpoints, num_arms, num_observations,
            reward_mean, num_gaussian_samples, initial_se, num_bootstrap_samples,
            estimate_plugin=True):
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
        cross12 = mean2[selected1]
        cross21 = mean1[selected2]
        double = jnp.where(select_first, cross12, cross21)
        symmetric_double = 0.5 * (cross12 + cross21)
        gaussian, _ = gaussian_weighted_estimate(
            gaussian_key, mean, count, m2, num_gaussian_samples, initial_se
        )
        estimates = jnp.stack([maximum, double, gaussian, symmetric_double])
        coverage = jnp.stack([
            jnp.all(count > 0),
            jnp.all((count1 > 0) & (count2 > 0)),
        ])
        if estimate_plugin:
            bootstrap_key = jax.random.fold_in(jax.random.fold_in(diagnostic_key, stop), 917)
            coefficient = bootstrap_coefficient(
                bootstrap_key, mean, count1, count2, m2, num_bootstrap_samples
            )
        else:
            coefficient = jnp.float32(0.5)
        return (stop, stats), (estimates, coverage, coefficient)

    _, results = jax.lax.scan(
        measure, (jnp.int32(0), initial), checkpoints
    )
    return results


LABELS = ["Maximum", "Double", "Gaussian-weighted", "Symmetric Double",
          "Fixed 0.25", "Fixed 0.50", "Fixed 0.75", "Oracle (calibrated)", "Plug-in bootstrap"]


@partial(jax.jit, static_argnames=(
    "num_runs", "num_arms", "num_observations", "num_gaussian_samples",
    "num_bootstrap_samples", "estimate_plugin",
))
def collect(key, reward_std, optimality_gap, checkpoints, num_runs=1000,
            num_arms=10, num_observations=5000, reward_mean=-0.1,
            num_gaussian_samples=100, initial_se=1.0, num_bootstrap_samples=128,
            estimate_plugin=True):
    keys = jax.random.split(key, num_runs)
    return jax.vmap(lambda k: run_one(
        k, reward_std, optimality_gap, checkpoints, num_arms, num_observations,
        reward_mean, num_gaussian_samples, initial_se, num_bootstrap_samples,
        estimate_plugin,
    ))(keys)


@jax.jit
def evaluate(calibration, estimates, coverage, plugin, target):
    cm, cd = calibration[:, :, 0] - target, calibration[:, :, 3] - target
    oracle = optimal_coefficient(cm, cd)
    bm, bd = cm.mean(axis=0), cd.mean(axis=0)
    delta_bias = bm - bd
    bias_cancel = jnp.where(jnp.abs(delta_bias) > 1e-8,
                            jnp.clip(-bd / jnp.where(jnp.abs(delta_bias) > 1e-8,
                                                     delta_bias, 1.), 0., 1.), 0.5)
    maximum, sd = estimates[:, :, 0], estimates[:, :, 3]
    fixed = sd[:, :, None] + (maximum - sd)[:, :, None] * jnp.array([.25, .5, .75])
    all_estimates = jnp.concatenate([
        estimates, fixed,
        (sd + oracle * (maximum - sd))[:, :, None],
        (sd + plugin * (maximum - sd))[:, :, None],
    ], axis=-1)
    errors = all_estimates - target
    squared = errors ** 2
    bias = errors.mean(axis=0)
    variance = all_estimates.var(axis=0)
    mse = squared.mean(axis=0)
    covariance = ((maximum - maximum.mean(axis=0)) * (sd - sd.mean(axis=0))).mean(axis=0)
    # Positive difference favors the interpolated estimator.
    diff = squared[:, :, 2:3] - squared[:, :, 4:]
    diff_mean = diff.mean(axis=0)
    diff_se = diff.std(axis=0, ddof=1) / jnp.sqrt(estimates.shape[0])
    return (bias, variance, mse, oracle, bias_cancel, plugin.mean(axis=0),
            plugin.std(axis=0), covariance, coverage.mean(axis=0), diff_mean, diff_se)


def main():
    num_runs = 1000
    num_calibration_runs = 1000
    num_arms = 10
    num_observations = 5000
    reward_mean = -0.1
    reward_std = 1.0
    num_gaussian_samples = 100
    num_bootstrap_samples = 128
    initial_se = 1.0
    optimality_gaps = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    report_at = [1, 10, 50, 100, 300, 1000, 5000]
    checkpoints = jnp.unique(jnp.concatenate([
        jnp.array(report_at, dtype=jnp.int32),
        jnp.rint(jnp.geomspace(1, num_observations, 50)).astype(jnp.int32),
    ]))
    summary = []
    for gap in optimality_gaps:
        kwargs = dict(num_arms=num_arms, num_observations=num_observations,
                      reward_mean=reward_mean, num_gaussian_samples=num_gaussian_samples,
                      initial_se=initial_se, num_bootstrap_samples=num_bootstrap_samples)
        calibration, _, _ = collect(jax.random.key(45), reward_std, gap, checkpoints,
                                    num_runs=num_calibration_runs, estimate_plugin=False, **kwargs)
        estimates, coverage, plugin = collect(jax.random.key(44), reward_std, gap, checkpoints,
                                              num_runs=num_runs, **kwargs)
        (bias, variance, mse, oracle, bias_cancel, plugin_mean, plugin_std,
         covariance, coverage, diff_mean, diff_se) = evaluate(
            calibration, estimates, coverage, plugin, reward_mean + gap
        )
        assert bool(jnp.allclose(mse, bias ** 2 + variance, rtol=1e-5, atol=1e-6))
        summary.append((mse, oracle, plugin_mean, diff_mean, diff_se))
        print(f"\nOptimality gap: {gap:g}; true maximum: {reward_mean + gap:g}")
        print(f"{'Samples':>8} {'Estimator':<23} {'Bias':>11} {'Variance':>11} {'MSE':>11}")
        for n in report_at:
            i = int(jnp.searchsorted(checkpoints, n))
            for j, label in enumerate(LABELS):
                print(f"{n:8d} {label:<23} {float(bias[i,j]):11.5f} "
                      f"{float(variance[i,j]):11.5f} {float(mse[i,j]):11.5f}")
        print("\nCoefficient diagnostics (oracle fitted on separate runs):")
        print(f"{'Samples':>8} {'Oracle':>9} {'Bias-cancel':>12} {'Plug-in':>9} {'Plug-in SD':>11} {'Cov(M,SD)':>12} {'Full coverage':>14}")
        for n in report_at:
            i = int(jnp.searchsorted(checkpoints, n))
            print(f"{n:8d} {float(oracle[i]):9.4f} {float(bias_cancel[i]):12.4f} "
                  f"{float(plugin_mean[i]):9.4f} {float(plugin_std[i]):11.4f} "
                  f"{float(covariance[i]):12.5f} {float(coverage[i,0]):14.4f}")
        fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
        for j, label in enumerate(LABELS):
            for ax, metric in zip(axes[:3], (bias, variance, mse)):
                ax.plot(checkpoints, metric[:, j], label=label, color=f"C{j}",
                        linestyle="--" if 4 <= j <= 6 else "-")
        axes[0].axhline(0, color="gray", linestyle=":")
        axes[0].legend(ncol=3, fontsize=8)
        axes[3].plot(checkpoints, oracle, label="Calibrated oracle", color="C7")
        axes[3].plot(checkpoints, bias_cancel, label="Bias-cancel (calibration)", linestyle=":", color="C3")
        axes[3].plot(checkpoints, plugin_mean, label="Mean plug-in", color="C8")
        axes[3].fill_between(checkpoints, jnp.maximum(0., plugin_mean - plugin_std),
                             jnp.minimum(1., plugin_mean + plugin_std), alpha=.15, color="C8",
                             label="±1 run SD (clipped)")
        axes[3].set_ylim(-.03, 1.03)
        axes[3].legend(fontsize=8)
        for ax, ylabel in zip(axes, ("Bias", "Variance", "MSE", "Coefficient λ")):
            ax.set_ylabel(ylabel)
            ax.set_xscale("log")
            ax.grid(alpha=.3)
        axes[-1].set_xlabel("Total reward observations")
        fig.suptitle(f"Interpolated maximum estimation (σ={reward_std:g}, Δ={gap:g})")
        fig.tight_layout()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, n in enumerate((300, num_observations)):
        i = int(jnp.searchsorted(checkpoints, n))
        print(f"\nMSE after {n} observations:")
        print(f"{'Gap':>6} " + " ".join(f"{x:>12}" for x in
              ["Max", "Double", "Gaussian", "SD", "λ=.25", "λ=.5", "λ=.75", "Oracle", "Plug-in"]))
        for gap, (mse, oracle, plugin, dm, ds) in zip(optimality_gaps, summary):
            print(f"{gap:6.2f} " + " ".join(f"{float(v):12.6f}" for v in mse[i]))
        print("\nGaussian MSE - interpolated MSE (positive favors interpolation; pointwise 95% CI):")
        for gap, (mse, oracle, plugin, dm, ds) in zip(optimality_gaps, summary):
            for j, label in enumerate(LABELS[4:]):
                d, se = float(dm[i,j]), float(ds[i,j])
                print(f"Δ={gap:g} {label:<23} {d: .6f} [{d-1.96*se: .6f}, {d+1.96*se: .6f}]")
        mses = jnp.stack([s[0][i] for s in summary])
        for j, label in enumerate(LABELS):
            axes[row,0].plot(optimality_gaps, mses[:,j], marker="o", label=label, color=f"C{j}",
                             linestyle="--" if 4 <= j <= 6 else "-")
        axes[row,1].plot(optimality_gaps, [float(s[1][i]) for s in summary], "o-", label="Oracle", color="C7")
        axes[row,1].plot(optimality_gaps, [float(s[2][i]) for s in summary], "o-", label="Mean plug-in", color="C8")
        axes[row,0].set_ylabel("MSE")
        axes[row,1].set_ylabel("Coefficient λ")
        axes[row,1].set_ylim(-.03, 1.03)
        for ax in axes[row]:
            ax.set_title(f"{n} observations")
            ax.set_xlabel("Optimality gap Δ")
            ax.grid(alpha=.3)
    axes[0,0].legend(fontsize=7, ncol=2)
    axes[0,1].legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
