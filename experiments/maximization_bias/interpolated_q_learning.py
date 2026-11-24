"""Compare interpolated TD backups in a two-step maximization-bias MDP.

Standalone JAX experiment. B terminates, so its Q values are sample means.
All methods share the same per-run, per-episode random streams, not necessarily
identical observations (their behavior policies can choose different actions).
"""

import argparse
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
METHODS = tuple((f"Lambda={x:g}", "interpolated", x) for x in LAMBDAS) + (
    ("Double Q", "double", 0.0),
    ("Gaussian", "gaussian", 0.0),
)


class State(NamedTuple):
    a_q: jax.Array          # two A actions, single shared table
    a_n: jax.Array
    a_double_q: jax.Array   # two independent A tables for conventional Double Q
    a_double_n: jax.Array
    b_mean: jax.Array       # all observations of each B action
    b_n: jax.Array
    b_m2: jax.Array
    b_split_mean: jax.Array # independent random partitions of B rewards
    b_split_n: jax.Array


def init_state(num_arms):
    return State(jnp.zeros(2), jnp.zeros(2, dtype=jnp.int32),
                 jnp.zeros((2, 2)), jnp.zeros((2, 2), dtype=jnp.int32),
                 jnp.zeros(num_arms), jnp.zeros(num_arms, dtype=jnp.int32),
                 jnp.zeros(num_arms), jnp.zeros((2, num_arms)),
                 jnp.zeros((2, num_arms), dtype=jnp.int32))


def epsilon_greedy(key, q, epsilon):
    greedy = q == jnp.max(q)
    probs = epsilon / q.size + (1 - epsilon) * greedy / greedy.sum()
    return jax.random.choice(key, q.size, p=probs)


def greedy_left_probability(q):
    greedy = q == jnp.max(q)
    return greedy[0] / greedy.sum()


def update_mean(mean, count, m2, split_mean, split_count, action, reward, fold):
    n = count[action] + 1
    delta = reward - mean[action]
    new_mean = mean[action] + delta / n
    mean = mean.at[action].set(new_mean)
    count = count.at[action].set(n)
    m2 = m2.at[action].add(delta * (reward - new_mean))
    ns = split_count[fold, action] + 1
    split_count = split_count.at[fold, action].set(ns)
    split_mean = split_mean.at[fold, action].add(
        (reward - split_mean[fold, action]) / ns)
    return mean, count, m2, split_mean, split_count


def cross_estimates(split_mean):
    i0 = jnp.argmax(split_mean[0])
    i1 = jnp.argmax(split_mean[1])
    return jnp.stack([split_mean[1, i0], split_mean[0, i1]])


def gaussian_backup(key, mean, count, m2, num_draws, initial_se):
    variance = jnp.maximum(m2, 0.) / jnp.maximum(count - 1, 1)
    se = jnp.where(count >= 2, jnp.sqrt(variance / jnp.maximum(count, 1)), initial_se)
    draws = mean + se * jax.random.normal(key, (num_draws, mean.size), dtype=mean.dtype)
    return mean[jnp.argmax(draws, axis=1)].mean()


def alpha(n, cap, exponent):
    return jnp.minimum(cap, n.astype(jnp.float32) ** (-exponent))


def run_one(key, branch_value, reward_std, coefficient, epsilon, alpha_cap,
            decay_exponent, arm_gap, initial_se, num_arms, num_episodes,
            num_gaussian_samples, mode):
    # Keeping the best B mean fixed permits an optional heterogeneous-arm test.
    means = jnp.full(num_arms, branch_value - arm_gap).at[0].set(branch_value)
    best_value = jnp.maximum(branch_value, 0.)

    def episode(carry, _):
        state, key = carry
        key, a_key, b_key, reward_key, a_split_key, b_split_key, gaussian_key = jax.random.split(key, 7)
        a_fold = jax.random.bernoulli(a_split_key).astype(jnp.int32)
        b_fold = jax.random.bernoulli(b_split_key).astype(jnp.int32)
        if mode == "double":
            a_values = state.a_double_q.mean(axis=0)
            b_values = state.b_split_mean.mean(axis=0)
            backup = cross_estimates(state.b_split_mean)[a_fold]
        else:
            a_values = state.a_q
            b_values = state.b_mean
            if mode == "gaussian":
                backup = gaussian_backup(gaussian_key, state.b_mean, state.b_n,
                                         state.b_m2, num_gaussian_samples, initial_se)
            else:
                maximum = jnp.max(state.b_mean)
                sd = cross_estimates(state.b_split_mean).mean()
                backup = coefficient * maximum + (1 - coefficient) * sd

        action_a = epsilon_greedy(a_key, a_values, epsilon)
        left = action_a == 0
        greedy_left = greedy_left_probability(a_values)
        target = jnp.where(left, backup, 0.)  # gamma=1, A reward=0
        # Update A before observing the terminal reward at B.
        if mode == "double":
            count = state.a_double_n.at[a_fold, action_a].add(1)
            old = state.a_double_q[a_fold, action_a]
            q = state.a_double_q.at[a_fold, action_a].add(
                alpha(count[a_fold, action_a], alpha_cap, decay_exponent) * (target - old))
            state = state._replace(a_double_n=count, a_double_q=q)
        else:
            count = state.a_n.at[action_a].add(1)
            old = state.a_q[action_a]
            q = state.a_q.at[action_a].add(alpha(count[action_a], alpha_cap, decay_exponent) * (target - old))
            state = state._replace(a_n=count, a_q=q)

        action_b = epsilon_greedy(b_key, b_values, epsilon)
        reward_b = means[action_b] + reward_std * jax.random.normal(reward_key, ())

        def visit_b(s):
            m, n, m2, sm, sn = update_mean(s.b_mean, s.b_n, s.b_m2,
                                           s.b_split_mean, s.b_split_n,
                                           action_b, reward_b, b_fold)
            return s._replace(b_mean=m, b_n=n, b_m2=m2, b_split_mean=sm, b_split_n=sn)

        state = jax.lax.cond(left, visit_b, lambda s: s, state)
        expected_return = jnp.where(left, means[action_b], 0.)
        reward = jnp.where(left, reward_b, 0.)
        regret = best_value - expected_return
        branch_regret = best_value - jnp.where(left, branch_value, 0.)
        optimal_greedy = jnp.where(branch_value > 0, greedy_left, 1 - greedy_left)
        # The backup is a pre-visit estimate of V*(B), even on Right episodes.
        metrics = jnp.stack([left.astype(jnp.float32), optimal_greedy, reward,
                             regret, branch_regret, backup - branch_value])
        return (state, key), metrics

    (_, _), metrics = jax.lax.scan(episode, (init_state(num_arms), key), None,
                                    length=num_episodes)
    return metrics


@partial(jax.jit, static_argnames=("num_runs", "num_arms", "num_episodes", "num_gaussian_samples", "mode"))
def run_batch(key, branch_value, reward_std, coefficient, epsilon=0.1,
              alpha_cap=1., decay_exponent=0.8, arm_gap=0., initial_se=1.,
              num_runs=200, num_arms=10, num_episodes=5000,
              num_gaussian_samples=100, mode="interpolated"):
    keys = jax.random.split(key, num_runs)
    return jax.vmap(lambda k: run_one(k, branch_value, reward_std, coefficient,
                                    epsilon, alpha_cap, decay_exponent, arm_gap,
                                    initial_se, num_arms, num_episodes,
                                    num_gaussian_samples, mode))(keys)


@jax.jit
def summarize(metrics, epsilon, branch_value):
    # Arrays are (runs, episodes, metrics). Excess removes the unavoidable
    # epsilon-greedy branch regret, but not exploration at B in heterogeneous cases.
    left, greedy, reward, regret, branch_regret, error = jnp.moveaxis(metrics, -1, 0)
    cumulative = jnp.cumsum(regret, axis=1)
    branch_excess = jnp.cumsum(branch_regret - jnp.abs(branch_value) * epsilon / 2, axis=1)
    return (left.mean(axis=0), greedy.mean(axis=0), reward.mean(axis=0),
            cumulative.mean(axis=0), branch_excess.mean(axis=0), error.mean(axis=0),
            (error ** 2).mean(axis=0), cumulative[:, -1], branch_excess[:, -1])


def rolling_mean(x, window):
    c = jnp.concatenate([jnp.zeros(1, dtype=x.dtype), jnp.cumsum(x)])
    t = jnp.arange(1, x.size + 1)
    start = jnp.maximum(t - window, 0)
    return (c[t] - c[start]) / (t - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--arm-gap", type=float, default=0., help="Best B arm minus the other nine")
    parser.add_argument("--gaussian-samples", type=int, default=100)
    parser.add_argument("--alpha-cap", type=float, default=1.)
    parser.add_argument("--decay-exponent", type=float, default=.8)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save-prefix", default=None, help="Optional prefix for PNG figures")
    args = parser.parse_args()
    if args.runs < 2 or args.episodes < 1 or args.gaussian_samples < 1 or args.arm_gap < 0 or not 0 < args.alpha_cap <= 1 or not .5 < args.decay_exponent <= 1:
        parser.error("Require runs >= 2, episodes and Gaussian samples >= 1, arm gap >= 0, 0 < alpha cap <= 1, 0.5 < decay exponent <= 1")

    epsilon = .1
    branch_values = (-.1, .1)
    reward_stds = (.5, 1., 2., 4.)
    key = jax.random.key(args.seed)
    t = jnp.arange(1, args.episodes + 1)
    tail = min(300, args.episodes)
    window = min(100, args.episodes)
    final_regrets = {}
    print(f"Runs={args.runs}, episodes={args.episodes}, seed={args.seed}, epsilon={epsilon}, "
          f"A alpha=min({args.alpha_cap:g}, N^-{args.decay_exponent:g}), B alpha=1/N, arm gap={args.arm_gap:g}")

    for branch in branch_values:
        fig, axes = plt.subplots(len(reward_stds), 3, figsize=(15, 12), squeeze=False, sharex=True)
        fig.suptitle(f"Best B mean = {branch:+.1f} ({'Left' if branch > 0 else 'Right'} optimal), arm gap={args.arm_gap:g}")
        for row, sigma in enumerate(reward_stds):
            summaries = []
            print(f"\nBest B mean={branch:+.1f}, reward std={sigma:g}", flush=True)
            print(f"{'Method':<15} {'Left%':>8} {'Greedy opt%':>12} {'Regret':>11} {'Branch excess':>14} {'B bias':>10} {'B MSE':>10}", flush=True)
            for label, mode, coefficient in METHODS:
                metrics = run_batch(key, branch, sigma, coefficient, epsilon=epsilon,
                                    alpha_cap=args.alpha_cap, decay_exponent=args.decay_exponent,
                                    arm_gap=args.arm_gap, num_runs=args.runs,
                                    num_episodes=args.episodes, num_gaussian_samples=args.gaussian_samples,
                                    mode=mode)
                summary = summarize(metrics, epsilon, branch)
                left, greedy, reward, regret, excess, bias, mse, per_run, per_run_excess = summary
                print(f"{label:<15} {100*left[-tail:].mean():8.2f} {100*greedy[-tail:].mean():12.2f} "
                      f"{regret[-1]:11.4f} {excess[-1]:14.4f} {bias[-tail:].mean():10.5f} {mse[-tail:].mean():10.5f}", flush=True)
                summaries.append(summary)
                axes[row, 0].plot(t, rolling_mean(left, window) * 100, label=label)
                axes[row, 1].plot(t, excess, label=label)
                axes[row, 2].plot(t, rolling_mean(mse, window), label=label)
                del metrics

            reference = summaries[4][7]  # Lambda=1, full-data maximum backup
            gaussian = summaries[-1][7]
            print("Paired total pseudo-regret differences (positive favors listed method; pointwise 95% CI):")
            for (label, _, _), summary in zip(METHODS, summaries):
                def comparison(ref):
                    d = ref - summary[7]
                    m = d.mean()
                    half = 1.96 * d.std(ddof=1) / jnp.sqrt(args.runs)
                    return f"{m:.4f} [{m-half:.4f}, {m+half:.4f}]"
                print(f"  {label:<15} Max - method: {comparison(reference):<32} Gaussian - method: {comparison(gaussian)}")
            final_regrets[(branch, sigma)] = [float(s[7].mean()) for s in summaries]
            axes[row, 0].axhline(100 * (1 - epsilon / 2 if branch > 0 else epsilon / 2),
                                 linestyle="--", color="gray", linewidth=1)
            axes[row, 0].set_ylabel(f"sigma={sigma:g}\nLeft (%)")
            axes[row, 1].set_ylabel("Cumulative branch\nexcess regret")
            axes[row, 2].set_ylabel("B backup MSE")
            for ax in axes[row]:
                ax.grid(alpha=.25)
                ax.set_xscale("log")
        for ax in axes[-1]:
            ax.set_xlabel("Episode")
        axes[0, 0].legend(fontsize=8, ncol=2)
        fig.tight_layout()
        if args.save_prefix:
            fig.savefig(f"{args.save_prefix}_{'left' if branch > 0 else 'right'}.png", dpi=160)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, branch in zip(axes, branch_values):
        for i, (label, _, _) in enumerate(METHODS):
            ax.plot(reward_stds, [final_regrets[(branch, sigma)][i] for sigma in reward_stds], marker="o", label=label)
        ax.set_title(f"{'Left' if branch > 0 else 'Right'} optimal")
        ax.set_xlabel("Reward standard deviation")
        ax.set_ylabel("Total pseudo-regret")
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    if args.save_prefix:
        fig.savefig(f"{args.save_prefix}_summary.png", dpi=160)
    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
