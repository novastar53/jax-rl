"""Factorial experiment: uniform versus epsilon-greedy sampling at A and B."""
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
            num_gaussian_samples, mode, uniform_a=False, uniform_b=False):
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

        action_a = epsilon_greedy(a_key, a_values, 1.0 if uniform_a else epsilon)
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

        action_b = epsilon_greedy(b_key, b_values, 1.0 if uniform_b else epsilon)
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

    (final_state, _), metrics = jax.lax.scan(episode, (init_state(num_arms), key), None,
                                    length=num_episodes)
    return metrics, final_state.b_n, final_state.b_split_n


@partial(jax.jit, static_argnames=("num_runs", "num_arms", "num_episodes", "num_gaussian_samples", "mode", "uniform_a", "uniform_b"))
def run_batch(key, branch_value, reward_std, coefficient, epsilon=0.1,
              alpha_cap=1., decay_exponent=0.8, arm_gap=0., initial_se=1.,
              num_runs=200, num_arms=10, num_episodes=5000,
              num_gaussian_samples=100, mode="interpolated", uniform_a=False, uniform_b=False):
    keys = jax.random.split(key, num_runs)
    return jax.vmap(lambda k: run_one(k, branch_value, reward_std, coefficient,
                                    epsilon, alpha_cap, decay_exponent, arm_gap,
                                    initial_se, num_arms, num_episodes,
                                    num_gaussian_samples, mode, uniform_a, uniform_b))(keys)


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


POLICIES = (
    ('Original', False, False), ('Uniform B', False, True),
    ('Uniform A', True, False), ('Uniform both', True, True),
)


@jax.jit
def visit_statistics(metrics, checkpoints):
    """First pre-update backup after exactly n B observations.

    Unreached checkpoints (including a visit on the final episode without a
    subsequent diagnostic) are censored. Report the fraction of runs retained.
    """
    left = metrics[:, :, 0].astype(jnp.int32)
    before = jnp.cumsum(left, axis=1) - left
    errors = metrics[:, :, 5]

    def gather(counts, error):
        idx = jnp.searchsorted(counts, checkpoints, side='left')
        valid = idx < counts.size
        idx = jnp.minimum(idx, counts.size - 1)
        return error[idx], valid & (counts[idx] == checkpoints)

    error, valid = jax.vmap(gather)(before, errors)
    n = valid.sum(axis=0)
    denom = jnp.maximum(n, 1)
    bias = jnp.where(n > 0, jnp.where(valid, error, 0.).sum(axis=0) / denom, jnp.nan)
    mse = jnp.where(n > 0, jnp.where(valid, error**2, 0.).sum(axis=0) / denom, jnp.nan)
    return bias, mse, n / metrics.shape[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=200)
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--reward-stds', type=float, nargs='+', default=[.5, 1., 2., 4.])
    parser.add_argument('--branch-values', type=float, nargs='+', default=[-.1, .1])
    parser.add_argument('--gaussian-samples', type=int, default=100)
    parser.add_argument('--alpha-cap', type=float, default=1.)
    parser.add_argument('--decay-exponent', type=float, default=.8)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--save-prefix', default=None)
    args = parser.parse_args()
    if (args.runs < 2 or args.episodes < 2 or args.gaussian_samples < 1
            or any(s <= 0 for s in args.reward_stds) or any(b == 0 for b in args.branch_values)
            or not 0 < args.alpha_cap <= 1 or not .5 < args.decay_exponent <= 1):
        parser.error('Invalid experimental parameters')
    epsilon = .1
    key = jax.random.key(args.seed)
    t = jnp.arange(1, args.episodes + 1)
    tail = min(300, args.episodes)
    window = min(100, args.episodes)
    checkpoints = jnp.array([n for n in (1, 5, 10, 20, 50, 100, 200, 300, 500, 1000, 2000, 4000) if n < args.episodes])
    print(f'Runs={args.runs}, episodes={args.episodes}, seed={args.seed}, epsilon={epsilon}, arm gap=0')
    print('Greedy loss evaluates the learned policy at each episode, not a separately executed trajectory.')
    print('Visit-matched metrics are conditional on reaching the checkpoint; plots require >=90% reach.')
    for branch in args.branch_values:
        for sigma in args.reward_stds:
            print(f'\nBest B mean={branch:+g}, sigma={sigma:g}', flush=True)
            fig, axes = plt.subplots(4, 4, figsize=(17, 13), squeeze=False)
            fig.suptitle(f'Best B mean={branch:+g}, sigma={sigma:g}')
            losses = {}
            for col, (policy, uniform_a, uniform_b) in enumerate(POLICIES):
                print(f'\n{policy}', flush=True)
                print(f"{'Method':<15} {'Greedy opt%':>11} {'B visits':>9} {'Min N':>8} {'Max N':>8} {'Full cov%':>10} {'Split cov%':>10} {'B bias':>10} {'B MSE':>10} {'Greedy loss':>12} {'Excess':>10}")
                details = []
                for label, mode, coefficient in METHODS:
                    metrics, counts, split_counts = run_batch(
                        key, branch, sigma, coefficient, epsilon=epsilon,
                        alpha_cap=args.alpha_cap, decay_exponent=args.decay_exponent,
                        num_runs=args.runs, num_episodes=args.episodes,
                        num_gaussian_samples=args.gaussian_samples, mode=mode,
                        uniform_a=uniform_a, uniform_b=uniform_b)
                    left, greedy, reward, regret, excess, bias, mse, per_run, per_run_excess = summarize(metrics, epsilon, branch)
                    loss = abs(branch) * (1 - metrics[:, :, 1]).sum(axis=1)
                    losses[policy, label] = loss
                    vb, vm, reach = visit_statistics(metrics, checkpoints)
                    details.append((label, counts.mean(axis=0), vb, vm, reach))
                    excess_str = '--' if uniform_a else f'{excess[-1]:.4f}'
                    print(f'{label:<15} {100*greedy[-tail:].mean():11.2f} {counts.sum(axis=1).mean():9.1f} '
                          f'{counts.min(axis=1).mean():8.2f} {counts.max(axis=1).mean():8.2f} '
                          f'{100*jnp.all(counts>0,axis=1).mean():10.2f} {100*jnp.all(split_counts>0,axis=(1,2)).mean():10.2f} '
                          f'{bias[-tail:].mean():10.5f} {mse[-tail:].mean():10.5f} {loss.mean():12.4f} {excess_str:>10}', flush=True)
                    axes[0,col].plot(t,100*rolling_mean(greedy,window),label=label)
                    axes[1,col].plot(t,rolling_mean(bias,window),label=label)
                    axes[2,col].plot(checkpoints,jnp.where(reach>=.9,vb,jnp.nan),marker='.',label=label)
                    axes[3,col].plot(checkpoints,jnp.where(reach>=.9,vm,jnp.nan),marker='.',label=label)
                    del metrics, counts, split_counts
                print('Mean per-arm counts (0 through 9):')
                for label, counts, _, _, _ in details:
                    print(f'{label:<15} '+' '.join(f'{v:7.1f}' for v in counts))
                print('Fixed B visits: conditional bias, MSE and reach fraction')
                for label, _, vb, vm, reach in details:
                    print(label)
                    for n,b,m,r in zip(checkpoints,vb,vm,reach):
                        print(f'  {int(n):5d} {b:10.5f} {m:10.5f} {100*r:6.1f}%')
                axes[0,col].set_title(policy)
                axes[0,col].set_ylim(-2,102)
                for row in (1,2):
                    axes[row,col].axhline(0,color='gray',linestyle='--',linewidth=1)
                for row in range(4):
                    axes[row,col].set_xscale('log')
                    axes[row,col].grid(alpha=.25)
                    axes[row,col].set_xlabel('Episode' if row<2 else 'B visits')
            print('\nPaired greedy-loss differences (first minus second; positive favors second):')
            for a,b in (('Original','Uniform B'),('Uniform A','Uniform both'),('Original','Uniform A'),('Uniform B','Uniform both')):
                print(f'{a} -> {b}')
                for label,_,_ in METHODS:
                    d=losses[a,label]-losses[b,label]
                    m=d.mean(); h=1.96*d.std(ddof=1)/jnp.sqrt(args.runs)
                    print(f'{label:<15} {m:10.4f} [{m-h:10.4f}, {m+h:10.4f}]')
            for ax,label in zip(axes[:,0],('Greedy optimal (%)','Episode B bias','Visit-matched B bias','Visit-matched B MSE')):
                ax.set_ylabel(label)
            axes[0,0].legend(fontsize=8,ncol=2)
            fig.tight_layout()
            if args.save_prefix:
                fig.savefig(f'{args.save_prefix}_mu{branch:+g}_sigma{sigma:g}.png',dpi=150)
            if args.no_show:
                plt.close(fig)
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
