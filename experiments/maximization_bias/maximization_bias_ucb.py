"""Compare epsilon-greedy, uniform and UCB exploration at B in the maximization-bias MDP."""
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
    a_double_q: jax.Array   # two A tables for conventional Double Q
    a_double_n: jax.Array
    b_mean: jax.Array       # all observations of each B action
    b_n: jax.Array
    b_m2: jax.Array
    b_split_mean: jax.Array # random partitions of B rewards
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


def ucb_action(key, q, counts, c):
    # c is in reward units, not scaled using the true reward variance.
    total = counts.sum()
    bonus = c * jnp.sqrt(jnp.log(jnp.maximum(total, 1)) / jnp.maximum(counts, 1))
    scores = jnp.where(counts == 0, jnp.inf, q + bonus)
    return epsilon_greedy(key, scores, 0.0)


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
            num_gaussian_samples, mode, b_policy="epsilon", ucb_c=1.):
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

        if b_policy == "ucb":
            action_b = ucb_action(b_key, b_values, state.b_n, ucb_c)
        else:
            action_b = epsilon_greedy(b_key, b_values, 1.0 if b_policy == "uniform" else epsilon)
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
        within_regret = jnp.where(left, branch_value - means[action_b], 0.)
        greedy_b = b_values == jnp.max(b_values)
        greedy_b_value = jnp.sum(greedy_b * means) / greedy_b.sum()
        greedy_total_loss = best_value - greedy_left * greedy_b_value
        metrics = jnp.stack([left.astype(jnp.float32), optimal_greedy, reward,
                             regret, branch_regret, backup - branch_value,
                             within_regret, greedy_total_loss])
        return (state, key), metrics

    (final_state, _), metrics = jax.lax.scan(episode, (init_state(num_arms), key), None,
                                    length=num_episodes)
    return metrics, final_state.b_n, final_state.b_split_n


@partial(jax.jit, static_argnames=("num_runs", "num_arms", "num_episodes", "num_gaussian_samples", "mode", "b_policy"))
def run_batch(key, branch_value, reward_std, coefficient, epsilon=0.1,
              alpha_cap=1., decay_exponent=0.8, arm_gap=0., initial_se=1.,
              num_runs=200, num_arms=10, num_episodes=5000,
              num_gaussian_samples=100, mode="interpolated", b_policy="epsilon", ucb_c=1.):
    keys = jax.random.split(key, num_runs)
    return jax.vmap(lambda k: run_one(k, branch_value, reward_std, coefficient,
                                    epsilon, alpha_cap, decay_exponent, arm_gap,
                                    initial_se, num_arms, num_episodes,
                                    num_gaussian_samples, mode, b_policy, ucb_c))(keys)


def rolling_mean(x, window):
    c = jnp.concatenate([jnp.zeros(1, dtype=x.dtype), jnp.cumsum(x)])
    t = jnp.arange(1, x.size + 1)
    start = jnp.maximum(t - window, 0)
    return (c[t] - c[start]) / (t - start)


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


def paired_ci(a, b):
    d = a - b
    m = d.mean()
    h = 1.96 * d.std(ddof=1) / jnp.sqrt(d.size)
    return float(m), float(m-h), float(m+h)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=200)
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--reward-stds', type=float, nargs='+', default=[1.])
    parser.add_argument('--branch-values', type=float, nargs='+', default=[-.1, .1])
    parser.add_argument('--arm-gaps', type=float, nargs='+', default=[0., .25])
    parser.add_argument('--ucb-cs', type=float, nargs='+', default=[.5, 1., 2.])
    parser.add_argument('--gaussian-samples', type=int, default=100)
    parser.add_argument('--alpha-cap', type=float, default=1.)
    parser.add_argument('--decay-exponent', type=float, default=.8)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--save-prefix', default=None)
    args = parser.parse_args()
    if (args.runs < 2 or args.episodes < 2 or args.gaussian_samples < 1
            or any(s <= 0 for s in args.reward_stds)
            or any(b == 0 for b in args.branch_values)
            or any(g < 0 for g in args.arm_gaps)
            or any(c <= 0 for c in args.ucb_cs)
            or not 0 < args.alpha_cap <= 1 or not .5 < args.decay_exponent <= 1):
        parser.error('Invalid experimental parameters')
    epsilon = .1
    key = jax.random.key(args.seed)
    t = jnp.arange(1, args.episodes + 1)
    tail = min(300, args.episodes)
    checkpoints = jnp.array([n for n in (1, 5, 10, 20, 50, 100, 200, 300, 500, 1000, 2000, 4000) if n < args.episodes])
    policies = [('Epsilon-greedy', 'epsilon', 1.), ('Uniform', 'uniform', 1.)] + [
        (f'UCB c={c:g}', 'ucb', c) for c in args.ucb_cs]
    print(f'Runs={args.runs}, episodes={args.episodes}, seed={args.seed}, epsilon={epsilon}, UCB c in reward units')
    print('A always epsilon-greedy. Total regret = branch regret + within-B regret.')
    print('Branch excess subtracts expected optimal epsilon-greedy branch cost only.')
    print('Tail metrics average the last 300 episodes (or fewer for short runs).')
    print('Visit-matched estimates are conditional on reach; plots require >=90% reach.')
    for gap in args.arm_gaps:
        for branch in args.branch_values:
            for sigma in args.reward_stds:
                print(f'\nBest B mean={branch:+g}, sigma={sigma:g}, arm gap={gap:g}', flush=True)
                fig, axes = plt.subplots(4, len(policies), figsize=(3.9*len(policies), 12), squeeze=False)
                fig.suptitle(f'Best B mean={branch:+g}, sigma={sigma:g}, arm gap={gap:g}')
                totals = {}
                for col, (policy, b_policy, c) in enumerate(policies):
                    print(f'\n{policy}', flush=True)
                    print(f"{'Method':<15} {'Greedy A%':>9} {'B visits':>9} {'Min N':>8} {'Max N':>8} {'Split cov%':>10} {'B bias':>10} {'B MSE':>10} {'Branch':>10} {'B regret':>10} {'Total':>10} {'Branch excess':>14} {'Greedy loss':>12}")
                    details = []
                    for label, mode, coefficient in METHODS:
                        metrics, counts, split_counts = run_batch(
                            key, branch, sigma, coefficient, epsilon=epsilon,
                            alpha_cap=args.alpha_cap, decay_exponent=args.decay_exponent,
                            arm_gap=gap, num_runs=args.runs, num_episodes=args.episodes,
                            num_gaussian_samples=args.gaussian_samples, mode=mode,
                            b_policy=b_policy, ucb_c=c)
                        left, greedy, _, regret, branch_regret, error, within, greedy_loss = jnp.moveaxis(metrics, -1, 0)
                        total = regret.sum(axis=1)
                        br = branch_regret.sum(axis=1)
                        wr = within.sum(axis=1)
                        gl = greedy_loss.sum(axis=1)
                        totals[policy, label] = (total, br, wr, gl)
                        vb, vm, reach = visit_statistics(metrics, checkpoints)
                        details.append((label, counts.mean(axis=0), vb, vm, reach))
                        baseline = abs(branch)*epsilon/2*args.episodes
                        print(f'{label:<15} {100*greedy[:,-tail:].mean():9.2f} {counts.sum(axis=1).mean():9.1f} '
                              f'{counts.min(axis=1).mean():8.2f} {counts.max(axis=1).mean():8.2f} '
                              f'{100*jnp.all(split_counts>0,axis=(1,2)).mean():10.2f} '
                              f'{error[:,-tail:].mean():10.5f} {(error[:,-tail:]**2).mean():10.5f} '
                              f'{br.mean():10.4f} {wr.mean():10.4f} {total.mean():10.4f} '
                              f'{br.mean()-baseline:14.4f} {gl.mean():12.4f}', flush=True)
                        axes[0,col].plot(t,jnp.cumsum(branch_regret.mean(axis=0)-abs(branch)*epsilon/2),label=label)
                        axes[1,col].plot(t,jnp.cumsum(within.mean(axis=0)),label=label)
                        axes[2,col].plot(t,jnp.cumsum(regret.mean(axis=0)),label=label)
                        axes[3,col].plot(checkpoints,jnp.where(reach>=.9,vb,jnp.nan),marker='.',label=label)
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
                    axes[3,col].axhline(0,color='gray',linestyle='--',linewidth=1)
                    for row in range(4):
                        axes[row,col].set_xscale('log')
                        axes[row,col].grid(alpha=.25)
                        axes[row,col].set_xlabel('Episode' if row<3 else 'B visits')
                print('\nPaired regret differences (baseline minus alternative; pointwise 95% CI):')
                for policy, _, _ in policies[1:]:
                    for label,_,_ in METHODS:
                        for baseline_policy in ('Epsilon-greedy', 'Uniform') if policy != 'Uniform' else ('Epsilon-greedy',):
                            a=totals[baseline_policy,label]; b=totals[policy,label]
                            m,lo,hi=paired_ci(a[0],b[0])
                            branch_diff=(a[1]-b[1]).mean(); within_diff=(a[2]-b[2]).mean()
                            print(f'{baseline_policy:>14} -> {policy:<13} {label:<15} total {m:9.4f} [{lo:9.4f}, {hi:9.4f}] branch {branch_diff:9.4f} B {within_diff:9.4f}')
                for row,label in enumerate(('Cumulative branch excess','Cumulative within-B regret','Cumulative total regret','Visit-matched B bias')):
                    axes[row,0].set_ylabel(label)
                    # Shared scales within each row for honest visual comparisons.
                    limits=[ax.get_ylim() for ax in axes[row]]
                    for ax in axes[row]:
                        ax.set_ylim(min(a[0] for a in limits),max(a[1] for a in limits))
                axes[0,0].legend(fontsize=8,ncol=2)
                fig.tight_layout()
                if args.save_prefix:
                    fig.savefig(f'{args.save_prefix}_gap{gap:g}_mu{branch:+g}_sigma{sigma:g}.png',dpi=150)
                if args.no_show:
                    plt.close(fig)
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
