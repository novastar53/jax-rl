"""Sweep independent UCB exploration coefficients at A and B; compare TD backups."""
import argparse
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import csv
from pathlib import Path


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
            num_gaussian_samples, mode, b_policy="epsilon", ucb_c=1.,
            a_policy="epsilon", a_ucb_c=1.):
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

        if a_policy == "ucb":
            a_counts = state.a_double_n.sum(axis=0) if mode == "double" else state.a_n
            action_a = ucb_action(a_key, a_values, a_counts, a_ucb_c)
        else:
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
    a_counts = final_state.a_double_n.sum(axis=0) if mode == "double" else final_state.a_n
    return metrics, final_state.b_n, final_state.b_split_n, a_counts


@partial(jax.jit, static_argnames=("num_runs", "num_arms", "num_episodes", "num_gaussian_samples", "mode", "b_policy", "a_policy"))
def run_batch(key, branch_value, reward_std, coefficient, epsilon=0.1,
              alpha_cap=1., decay_exponent=0.8, arm_gap=0., initial_se=1.,
              num_runs=200, num_arms=10, num_episodes=5000,
              num_gaussian_samples=100, mode="interpolated", b_policy="epsilon", ucb_c=1.,
              a_policy="epsilon", a_ucb_c=1.):
    keys = jax.random.split(key, num_runs)
    return jax.vmap(lambda k: run_one(k, branch_value, reward_std, coefficient,
                                    epsilon, alpha_cap, decay_exponent, arm_gap,
                                    initial_se, num_arms, num_episodes,
                                    num_gaussian_samples, mode, b_policy, ucb_c,
                                    a_policy, a_ucb_c))(keys)


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
    parser.add_argument('--a-cs', type=float, nargs='+', default=[.1, .5, 1., 2.])
    parser.add_argument('--b-cs', type=float, nargs='+', default=[.1, .5, 1., 2.])
    parser.add_argument('--gaussian-samples', type=int, default=100)
    parser.add_argument('--alpha-cap', type=float, default=1.)
    parser.add_argument('--decay-exponent', type=float, default=.8)
    parser.add_argument('--methods', nargs='+', choices=[m[0] for m in METHODS], default=None)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--save-prefix', default=None)
    parser.add_argument('--csv', default='dual_ucb_results.csv')
    args = parser.parse_args()
    if (args.runs < 2 or args.episodes < 2 or args.gaussian_samples < 1
            or any(s <= 0 for s in args.reward_stds)
            or any(b == 0 for b in args.branch_values)
            or any(g < 0 for g in args.arm_gaps)
            or any(c <= 0 for c in args.a_cs + args.b_cs)
            or len(set(args.a_cs)) != len(args.a_cs) or len(set(args.b_cs)) != len(args.b_cs)
            or not 0 < args.alpha_cap <= 1 or not .5 < args.decay_exponent <= 1):
        parser.error('Invalid experimental parameters')
    epsilon = .1
    key = jax.random.key(args.seed)
    tail = min(300, args.episodes)
    a_policies = [('eps', 'epsilon', 1.)] + [(f'UCB {c:g}', 'ucb', c) for c in args.a_cs]
    b_policies = [('eps', 'epsilon', 1.), ('uniform', 'uniform', 1.)] + [(f'UCB {c:g}', 'ucb', c) for c in args.b_cs]
    methods = tuple(m for m in METHODS if args.methods is None or m[0] in args.methods)
    environments = [(gap, branch, sigma) for gap in args.arm_gaps for branch in args.branch_values for sigma in args.reward_stds]
    checkpoints = jnp.array([n for n in (1, 10, 50, 100, 200, 500, 1000, 2000, 4000) if n < args.episodes])
    print(f'Runs={args.runs}, episodes={args.episodes}, seed={args.seed}, epsilon={epsilon}, UCB c in reward units')
    print(f'{len(environments)*len(methods)*len(a_policies)*len(b_policies)} configurations; A and B bonuses affect behavior only.')
    print('Raw cumulative pseudo-regret (branch + within-B). No epsilon baseline subtracted.')
    print(f'Tail metrics average last {tail} episodes; greedy loss evaluates both greedy decisions.')
    print('Visit-matched diagnostics are conditional on reaching the checkpoint.')
    rows = []
    pooled = {}
    for gap, branch, sigma in environments:
        print(f'\nBest B mean={branch:+g}, sigma={sigma:g}, arm gap={gap:g}', flush=True)
        totals = {}
        for a_label, a_policy, ca in a_policies:
            for b_label, b_policy, cb in b_policies:
                print(f'\nA={a_label}, B={b_label}', flush=True)
                print(f"{'Method':<15} {'Greedy A%':>9} {'B visits':>9} {'Min N':>8} {'Max N':>8} {'Split cov%':>10} {'B bias':>10} {'B MSE':>10} {'Branch':>10} {'B regret':>10} {'Total':>10} {'Greedy loss':>12} {'A gain':>10} {'95% CI (A gain)':>23}")
                for label, mode, coefficient in methods:
                    metrics, counts, split_counts, a_counts = run_batch(
                        key, branch, sigma, coefficient, epsilon=epsilon,
                        alpha_cap=args.alpha_cap, decay_exponent=args.decay_exponent,
                        arm_gap=gap, num_runs=args.runs, num_episodes=args.episodes,
                        num_gaussian_samples=args.gaussian_samples, mode=mode,
                        a_policy=a_policy, a_ucb_c=ca, b_policy=b_policy, ucb_c=cb)
                    left, greedy, _, regret, branch_regret, error, within, greedy_loss = jnp.moveaxis(metrics, -1, 0)
                    total, br, wr, gl = (x.sum(axis=1) for x in (regret, branch_regret, within, greedy_loss))
                    k = (a_label, b_label, label)
                    totals[k] = (total, br, wr)
                    pooled.setdefault(k, []).append(total)
                    gain, lo, hi = paired_ci(totals['eps', b_label, label][0], total)
                    vb, vm, reach = visit_statistics(metrics, checkpoints)
                    row = dict(gap=gap, branch_value=branch, sigma=sigma, a_policy=a_label,
                               b_policy=b_label, method=label, greedy_a_pct=float(100*greedy[:,-tail:].mean()),
                               b_visits=float(counts.sum(axis=1).mean()), min_b_n=float(counts.min(axis=1).mean()),
                               max_b_n=float(counts.max(axis=1).mean()),
                               split_coverage_pct=float(100*jnp.all(split_counts>0,axis=(1,2)).mean()),
                               b_bias=float(error[:,-tail:].mean()), b_mse=float((error[:,-tail:]**2).mean()),
                               branch_regret=float(br.mean()), within_b_regret=float(wr.mean()),
                               total_regret=float(total.mean()), greedy_loss=float(gl.mean()),
                               a_gain=gain, a_gain_lo=lo, a_gain_hi=hi,
                               a_left_count=float(a_counts[:,0].mean()), a_right_count=float(a_counts[:,1].mean()))
                    for n, bias, mse, r in zip(checkpoints, vb, vm, reach):
                        row.update({f'bias_{int(n)}':float(bias), f'mse_{int(n)}':float(mse), f'reach_{int(n)}':float(r)})
                    rows.append(row)
                    print(f'{label:<15} {row["greedy_a_pct"]:9.2f} {row["b_visits"]:9.1f} {row["min_b_n"]:8.2f} '
                          f'{row["max_b_n"]:8.2f} {row["split_coverage_pct"]:10.2f} {row["b_bias"]:10.5f} '
                          f'{row["b_mse"]:10.5f} {row["branch_regret"]:10.4f} {row["within_b_regret"]:10.4f} '
                          f'{row["total_regret"]:10.4f} {row["greedy_loss"]:12.4f} {gain:10.4f} [{lo:9.4f}, {hi:9.4f}]', flush=True)
                    del metrics, counts, split_counts, a_counts
        print('\nTop configurations in this environment (descriptive, not held-out tuning):')
        for i, (k, (total, br, wr)) in enumerate(sorted(totals.items(), key=lambda x: float(x[1][0].mean()))[:10], 1):
            print(f'{i:2d}. {k[2]:<15} A={k[0]:<8} B={k[1]:<8} total={total.mean():9.4f} branch={br.mean():9.4f} within B={wr.mean():9.4f}')
        # Use a common color scale across methods within an environment.
        vmax = max(float(x[0].mean()) for x in totals.values())
        fig, axes = plt.subplots(2, 4, figsize=(18, 8), squeeze=False, layout='constrained')
        fig.suptitle(f'Total regret: best B mean={branch:+g}, sigma={sigma:g}, gap={gap:g}')
        for ax, (label, _, _) in zip(axes.flat, methods):
            matrix = jnp.array([[totals[a,b,label][0].mean() for b,_,_ in b_policies] for a,_,_ in a_policies])
            im = ax.imshow(matrix, vmin=0., vmax=max(vmax,1e-6), aspect='auto', cmap='viridis_r')
            ax.set_xticks(range(len(b_policies)), [p[0] for p in b_policies], rotation=45, ha='right')
            ax.set_yticks(range(len(a_policies)), [p[0] for p in a_policies])
            ax.set_title(label)
            ax.set_xlabel('Policy at B'); ax.set_ylabel('Policy at A')
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    v=float(matrix[i,j]); ax.text(j,i,f'{v:.0f}',ha='center',va='center',fontsize=8,
                                                 color='white' if v > vmax*.55 else 'black')
        for ax in list(axes.flat)[len(methods):]:
            ax.set_visible(False)
        fig.colorbar(im, ax=list(axes.flat)[:len(methods)], label='Total pseudo-regret', shrink=.85)
        if args.save_prefix:
            fig.savefig(f'{args.save_prefix}_gap{gap:g}_mu{branch:+g}_sigma{sigma:g}.png', dpi=150)
        if args.no_show:
            plt.close(fig)
    # Reused seeds couple environments; average within each run before computing CIs.
    averages = {k:jnp.stack(v).mean(axis=0) for k,v in pooled.items()}
    baseline_key = ('eps', 'eps', 'Lambda=1')
    prior_key = ('eps', 'UCB 0.5', 'Lambda=1')
    print('\nOverall ranking (equally weighted environments; selected on these same runs):')
    print('Paired CIs are pointwise and not adjusted for hyperparameter search.')
    for i,(k,v) in enumerate(sorted(averages.items(),key=lambda x:float(x[1].mean())),1):
        out = f'{i:3d}. {k[2]:<15} A={k[0]:<8} B={k[1]:<8} mean total={v.mean():9.4f}'
        for base, name in ((baseline_key,'Q eps/eps'), (prior_key,'Q eps/UCB .5')):
            if base in averages:
                d,lo,hi=paired_ci(averages[base],v)
                out+=f' | {name} gain={d:9.4f} [{lo:9.4f}, {hi:9.4f}]'
        print(out)
    with Path(args.csv).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f'All diagnostics saved to {args.csv}')
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
