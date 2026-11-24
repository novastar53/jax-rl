"""Separate optimistic exploration TD backups from value learning in the maximization-bias MDP."""
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
    a_optimistic_q: jax.Array


def init_state(num_arms):
    return State(jnp.zeros(2), jnp.zeros(2, dtype=jnp.int32),
                 jnp.zeros((2, 2)), jnp.zeros((2, 2), dtype=jnp.int32),
                 jnp.zeros(num_arms), jnp.zeros(num_arms, dtype=jnp.int32),
                 jnp.zeros(num_arms), jnp.zeros((2, num_arms)),
                 jnp.zeros((2, num_arms), dtype=jnp.int32), jnp.zeros(2))


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


def optimistic_backup(mean, count, m2, beta, prior_scale):
    # Two prior degrees of freedom stabilize the variance at very small n.
    # Finite prior SE for unseen arms, unlike the action-selection bonus.
    variance = (jnp.maximum(m2, 0.) + 2 * prior_scale**2) / (jnp.maximum(count - 1, 0) + 2)
    se = jnp.sqrt(variance / jnp.maximum(count, 1))
    bonus = beta * se * jnp.sqrt(jnp.log(1. + jnp.maximum(count.sum(), 1)))
    return jnp.max(mean + bonus)


def alpha(n, cap, exponent):
    return jnp.minimum(cap, n.astype(jnp.float32) ** (-exponent))


def run_one(key, branch_value, reward_std, coefficient, epsilon, alpha_cap,
            decay_exponent, arm_gap, initial_se, num_arms, num_episodes,
            num_gaussian_samples, mode, b_policy="epsilon", ucb_c=1.,
            a_policy="epsilon", a_ucb_c=1., beta=0.):
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

        optimistic = optimistic_backup(state.b_mean, state.b_n, state.b_m2, beta, initial_se)
        if a_policy == "propagated":
            # All primary backup operators share the same behavior in this arm.
            a_counts = state.a_double_n.sum(axis=0) if mode == "double" else state.a_n
            action_a = ucb_action(a_key, state.a_optimistic_q, a_counts, a_ucb_c)
            b_values_behavior = state.b_mean
            n = a_counts[action_a] + 1
            target_plus = jnp.where(action_a == 0, optimistic, 0.)
            old_plus = state.a_optimistic_q[action_a]
            state = state._replace(a_optimistic_q=state.a_optimistic_q.at[action_a].add(
                alpha(n, alpha_cap, decay_exponent) * (target_plus - old_plus)))
        elif a_policy == "ucb":
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

        if a_policy != "propagated":
            b_values_behavior = b_values
        if b_policy == "ucb":
            action_b = ucb_action(b_key, b_values_behavior, state.b_n, ucb_c)
        else:
            action_b = epsilon_greedy(b_key, b_values_behavior, 1.0 if b_policy == "uniform" else epsilon)
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
                             within_regret, greedy_total_loss, optimistic - branch_value])
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
              a_policy="epsilon", a_ucb_c=1., beta=0.):
    keys = jax.random.split(key, num_runs)
    return jax.vmap(lambda k: run_one(k, branch_value, reward_std, coefficient,
                                    epsilon, alpha_cap, decay_exponent, arm_gap,
                                    initial_se, num_arms, num_episodes,
                                    num_gaussian_samples, mode, b_policy, ucb_c,
                                    a_policy, a_ucb_c, beta))(keys)


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


# Tuned on Step 51, frozen here before the independent-seed evaluation.
TUNED = {"Lambda=0": (2., 1.), "Lambda=0.25": (1., 2.),
         "Lambda=0.5": (.5, 2.), "Lambda=0.75": (.1, 1.),
         "Lambda=1": (.1, .5), "Double Q": (2., 2.), "Gaussian": (.5, 1.)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=int, default=200)
    p.add_argument('--episodes', type=int, default=5000)
    p.add_argument('--seed', type=int, default=45)
    p.add_argument('--reward-stds', type=float, nargs='+', default=[1.])
    p.add_argument('--branch-values', type=float, nargs='+', default=[-.1, .1])
    p.add_argument('--arm-gaps', type=float, nargs='+', default=[0., .25])
    p.add_argument('--betas', type=float, nargs='+', default=[0., .25, .5, 1., 2.])
    p.add_argument('--b-cs', type=float, nargs='+', default=[.5, 1.])
    p.add_argument('--a-c', type=float, default=.1)
    p.add_argument('--prior-scale', type=float, default=1.)
    p.add_argument('--gaussian-samples', type=int, default=100)
    p.add_argument('--csv', default='propagated_uncertainty_results.csv')
    p.add_argument('--no-show', action='store_true')
    p.add_argument('--save-prefix', default=None)
    args = p.parse_args()
    if (args.runs < 2 or args.episodes < 2 or args.gaussian_samples < 1
        or args.a_c <= 0 or args.prior_scale <= 0 or any(c <= 0 for c in args.b_cs)
        or any(b < 0 for b in args.betas) or any(s <= 0 for s in args.reward_stds)
        or any(v == 0 for v in args.branch_values) or any(g < 0 for g in args.arm_gaps)
        or len(set(args.betas)) != len(args.betas) or len(set(args.b_cs)) != len(args.b_cs)):
        p.error('Invalid experimental parameters')
    key = jax.random.key(args.seed)
    tail = min(300, args.episodes)
    envs = [(g,v,s) for g in args.arm_gaps for v in args.branch_values for s in args.reward_stds]
    # A config is (policy at A, local c_A, c_B, beta). Freeze the old winner.
    configs = [(m, 'tuned local', 'ucb', *TUNED[m], 0.) for m,_,_ in METHODS]
    configs += [(m, 'matched local', 'ucb', args.a_c, cb, 0.) for cb in args.b_cs for m,_,_ in METHODS
                if (args.a_c,cb) != TUNED[m]]
    configs += [(m, 'propagated', 'propagated', args.a_c, cb, beta)
                for cb in args.b_cs for beta in args.betas for m,_,_ in METHODS]
    method_map = {m:(mode,coef) for m,mode,coef in METHODS}
    rows, pooled = [], {}
    print(f'Runs={args.runs}, episodes={args.episodes}, seed={args.seed}; {len(configs)*len(envs)} configurations')
    print('Propagated policies use a separate optimistic A table and pooled B estimates for behavior.')
    print('Beta=0 is max-backup exploration; primary learning targets never include bonuses.')
    print('All methods with identical propagated settings share trajectories. Raw pseudo-regret is used.')
    for gap, branch, sigma in envs:
        print(f'\nBest B mean={branch:+g}, sigma={sigma:g}, gap={gap:g}', flush=True)
        reference = None
        behavior = {}
        print(f'{"Method":<13} {"Policy":<14} {"cA":>4} {"cB":>4} {"beta":>5} {"Branch":>10} {"Within B":>10} {"Total":>10} {"Greedy%":>8} {"B bias":>9} {"B MSE":>9} {"Greedy loss":>12} {"Greedy tail":>12}', flush=True)
        for m,label,policy,ca,cb,beta in configs:
            mode,coef = method_map[m]
            metrics, counts, splits, a_counts = run_batch(key,branch,sigma,coef,
                arm_gap=gap,initial_se=args.prior_scale,num_runs=args.runs,
                num_episodes=args.episodes,num_gaussian_samples=args.gaussian_samples,
                mode=mode,a_policy=policy,a_ucb_c=ca,b_policy='ucb',ucb_c=cb,beta=beta)
            left,greedy,_,regret,br,error,wr,gl,optimistic_error = jnp.moveaxis(metrics,-1,0)
            total, branch_total, within_total, greedy_total = (x.sum(axis=1) for x in (regret,br,wr,gl))
            if policy == 'propagated':
                k = (ca,cb,beta)
                if k not in behavior:
                    behavior[k] = (left, regret)
                else:
                    assert bool(jnp.array_equal(left,behavior[k][0])) and bool(jnp.array_equal(regret,behavior[k][1])), 'Propagated behavior mismatch'
            if m == 'Lambda=1' and label == 'tuned local':
                reference = total
            row = dict(gap=gap,branch_value=branch,sigma=sigma,method=m,policy=label,c_a=ca,c_b=cb,beta=beta,
                total_regret=float(total.mean()),branch_regret=float(branch_total.mean()),within_b_regret=float(within_total.mean()),
                greedy_a_pct=float(100*greedy[:,-tail:].mean()),b_bias=float(error[:,-tail:].mean()),b_mse=float((error[:,-tail:]**2).mean()),
                greedy_loss=float(greedy_total.mean()),greedy_tail_loss=float(gl[:,-tail:].mean()),
                b_visits=float(counts.sum(axis=1).mean()),min_b_n=float(counts.min(axis=1).mean()),
                max_b_n=float(counts.max(axis=1).mean()),split_coverage_pct=float(100*jnp.all(splits>0,axis=(1,2)).mean()),
                optimistic_b_bias=float(optimistic_error[:,-tail:].mean()) if policy=='propagated' else float('nan'))
            rows.append(row)
            pooled.setdefault((m,label,ca,cb,beta),[]).append((total,greedy_total,gl[:,-tail:].mean(axis=1)))
            print(f'{m:<13} {label:<14} {ca:4g} {cb:4g} {beta:5g} {row["branch_regret"]:10.4f} {row["within_b_regret"]:10.4f} {row["total_regret"]:10.4f} {row["greedy_a_pct"]:8.2f} {row["b_bias"]:9.5f} {row["b_mse"]:9.5f} {row["greedy_loss"]:12.4f} {row["greedy_tail_loss"]:12.5f}',flush=True)
            del metrics, counts, splits, a_counts
        del behavior
    averages = {k:tuple(jnp.stack([v[i] for v in vs]).mean(axis=0) for i in range(3)) for k,vs in pooled.items()}
    ref = averages['Lambda=1','tuned local',.1,.5,0.]
    print('\nOverall: mean over environments per run; pointwise 95% CIs, not adjusted for search.')
    print('Tuned local baselines use Step 51 coefficients, with the new seed.')
    for k,v in sorted(averages.items(),key=lambda kv:float(kv[1][0].mean())):
        m,label,ca,cb,beta=k
        gain,lo,hi=paired_ci(ref[0],v[0])
        print(f'{m:<13} {label:<14} cA={ca:g} cB={cb:g} beta={beta:g} total={v[0].mean():9.4f} greedy loss={v[1].mean():9.4f} tail={v[2].mean():.5f} Q baseline gain={gain:9.4f} [{lo:.4f}, {hi:.4f}]')
    print('\nMatched behavior: primary greedy loss gain versus Maximum for each propagated setting.')
    for k,v in averages.items():
        m,label,ca,cb,beta=k
        if label!='propagated' or m=='Lambda=1': continue
        d,lo,hi=paired_ci(averages['Lambda=1',label,ca,cb,beta][1],v[1])
        print(f'{m:<13} cB={cb:g} beta={beta:g}: {d:9.4f} [{lo:.4f}, {hi:.4f}]')
    with Path(args.csv).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'Saved {len(rows)} rows to {args.csv}')
    fig, axes = plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for cb in args.b_cs:
        axes[0].plot(args.betas,[float(averages['Lambda=1','propagated',args.a_c,cb,b][0].mean()) for b in args.betas],marker='o',label=f'cB={cb:g}')
    axes[0].axhline(float(ref[0].mean()),ls='--',color='black',label='Tuned Q baseline')
    axes[0].set(xlabel='Beta',ylabel='Total pseudo-regret',title='Shared behavior, mean over environments'); axes[0].legend()
    cb=args.b_cs[0]
    for m,_,_ in METHODS:
        axes[1].plot(args.betas,[float(averages[m,'propagated',args.a_c,cb,b][1].mean()) for b in args.betas],marker='o',label=m)
    axes[1].set(xlabel='Beta',ylabel='Cumulative greedy-policy loss',title=f'Primary learned policy, cB={cb:g}'); axes[1].legend(fontsize=8)
    if args.save_prefix: fig.savefig(f'{args.save_prefix}_summary.png',dpi=150)
    if args.no_show: plt.close(fig)
    else: plt.show()


if __name__ == '__main__':
    main()
