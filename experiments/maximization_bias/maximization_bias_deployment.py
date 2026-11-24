"""Train with Maximum + UCB; evaluate frozen Maximum and Gaussian policies."""
import argparse
import csv
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


class State(NamedTuple):
    q_max: jax.Array
    q_gaussian: jax.Array
    a_n: jax.Array
    b_mean: jax.Array
    b_n: jax.Array
    b_m2: jax.Array


def init_state(k):
    return State(jnp.zeros(2), jnp.zeros(2), jnp.zeros(2, jnp.int32),
                 jnp.zeros(k), jnp.zeros(k, jnp.int32), jnp.zeros(k))


def greedy_probabilities(q):
    mask = q == jnp.max(q)
    return mask / mask.sum()


def ucb_action(key, q, n, c):
    bonus = c * jnp.sqrt(jnp.log(jnp.maximum(n.sum(), 1)) / jnp.maximum(n, 1))
    scores = jnp.where(n == 0, jnp.inf, q + bonus)
    return jax.random.choice(key, q.size, p=greedy_probabilities(scores))


def gaussian_backup(key, mean, count, m2, num_draws, initial_se=1.):
    variance = jnp.maximum(m2, 0.) / jnp.maximum(count - 1, 1)
    se = jnp.where(count >= 2, jnp.sqrt(variance / jnp.maximum(count, 1)), initial_se)
    draws = mean + se * jax.random.normal(key, (num_draws, mean.size), dtype=mean.dtype)
    return mean[jnp.argmax(draws, axis=1)].mean()


def train_one(key, branch, sigma, gap, ca, cb, checkpoints, num_arms,
              num_episodes, num_draws, alpha_cap=1., decay_exponent=.8):
    means = jnp.full(num_arms, branch - gap).at[0].set(branch)
    best = jnp.maximum(branch, 0.)

    def episode(carry, _):
        s, key = carry
        # Preserve Step 52 key allocation (unused split keys are reserved).
        key, ak, bk, rk, _, _, gk = jax.random.split(key, 7)
        a = ucb_action(ak, s.q_max, s.a_n, ca)
        b = ucb_action(bk, s.b_mean, s.b_n, cb)
        left = a == 0
        r = means[b] + sigma * jax.random.normal(rk, ())
        targets = jnp.where(left, jnp.stack((s.b_mean.max(), gaussian_backup(
            gk, s.b_mean, s.b_n, s.b_m2, num_draws))), 0.)
        n = s.a_n.at[a].add(1)
        lr = jnp.minimum(alpha_cap, n[a].astype(jnp.float32) ** (-decay_exponent))
        qm = s.q_max.at[a].add(lr * (targets[0] - s.q_max[a]))
        qg = s.q_gaussian.at[a].add(lr * (targets[1] - s.q_gaussian[a]))
        s = s._replace(q_max=qm, q_gaussian=qg, a_n=n)

        def visit(s):
            bn = s.b_n.at[b].add(1)
            delta = r - s.b_mean[b]
            mean = s.b_mean.at[b].add(delta / bn[b])
            m2 = s.b_m2.at[b].add(delta * (r - mean[b]))
            return s._replace(b_n=bn, b_mean=mean, b_m2=m2)

        s = jax.lax.cond(left, visit, lambda s: s, s)
        branch_regret = best - jnp.where(left, branch, 0.)
        within_regret = jnp.where(left, branch - means[b], 0.)
        # Post-update policy, including the most recent training observation.
        p_left = jnp.stack((greedy_probabilities(s.q_max)[0],
                            greedy_probabilities(s.q_gaussian)[0]))
        p_best_b = greedy_probabilities(s.b_mean)[0]
        metrics = jnp.concatenate((jnp.stack((branch_regret, within_regret)),
                                   p_left, jnp.stack((p_best_b, s.b_n.sum()))))
        return (s, key), metrics

    (_, _), m = jax.lax.scan(episode, (init_state(num_arms), key), None, length=num_episodes)
    i = checkpoints - 1
    totals = jnp.cumsum(m[:, :2], axis=0)
    return totals[i], m[i, 2:4], m[i, 4], m[i, 5], totals[-1]


@partial(jax.jit, static_argnames=('num_runs', 'num_arms', 'num_episodes', 'num_draws'))
def train_batch(key, branch, sigma, gap, ca, cb, checkpoints, num_runs=200,
                num_arms=10, num_episodes=10000, num_draws=100):
    return jax.vmap(lambda k: train_one(k, branch, sigma, gap, ca, cb,
        checkpoints, num_arms, num_episodes, num_draws))(jax.random.split(key, num_runs))


@partial(jax.jit, static_argnames=('episodes',))
def evaluate_batch(key, p_left, p_best_b, branch, gap, sigma, episodes):
    """Fresh frozen-policy episodes, paired across learners and independent of training."""
    keys = jax.random.split(key, p_left.shape[0] * p_left.shape[1])
    keys = keys.reshape(p_left.shape[:2] + keys.shape[1:])
    best = jnp.maximum(branch, 0.)

    def evaluate(k, p, pb):
        ak, bk, rk = jax.random.split(k, 3)
        # Common random numbers for the two deployment policies.
        left = jax.random.uniform(ak, (episodes,))[:, None] < p[None, :]
        wrong = jax.random.uniform(bk, (episodes,)) >= pb
        mean = jnp.where(left, branch - gap * wrong[:, None], 0.)
        reward = mean + left * sigma * jax.random.normal(rk, (episodes, 1))
        return (best - mean).sum(axis=0), (best - reward).sum(axis=0)

    return jax.vmap(jax.vmap(evaluate))(keys, p_left, p_best_b)


def paired_ci(x, y):
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    m = d.mean()
    h = 1.96 * d.std(ddof=1) / np.sqrt(d.size)
    return m, m-h, m+h


def expected_loss(p_left, p_best_b, branch, gap):
    # B greedy actions are identical in the two learners. Random ties uniform.
    return np.maximum(branch, 0.) - p_left * (branch - gap * (1-p_best_b[..., None]))


def summarize(training, p_left, p_b, visits, continuous, sampled, realized,
              branch, gap, checkpoints, deployment, lifetime):
    training = np.asarray(training, dtype=np.float64)
    p_left, p_b, visits, continuous, sampled, realized = map(np.asarray,
        (p_left, p_b, visits, continuous, sampled, realized))
    loss = expected_loss(p_left, p_b, branch, gap)
    train = training.sum(axis=-1)
    return dict(training=train, branch=training[..., 0], within=training[..., 1],
                visits=visits, loss=loss, p_left=p_left, p_best_b=p_b,
                deploy=deployment*loss, sampled=sampled, realized=realized,
                total=train[..., None]+deployment*loss,
                lifetime=train[..., None]+(lifetime-checkpoints)[None,:,None]*loss,
                continuous=continuous.sum(axis=-1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=int, default=200)
    p.add_argument('--budgets', type=int, nargs='+', default=[100,500,1000,2000,5000])
    p.add_argument('--deployment', type=int, default=5000)
    p.add_argument('--lifetime', type=int, default=10000)
    p.add_argument('--seed', type=int, default=46)
    p.add_argument('--a-c', type=float, default=.1)
    p.add_argument('--b-cs', type=float, nargs='+', default=[.5,1.])
    p.add_argument('--reward-stds', type=float, nargs='+', default=[1.])
    p.add_argument('--branch-values', type=float, nargs='+', default=[-.1,.1])
    p.add_argument('--arm-gaps', type=float, nargs='+', default=[0.,.25])
    p.add_argument('--gaussian-samples', type=int, default=100)
    p.add_argument('--csv', default='deployment_results.csv')
    p.add_argument('--no-show', action='store_true')
    p.add_argument('--save-prefix', default=None)
    args = p.parse_args()
    budgets = np.array(sorted(set(args.budgets)), dtype=np.int32)
    if (args.runs < 2 or budgets.min() < 1 or budgets.max() > args.lifetime
        or args.deployment < 1 or args.gaussian_samples < 1 or args.a_c <= 0
        or any(c <= 0 for c in args.b_cs) or len(set(args.b_cs)) != len(args.b_cs)
        or any(s <= 0 for s in args.reward_stds) or any(g < 0 for g in args.arm_gaps)
        or any(v == 0 for v in args.branch_values)):
        p.error('Invalid parameters; lifetime must be at least the largest budget.')
    train_key = jax.random.key(args.seed)
    eval_key = jax.random.fold_in(train_key, 123456)
    rows, pooled = [], {cb: [] for cb in args.b_cs}
    print(f'Runs={args.runs}, seed={args.seed}, cA={args.a_c:g}, budgets={budgets.tolist()}')
    print(f'Frozen greedy deployment={args.deployment}; fixed lifetime={args.lifetime}; Maximum+UCB collects all data.')
    print('Primary deployment costs are exact conditional expected pseudo-regret; MC and realized costs use fresh evaluation episodes.')
    print('Identical training data and identical greedy B policy; only A deployment decisions differ.')
    for gap in args.arm_gaps:
        for branch in args.branch_values:
            for sigma in args.reward_stds:
                print(f'\nB mean={branch:+g}, gap={gap:g}, sigma={sigma:g}', flush=True)
                for cb in args.b_cs:
                    raw = train_batch(train_key, branch, sigma, gap, args.a_c, cb, jnp.asarray(budgets),
                                      num_runs=args.runs, num_episodes=args.lifetime, num_draws=args.gaussian_samples)
                    sampled, realized = evaluate_batch(eval_key, raw[1], raw[2], branch, gap, sigma, args.deployment)
                    d = summarize(*raw, sampled, realized, branch, gap, budgets, args.deployment, args.lifetime)
                    pooled[cb].append(d)
                    print(f'cB={cb:g}; continuing Maximum+UCB lifetime regret={d["continuous"].mean():.4f}')
                    print(' Budget   Train   B visits  Loss/ep M/G      Deploy M/G        Total M/G       Lifetime M/G     Deployment gain [95% CI]')
                    for i,t in enumerate(budgets):
                        gain,lo,hi = paired_ci(d['deploy'][:,i,0],d['deploy'][:,i,1])
                        mean = lambda k: d[k][:,i].mean(axis=0)
                        fmt = lambda k: '/'.join(f'{v:.4f}' for v in mean(k))
                        print(f'{t:7d} {mean("training"):8.3f} {mean("visits"):8.1f} {fmt("loss"):>17} {fmt("deploy"):>20} {fmt("total"):>20} {fmt("lifetime"):>20} {gain:9.4f} [{lo:.4f}, {hi:.4f}]', flush=True)
                        for j,m in enumerate(('Maximum','Gaussian')):
                            rows.append(dict(branch_value=branch,gap=gap,sigma=sigma,c_a=args.a_c,c_b=cb,
                                budget=int(t),deployment_episodes=args.deployment,lifetime_episodes=args.lifetime,method=m,
                                training_regret=mean('training'),training_branch=mean('branch'),training_within_b=mean('within'),
                                b_visits=mean('visits'),greedy_left_pct=100*mean('p_left')[j],greedy_arm0_pct=100*mean('p_best_b'),
                                greedy_loss_per_episode=mean('loss')[j],expected_deployment_regret=mean('deploy')[j],
                                sampled_deployment_pseudo_regret=mean('sampled')[j],realized_deployment_regret=mean('realized')[j],
                                expected_combined_regret=mean('total')[j],expected_lifetime_regret=mean('lifetime')[j],
                                continuing_lifetime_regret=d['continuous'].mean(),
                                gaussian_deployment_gain=gain,gain_ci_low=lo,gain_ci_high=hi))
    avg = {cb:{k:np.stack([d[k] for d in ds]).mean(axis=0) for k in ds[0]} for cb,ds in pooled.items()}
    print('\nOverall: equally weighted environments; paired pointwise 95% CIs, unadjusted for search.')
    print('Gain = Maximum minus Gaussian. Positive favors Gaussian. M/G columns show Maximum/Gaussian.')
    ranking = []
    for cb,d in avg.items():
        print(f'\ncB={cb:g}; continuing Maximum+UCB lifetime regret={d["continuous"].mean():.4f}')
        print(' Budget   Train   Deploy M/G          MC pseudo M/G       Realized M/G       Total M/G          Lifetime M/G       Gaussian deployment gain [95% CI]')
        for i,t in enumerate(budgets):
            fmt = lambda k: '/'.join(f'{v:.4f}' for v in d[k][:,i].mean(axis=0))
            gain,lo,hi = paired_ci(d['deploy'][:,i,0],d['deploy'][:,i,1])
            print(f'{t:7d} {d["training"][:,i].mean():8.3f} {fmt("deploy"):>21} {fmt("sampled"):>21} {fmt("realized"):>21} {fmt("total"):>21} {fmt("lifetime"):>21} {gain:9.4f} [{lo:.4f}, {hi:.4f}]')
            for j,m in enumerate(('Maximum','Gaussian')):
                ranking.append((d['lifetime'][:,i,j].mean(), cb, int(t), m, d['lifetime'][:,i,j]))
    # Cross-setting comparisons at equal total lifetime, against the preselected cB=.5 continuous baseline if available.
    base_cb = .5 if .5 in avg else args.b_cs[0]
    ref = avg[base_cb]['continuous']
    print(f'\nFixed-lifetime ranking (selected on these runs); continuous Q+UCB cB={base_cb:g} baseline={ref.mean():.4f}')
    for val,cb,t,m,per_run in sorted(ranking,key=lambda r:r[0]):
        gain,lo,hi=paired_ci(ref,per_run)
        print(f'{m:<9} cB={cb:g} budget={t:5d} lifetime={val:10.4f} gain={gain:9.4f} [{lo:.4f}, {hi:.4f}]')
    with open(args.csv,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'Saved {len(rows)} rows to {args.csv}')
    fig, axes = plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for cb,d in avg.items():
        for j,m in enumerate(('Maximum','Gaussian')):
            for ax,k in zip(axes,('deploy','total','lifetime')):
                ax.plot(budgets,d[k].mean(axis=0)[:,j],marker='o',label=f'{m}, cB={cb:g}')
    axes[2].axhline(ref.mean(),ls='--',color='black',label=f'Continue Q, cB={base_cb:g}')
    for ax,title in zip(axes,('Frozen deployment','Training + deployment','Fixed lifetime')):
        ax.set(xscale='log',xlabel='Training episodes',ylabel='Expected pseudo-regret',title=title)
        ax.legend(fontsize=8)
    if args.save_prefix: fig.savefig(f'{args.save_prefix}_summary.png',dpi=150)
    if args.no_show: plt.close(fig)
    else: plt.show()


if __name__ == '__main__':
    main()
