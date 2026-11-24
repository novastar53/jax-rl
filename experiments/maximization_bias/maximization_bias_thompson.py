"""Coherent posterior sampling versus local UCB in the maximization-bias MDP."""
import argparse
import csv
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr
import matplotlib.pyplot as plt
import numpy as np


METHODS = ('Q + UCB (0.5)', 'Q + UCB (1)', 'Lookahead UCB',
           'Thompson B only', 'Coherent Thompson', 'Thompson known noise')
CHECKPOINTS = (100, 500, 1000, 2000, 5000, 10000)


class State(NamedTuple):
    q_a: jax.Array
    n_a: jax.Array
    mean: jax.Array
    n: jax.Array
    m2: jax.Array


def init_state(k):
    return State(jnp.zeros(2), jnp.zeros(2, jnp.int32),
                 jnp.zeros(k), jnp.zeros(k, jnp.int32), jnp.zeros(k))


def greedy_probabilities(q):
    mask = q == q.max()
    return mask / mask.sum()


def ucb_action(key, q, n, c):
    bonus = c * jnp.sqrt(jnp.log(jnp.maximum(n.sum(), 1)) / jnp.maximum(n, 1))
    scores = jnp.where(n == 0, jnp.inf, q + bonus)
    return jax.random.choice(key, q.size, p=greedy_probabilities(scores))


def posterior(mean, n, m2, prior_scale=1., prior_count=1., known_sigma=None):
    """Normal approximation to independent normal-inverse-gamma posteriors.

    Prior mean zero, kappa=prior_count, alpha=2, beta=prior_scale**2.
    With known_sigma, use an exact conjugate normal posterior (oracle noise).
    """
    kappa = prior_count + n
    loc = n * mean / kappa
    if known_sigma is not None:
        return loc, known_sigma / jnp.sqrt(kappa)
    alpha = 2. + .5 * n
    beta = prior_scale**2 + .5 * jnp.maximum(m2, 0.) + .5 * prior_count * n * mean**2 / kappa
    se = jnp.sqrt(beta / ((alpha - 1.) * kappa))
    return loc, se


def branch_probability(loc, se):
    return -jnp.expm1(log_ndtr(-loc / se).sum())


def update_b(s, b, reward):
    n = s.n.at[b].add(1)
    delta = reward - s.mean[b]
    mean = s.mean.at[b].add(delta / n[b])
    m2 = s.m2.at[b].add(delta * (reward - mean[b]))
    return s._replace(n=n, mean=mean, m2=m2)


def run_one(key, branch, sigma, gap, mode, num_arms, num_episodes,
            ca=.1, cb=.5, prior_scale=1., prior_count=1.):
    means = jnp.full(num_arms, branch - gap).at[0].set(branch)
    best = jnp.maximum(branch, 0.)
    known = sigma if mode == 5 else None

    def episode(carry, _):
        s, key = carry
        # Same random-key layout as Step 53 for the UCB controls.
        key, ak, bk, rk, sk, _, _ = jax.random.split(key, 7)
        loc, se = posterior(s.mean, s.n, s.m2, prior_scale, prior_count, known)
        if mode >= 3:
            sample = loc + se * jax.random.normal(sk, (num_arms,))
            b = jnp.argmax(sample)
        else:
            b = ucb_action(bk, s.mean, s.n, 1. if mode == 1 else cb)
        if mode >= 4:
            a = jnp.where(sample[b] > 0., 0, 1)
        elif mode == 2:
            a = ucb_action(ak, jnp.stack((s.mean.max(), jnp.asarray(0.))), s.n_a, ca)
        else:
            a = ucb_action(ak, s.q_a, s.n_a, ca)
        left = a == 0
        r = means[b] + sigma * jax.random.normal(rk, ())
        n_a = s.n_a.at[a].add(1)
        alpha = n_a[a].astype(jnp.float32) ** (-.8)
        target = jnp.where(left, s.mean.max(), 0.)
        q_a = s.q_a.at[a].add(alpha * (target - s.q_a[a]))
        s = s._replace(n_a=n_a, q_a=q_a)
        s = jax.lax.cond(left, lambda x: update_b(x, b, r), lambda x: x, s)
        loc, se = posterior(s.mean, s.n, s.m2, prior_scale, prior_count, known)
        p = branch_probability(loc, se)
        # Common frozen decision diagnostics, using posterior means at B.
        p_best = greedy_probabilities(loc)[0]
        posterior_left = jnp.where(p > .5, 1., jnp.where(p < .5, 0., .5))
        frozen_loss = best - posterior_left * (branch - gap * (1. - p_best))
        mean_left = greedy_probabilities(jnp.stack((loc.max(), jnp.asarray(0.))))[0]
        mean_loss = best - mean_left * (branch - gap * (1. - p_best))
        branch_regret = best - jnp.where(left, branch, 0.)
        within = jnp.where(left, branch - means[b], 0.)
        correct = jnp.where(branch > 0, posterior_left, 1-posterior_left)
        metrics = jnp.stack((branch_regret, within, left, p, (p - (branch > 0))**2,
                             correct, frozen_loss, mean_loss, s.n.min(), s.n.max(),
                             s.n.sum(), loc.max() - branch, (loc.max() - branch)**2))
        return (s, key), metrics

    (s, _), m = jax.lax.scan(episode, (init_state(num_arms), key), None, length=num_episodes)
    return m, s.n


@partial(jax.jit, static_argnames=('mode','num_runs','num_arms','num_episodes'))
def run_batch(key, branch, sigma, gap, mode, num_runs=200, num_arms=10,
              num_episodes=10000, ca=.1, cb=.5, prior_scale=1., prior_count=1.):
    return jax.vmap(lambda k: run_one(k, branch, sigma, gap, mode, num_arms,
        num_episodes, ca, cb, prior_scale, prior_count))(jax.random.split(key, num_runs))


def paired_ci(x, y):
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    mid = d.mean()
    h = 1.96 * d.std(ddof=1) / np.sqrt(d.size)
    return mid, mid-h, mid+h


def summarize(raw, checkpoints):
    m, counts = map(np.asarray, raw)
    cumulative = np.cumsum(m[:,:,:2], axis=1, dtype=np.float64)
    total = cumulative.sum(axis=-1)
    tail = m[:, -min(300,m.shape[1]):].mean(axis=1)
    return dict(total=total[:, -1], branch=cumulative[:,-1,0], within=cumulative[:,-1,1],
                curve=total.mean(axis=0), p_curve=m[:,:,3].mean(axis=0),
                left_curve=m[:,:,2].mean(axis=0), tail=tail, counts=counts,
                checkpoints=m[:,checkpoints-1], checkpoint_regret=total[:,checkpoints-1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=int, default=200)
    p.add_argument('--episodes', type=int, default=10000)
    p.add_argument('--seed', type=int, default=47)
    p.add_argument('--reward-stds', type=float, nargs='+', default=[1.])
    p.add_argument('--branch-values', type=float, nargs='+', default=[-.1,.1])
    p.add_argument('--arm-gaps', type=float, nargs='+', default=[0.,.25])
    p.add_argument('--a-c', type=float, default=.1)
    p.add_argument('--b-c', type=float, default=.5)
    p.add_argument('--prior-scale', type=float, default=1.)
    p.add_argument('--prior-count', type=float, default=1.)
    p.add_argument('--csv', default='thompson_results.csv')
    p.add_argument('--save-prefix', default=None)
    p.add_argument('--no-show', action='store_true')
    args = p.parse_args()
    if (args.runs < 2 or args.episodes < 1 or min(args.a_c,args.b_c,args.prior_scale,args.prior_count)<=0
        or any(s<=0 for s in args.reward_stds) or any(g<0 for g in args.arm_gaps)
        or any(v==0 for v in args.branch_values)):
        p.error('Invalid experiment parameters')
    checkpoints = np.array(sorted(set([t for t in CHECKPOINTS if t <= args.episodes]+[args.episodes])))
    key = jax.random.key(args.seed)
    names = list(METHODS)
    names[0] = f'Q + UCB ({args.b_c:g})'
    names[2] = f'Lookahead UCB ({args.b_c:g})'
    envs = [(b,g,s) for g in args.arm_gaps for b in args.branch_values for s in args.reward_stds]
    fig, axes = plt.subplots(len(envs),3,figsize=(14,3.3*len(envs)),squeeze=False,layout='constrained')
    rows, pooled = [], [[] for _ in names]
    print(f'Runs={args.runs}, episodes={args.episodes}, seed={args.seed}, prior_scale={args.prior_scale:g}, prior_count={args.prior_count:g}')
    print('Right value known to lookahead/Thompson; Q+UCB uses TD. Known-noise Thompson is an oracle diagnostic.')
    print('Posterior probability is approximate except known-noise conjugate mode. Calibration is empirical on these fixed environments.')
    print('Regret = branch + within-B. Tail averages last 300 episodes; counts are final.')
    for i,(branch,gap,sigma) in enumerate(envs):
        print(f'\nB mean={branch:+g}, gap={gap:g}, sigma={sigma:g}',flush=True)
        print('Method                       Branch     Within B       Total     Left%     P(Left*)   Brier   Correct%   Min N    Max N   Frozen P loss   Frozen mean loss   Q gain [95% CI]')
        base = None
        for mode,name in enumerate(names):
            raw = run_batch(key,branch,sigma,gap,mode,num_runs=args.runs,num_episodes=args.episodes,
                            ca=args.a_c,cb=args.b_c,prior_scale=args.prior_scale,prior_count=args.prior_count)
            d = summarize(raw,checkpoints)
            del raw
            if mode == 0: base = d['total']
            gain,lo,hi = paired_ci(base,d['total'])
            t = d['tail'].mean(axis=0)
            print(f'{name:26s} {d["branch"].mean():10.4f} {d["within"].mean():10.4f} {d["total"].mean():10.4f} {100*t[2]:9.2f} {t[3]:11.5f} {t[4]:8.5f} {100*t[5]:9.2f} {d["counts"].min(axis=1).mean():8.2f} {d["counts"].max(axis=1).mean():8.2f} {t[6]:15.5f} {t[7]:17.5f} {gain:10.4f} [{lo:.4f}, {hi:.4f}]',flush=True)
            pooled[mode].append(d)
            for j,cp in enumerate(checkpoints):
                z=d['checkpoints'][:,j].mean(axis=0)
                rows.append(dict(branch_value=branch,gap=gap,sigma=sigma,method=name,episode=int(cp),
                    cumulative_regret=d['checkpoint_regret'][:,j].mean(),posterior_left_probability=z[3],
                    brier=z[4],posterior_branch_accuracy=100*z[5],posterior_decision_loss=z[6],
                    posterior_mean_greedy_loss=z[7],min_b_count=z[8],max_b_count=z[9],b_visits=z[10],
                    posterior_max_bias=z[11],posterior_max_mse=z[12],final_regret_gain=gain,final_gain_ci_low=lo,final_gain_ci_high=hi))
            x=np.arange(1,args.episodes+1)
            axes[i,0].plot(x,d['curve'],label=name)
            axes[i,1].plot(x,d['p_curve'],label=name)
            l=d['left_curve']; win=min(100,len(l)); cs=np.r_[0.,np.cumsum(l,dtype=np.float64)]
            starts=np.maximum(0,x-win)
            axes[i,2].plot(x,(cs[x]-cs[starts])/(x-starts),label=name)
        for ax,title in zip(axes[i],('Cumulative regret','Posterior P(Left optimal)','Rolling Left frequency')):
            ax.set(xscale='log',xlabel='Episode',title=title)
        axes[i,0].set_ylabel(f'B={branch:+g}, gap={gap:g}, sigma={sigma:g}')
        axes[i,1].axhline(float(branch>0),ls=':',color='black'); axes[i,1].set_ylim(-.03,1.03)
        axes[i,2].set_ylim(-.03,1.03)
    axes[0,0].legend(fontsize=7)
    print('\nOverall: equal-weight environments; paired pointwise 95% intervals.')
    avg = [np.stack([d['total'] for d in ds]).mean(axis=0) for ds in pooled]
    for mode in sorted(range(len(names)),key=lambda m:avg[m].mean()):
        gain,lo,hi=paired_ci(avg[0],avg[mode]); ds=pooled[mode]
        print(f'{names[mode]:26s} total={avg[mode].mean():10.4f} branch={np.mean([d["branch"].mean() for d in ds]):9.4f} within={np.mean([d["within"].mean() for d in ds]):9.4f} gain={gain:9.4f} [{lo:.4f}, {hi:.4f}]')
    print('\nCheckpoint posterior Brier scores and frozen mean-greedy losses (per episode):')
    for name,ds in zip(names,pooled):
        z=np.stack([d['checkpoints'] for d in ds]).mean(axis=(0,1))
        print(name, ' | '.join(f'{cp}: {a[4]:.4f}/{a[7]:.4f}' for cp,a in zip(checkpoints,z)))
    print('\nFinal posterior reliability (fixed environment mixture; not prior-predictive calibration):')
    for name,ds in zip(names,pooled):
        probs=np.concatenate([d['checkpoints'][:,-1,3] for d in ds]); truth=np.repeat([b>0 for b,_,_ in envs],args.runs)
        print(name)
        for k in range(10):
            mask=np.minimum((probs*10).astype(int),9)==k
            if mask.any(): print(f'  [{k/10:.1f},{(k+1)/10:.1f}] n={mask.sum():4d} predicted={probs[mask].mean():.4f} observed={truth[mask].mean():.4f}')
    with open(args.csv,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'Saved {len(rows)} rows to {args.csv}')
    fig2, ax=plt.subplots(figsize=(9,4),layout='constrained')
    for name,ds in zip(names,pooled): ax.plot(np.arange(1,args.episodes+1),np.mean([d['curve'] for d in ds],axis=0),label=name)
    ax.set(xscale='log',xlabel='Episode',ylabel='Mean cumulative pseudo-regret',title='Equal-weight environments'); ax.legend(fontsize=8)
    if args.save_prefix:
        fig.savefig(f'{args.save_prefix}_environments.png',dpi=150); fig2.savefig(f'{args.save_prefix}_summary.png',dpi=150)
    if args.no_show: plt.close('all')
    else: plt.show()


if __name__=='__main__':
    main()
