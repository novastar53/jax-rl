"""Bias and variance of LEARNED TD(0) and first-visit MC value estimates.

Run from a project: uv run python examples/chapter06_learning_bias_variance.py
Dependencies: numpy, matplotlib. No dependency on the frozen-target script.

Standalone reference Gridworld: 4x4, row-major states 0..15, terminals
0 and 15, clipped boundary moves, reward -1 including entry to a terminal.
Uniform-random policy, gamma=1, start state 5, initial estimates zero.
The policy stays fixed: this is prediction, not SARSA or Q-learning.

Independent runs are batched for speed. Within a run both learners see
identical complete episodes. TD updates online; MC updates each visited
state once, at episode end, using its first-visit return. Both use the
same constant alpha. This compares these settings, not optimally tuned
algorithms or equal numbers of updates (TD can revisit a state).
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SIZE = 4
NUM_STATES = SIZE**2
START = 5
TERMINALS = (0, 15)
NONTERMINALS = np.arange(1, 15)


def transition_table():
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    table = np.empty((NUM_STATES, 4), dtype=int)
    for state in range(NUM_STATES):
        row, col = divmod(state, SIZE)
        for action, (dr, dc) in enumerate(moves):
            nr = min(max(row + dr, 0), SIZE - 1)
            nc = min(max(col + dc, 0), SIZE - 1)
            table[state, action] = state if state in TERMINALS else nr * SIZE + nc
    return table


def exact_values(table):
    # Nonterminal states are 1..14, so their equation index is state - 1.
    matrix = np.eye(len(NONTERMINALS))
    for state in NONTERMINALS:
        for successor in table[state]:
            if successor not in TERMINALS:
                matrix[state - 1, successor - 1] -= 0.25
    values = np.zeros(NUM_STATES)
    values[NONTERMINALS] = np.linalg.solve(matrix, -np.ones(14))
    return values


def train(num_runs, num_episodes, alpha, seed, checkpoints, max_steps=100_000):
    """Return snapshots shaped (checkpoints, independent runs, states)."""
    rng = np.random.default_rng(seed)
    table = transition_table()
    td_v = np.zeros((num_runs, NUM_STATES))
    mc_v = np.zeros_like(td_v)
    td_history, mc_history = [], []
    checkpoints = set(checkpoints)

    for episode in range(1, num_episodes + 1):
        states = np.full(num_runs, START, dtype=int)
        active = np.ones(num_runs, dtype=bool)
        lengths = np.zeros(num_runs, dtype=int)
        first_visit = np.full((num_runs, NUM_STATES), -1, dtype=int)

        for step in range(max_steps):
            rows = np.flatnonzero(active)
            old_states = states[rows]
            unseen = first_visit[rows, old_states] == -1
            first_visit[rows[unseen], old_states[unseen]] = step

            # Random actions do not depend on either learner's estimates.
            actions = rng.integers(4, size=len(rows))
            successors = table[old_states, actions]
            terminated = (successors == TERMINALS[0]) | (successors == TERMINALS[1])

            # TD(0): update the state just left using the CURRENT next value.
            target = -1.0 + np.where(terminated, 0.0, td_v[rows, successors])
            td_v[rows, old_states] += alpha * (target - td_v[rows, old_states])

            states[rows] = successors
            lengths[rows] += 1
            active[rows[terminated]] = False
            if not active.any():
                break
        else:
            raise RuntimeError("Episode exceeded max_steps; no truncated returns used.")

        # First-visit MC: from first visit t to termination T, G_t = -(T-t).
        rows, visited_states = np.nonzero(first_visit >= 0)
        returns = -(lengths[rows] - first_visit[rows, visited_states]).astype(float)
        mc_v[rows, visited_states] += alpha * (returns - mc_v[rows, visited_states])

        if episode in checkpoints:
            td_history.append(td_v.copy())
            mc_history.append(mc_v.copy())

    return {"TD(0)": np.asarray(td_history), "MC": np.asarray(mc_history)}


def statistics(history, truth):
    # Average over runs, keeping checkpoint and state axes.
    mean = history.mean(axis=1)
    bias = mean - truth
    variance = history.var(axis=1, ddof=0)
    mse = ((history - truth) ** 2).mean(axis=1)
    np.testing.assert_allclose(mse, bias**2 + variance, rtol=1e-10, atol=1e-10)
    return {"mean": mean, "bias": bias, "bias_squared": bias**2,
            "variance": variance, "mse": mse}


def plot_results(checkpoints, stats, truth, alpha, num_runs, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {"TD(0)": "#ce7025", "MC": "#2673b8"}
    for name, result in stats.items():
        color = colors[name]
        mean = result["mean"][:, START]
        sd = np.sqrt(result["variance"][:, START])
        axes[0, 0].plot(checkpoints, mean, label=name, color=color)
        axes[0, 0].fill_between(checkpoints, mean - sd, mean + sd,
                                color=color, alpha=0.12)
        for ax, metric in ((axes[0, 1], "bias_squared"),
                           (axes[1, 0], "variance"), (axes[1, 1], "mse")):
            ax.plot(checkpoints, result[metric][:, START], label=name, color=color)

    axes[0, 0].axhline(truth[START], color="#222222", linestyle="--", label="True value = -18")
    for ax, title, ylabel in zip(axes.flat,
        ("Mean learned value (shading: +/- 1 SD across runs)", "Squared bias of learned values",
         "Variance of learned values", "MSE = squared bias + variance"),
        ("Value estimate", "Squared bias", "Variance", "Mean squared error")):
        ax.set(title=title, xlabel="Training episodes per run", ylabel=ylabel, xscale="log")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
        ax.legend(frameon=False)
    fig.suptitle(f"Learning in Gridworld: state 5 | {num_runs} runs | alpha = {alpha:g}", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_distributions(checkpoints, histories, truth, path):
    """Show distributions of learned V(5) across runs at fixed training times."""
    stages = [n for n in (10, 100, 1000) if n in checkpoints]
    if not stages:
        stages = [int(checkpoints[-1])]
    samples = [history[np.flatnonzero(checkpoints == n)[0], :, START]
               for history in histories.values() for n in stages]
    low = min(truth[START], min(x.min() for x in samples)) - 1
    high = max(truth[START], max(x.max() for x in samples)) + 1
    bins = np.linspace(low, high, 61)
    fig, axes = plt.subplots(2, len(stages), figsize=(5 * len(stages), 7),
                             sharex=True, squeeze=False)
    for row, (name, history) in enumerate(histories.items()):
        for col, n in enumerate(stages):
            ax = axes[row, col]
            values = history[np.flatnonzero(checkpoints == n)[0], :, START]
            color = "#ce7025" if name == "TD(0)" else "#2673b8"
            ax.hist(values, bins=bins, weights=np.ones(len(values)) / len(values),
                    color=color, alpha=0.8)
            ax.axvline(truth[START], color="#222222", linestyle="--", label="True value (-18)")
            ax.axvline(values.mean(), color="#ae3150", label="Mean learned value")
            ax.set(title=f"{name}: after {n:,} episodes", xlabel="Learned value of state 5",
                   ylabel="Fraction of training runs")
            ax.text(0.03, 0.95,
                    f"Bias = {values.mean() - truth[START]:+.2f}\nSD = {values.std():.2f}",
                    transform=ax.transAxes, va="top", fontsize=10,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("Learned estimates across independent runs: gap shows bias, spread shows variance",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("learning_bias_variance"))
    args = parser.parse_args()
    if args.runs < 2 or args.episodes < 1 or not 0 < args.alpha <= 1:
        parser.error("Require runs >= 2, episodes >= 1, and 0 < alpha <= 1")

    # Dense early measurements plus logarithmically spaced later checkpoints.
    checkpoints = np.unique(np.r_[np.arange(1, min(args.episodes, 10) + 1),
        np.geomspace(1, args.episodes, min(args.episodes, 80)).astype(int),
        [n for n in (10, 100, 1000) if n <= args.episodes], args.episodes]).astype(int)
    truth = exact_values(transition_table())
    print(f"Training {args.runs} independent runs for {args.episodes} episodes each...", flush=True)
    histories = train(args.runs, args.episodes, args.alpha, args.seed, checkpoints)
    stats = {name: statistics(history, truth) for name, history in histories.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_results(checkpoints, stats, truth, args.alpha, args.runs,
                 args.output_dir / "learning_bias_variance.png")
    plot_distributions(checkpoints, histories, truth,
                       args.output_dir / "learned_value_distributions.png")

    # Save raw estimates so other states and summaries can be examined later.
    np.savez_compressed(args.output_dir / "value_estimates.npz",
        checkpoints=checkpoints, truth=truth, td=histories["TD(0)"], mc=histories["MC"],
        alpha=args.alpha, seed=args.seed, num_runs=args.runs, num_episodes=args.episodes)
    with (args.output_dir / "statistics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episodes", "method", "state", "mean", "bias", "bias_squared", "variance", "mse"])
        for k, episode in enumerate(checkpoints):
            for name, result in stats.items():
                for state in NONTERMINALS:
                    writer.writerow([episode, name, state] + [result[m][k, state]
                        for m in ("mean", "bias", "bias_squared", "variance", "mse")])

    print(f"\nState 5 after {args.episodes} episodes; true value = {truth[START]:.6f}")
    print(f"{'Method':<10} {'Mean':>12} {'Bias':>12} {'Bias^2':>12} {'Variance':>12} {'MSE':>12}")
    for name, result in stats.items():
        print(f"{name:<10}" + "".join(f"{result[m][-1, START]:>12.4f}"
              for m in ("mean", "bias", "bias_squared", "variance", "mse")))
    print("\nVariance is across independent training runs at a fixed episode count.")
    print("Shading is standard deviation of estimates, not uncertainty of their mean.")
    print("MC targets are unbiased; finite-training MC estimates can still be biased.")
    print("Constant alpha maintains fluctuations; this is not a convergence guarantee.")
    print(f"Outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
