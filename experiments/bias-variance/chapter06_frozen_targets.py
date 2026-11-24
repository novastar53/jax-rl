"""Compare frozen TD(0) and Monte Carlo targets in the 4x4 Gridworld.

Run: python chapter06_frozen_targets.py
Dependencies: numpy, matplotlib. No learning takes place.

This standalone environment uses the same rules as the study Gridworld:
row-major states, terminals 0 and 15, reward -1 on every transition,
and actions up/right/down/left with moves clipped at the boundary.
It does not depend on, or modify, the jax_rl project.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SIZE = 4
NUM_STATES = SIZE * SIZE
TERMINALS = (0, NUM_STATES - 1)
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))


def next_state(state, action):
    if state in TERMINALS:
        return state
    row, col = divmod(state, SIZE)
    dr, dc = MOVES[action]
    row = min(max(row + dr, 0), SIZE - 1)
    col = min(max(col + dc, 0), SIZE - 1)
    return row * SIZE + col


def exact_values():
    """Solve v = -1 + P v for nonterminal states under uniform actions."""
    states = [s for s in range(NUM_STATES) if s not in TERMINALS]
    index = {s: i for i, s in enumerate(states)}
    matrix = np.eye(len(states))
    for state in states:
        for action in range(4):
            successor = next_state(state, action)
            if successor not in TERMINALS:
                matrix[index[state], index[successor]] -= 0.25
    values = np.zeros(NUM_STATES)
    values[states] = np.linalg.solve(matrix, -np.ones(len(states)))
    return values


def sample_targets(num_episodes, seed, start_state=5, max_steps=100_000):
    """Use independent rollouts, recording one MC and TD target per episode.

    max_steps is a failure guard: raise, never use a truncated MC return.
    Both targets use the same episode's initial transition. V stays zero.
    """
    rng = np.random.default_rng(seed)
    frozen_v = np.zeros(NUM_STATES)
    mc_targets = np.empty(num_episodes)
    td_targets = np.empty(num_episodes)

    for episode in range(num_episodes):
        state = start_state
        total_reward = 0.0
        for step in range(max_steps):
            action = int(rng.integers(4))  # Fixed uniform-random policy.
            successor = next_state(state, action)
            reward = -1.0
            terminated = successor in TERMINALS
            if step == 0:
                td_targets[episode] = reward
                if not terminated:
                    td_targets[episode] += frozen_v[successor]
            total_reward += reward  # gamma = 1.
            state = successor
            if terminated:
                mc_targets[episode] = total_reward
                break
        else:
            raise RuntimeError("Episode did not terminate; no results retained.")

    return mc_targets, td_targets


def summarize(targets, truth):
    mean = targets.mean()
    bias = mean - truth
    variance = targets.var(ddof=0)
    mse = np.mean((targets - truth) ** 2)
    # Exact decomposition for the empirical distribution of these samples.
    np.testing.assert_allclose(mse, bias**2 + variance, atol=1e-10)
    return mean, bias, variance, mse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("frozen_targets.png"))
    args = parser.parse_args()
    if args.episodes < 2:
        parser.error("--episodes must be at least 2")

    values = exact_values()
    truth = values[5]
    mc, td = sample_targets(args.episodes, args.seed)
    np.testing.assert_allclose(td, -1.0)
    print("Exact values under the fixed uniform-random policy:")
    print(values.reshape(SIZE, SIZE).round(6))
    print(f"\nStart state: 5; true value: {truth:.6f}")
    print(f"Episodes: {args.episodes}; seed: {args.seed}; frozen V = 0\n")
    print(f"{'Method':<10} {'Mean':>12} {'Bias':>12} {'Variance':>12} {'MSE':>12}")
    for name, targets in (("MC", mc), ("TD(0)", td)):
        stats = summarize(targets, truth)
        print(f"{name:<10}" + "".join(f"{x:>13.6f}" for x in stats))
    print(f"\nMC mean standard error: {mc.std(ddof=1) / np.sqrt(len(mc)):.6f}")
    print("MC's empirical bias fluctuates around zero across repeated samples.")
    print("TD's true target bias here is +17, and its variance is zero.")
    print("These are target statistics, not statistics of trained value estimates.")

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    edges = np.linspace(min(mc.min(), truth) - 1, 0, 65)
    for ax, name, targets, color in (
        (axes[0], "Monte Carlo: complete sampled return", mc, "#2673b8"),
        (axes[1], "TD(0): one-step target with frozen V = 0", td, "#ce7025"),
    ):
        weights = np.full(len(targets), 1 / len(targets))
        ax.hist(targets, bins=edges, weights=weights, color=color, alpha=0.8)
        ax.axvline(truth, color="#222222", linestyle="--", label=f"True value = {truth:.0f}")
        ax.axvline(targets.mean(), color="#b3314a", label=f"Sample mean = {targets.mean():.2f}")
        ax.set(title=name, ylabel="Fraction of episodes")
        ax.legend(loc="upper left", frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Target for state 5 (gamma = 1)")
    fig.suptitle("Bias is displacement; variance is spread", fontsize=16)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"\nPlot saved to {args.output}")


if __name__ == "__main__":
    main()
