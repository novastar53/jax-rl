import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jax_rl.envs.gridworld import Gridworld
from jax_rl.td.td_prediction import n_step_td_prediction_jit


def main():
    env = Gridworld(size=4)

    policy = jnp.full(
        (env.num_states, env.num_actions),
        1.0 / env.num_actions,
    )

    true_values = np.array([
        [0, -14, -20, -22],
        [-14, -18, -20, -20],
        [-20, -20, -18, -14],
        [-22, -20, -14, 0],
    ], dtype=np.float32).reshape(-1)

    nonterminal = np.arange(1, env.num_states - 1)

    n_values = [1, 3, 5]
    episode_counts = [10, 30, 100, 300]
    num_runs = 20

    alpha = 0.1
    gamma = 1.0

    keys = jax.random.split(
        jax.random.key(42), num_runs
    )

    results = {}

    for n in n_values:
        errors = np.zeros(
            (num_runs, len(episode_counts))
        )

        for j, num_episodes in enumerate(episode_counts):
            for run, key in enumerate(keys):
                V = n_step_td_prediction_jit(
                    env,
                    policy,
                    key,
                    num_episodes=num_episodes,
                    n=n,
                    alpha=alpha,
                    gamma=gamma,
                    max_steps=1000,
                )

                V = np.asarray(V)

                errors[run, j] = np.sqrt(
                    np.mean(
                        (
                            V[nonterminal]
                            - true_values[nonterminal]
                        ) ** 2
                    )
                )

        results[n] = errors

    fig, ax = plt.subplots(figsize=(8, 5))

    for n, errors in results.items():
        mean = errors.mean(axis=0)
        stderr = errors.std(axis=0, ddof=1) / np.sqrt(num_runs)

        line, = ax.plot(
            episode_counts,
            mean,
            marker="o",
            label=f"n = {n}",
        )

        ax.fill_between(
            episode_counts,
            mean - 1.96 * stderr,
            mean + 1.96 * stderr,
            color=line.get_color(),
            alpha=0.15,
        )

        print(
            f"n={n}: final RMSE = "
            f"{mean[-1]:.4f} ± {stderr[-1]:.4f} (SE)"
        )

    ax.set_xlabel("Training episodes")
    ax.set_ylabel("RMSE against true state values")
    ax.set_title("n-step TD prediction: Effect of n")
    ax.set_xscale("log")
    ax.set_xticks(episode_counts)
    ax.set_xticklabels(episode_counts)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
