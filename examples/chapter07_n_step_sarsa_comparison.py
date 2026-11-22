from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jax_rl.envs.cliff_walking import CliffWalking
from jax_rl.td.sarsa import n_step_sarsa_jit
from jax_rl.utils.policies import EpsilonGreedyPolicy


@partial(
    jax.jit,
    static_argnames=("env", "policy", "max_steps"),
)
def evaluate_policy(
    env,
    policy,
    Q,
    keys,
    max_steps=1000,
):
    """Evaluate a learned Q-table over independent episodes."""

    def run_episode(key):
        key, reset_key = jax.random.split(key)
        env_state, observation = env.reset(reset_key)

        initial = (
            key,
            env_state,
            observation,
            jnp.float32(0.0),
            jnp.int32(0),
            jnp.bool_(False),
        )

        def condition(carry):
            _, _, _, _, steps, terminated = carry
            return (~terminated) & (steps < max_steps)

        def step(carry):
            key, env_state, observation, total_reward, steps, _ = carry

            key, action_key, step_key = jax.random.split(key, 3)

            action = policy.select_action(
                action_key,
                Q[observation],
            )

            next_env_state, next_observation, reward, terminated, _ = (
                env.step(step_key, env_state, action)
            )

            return (
                key,
                next_env_state,
                next_observation,
                total_reward + reward,
                steps + 1,
                terminated,
            )

        final = jax.lax.while_loop(
            condition,
            step,
            initial,
        )

        return final[3]

    return jax.vmap(run_episode)(keys)


def main():
    env = CliffWalking()
    policy = EpsilonGreedyPolicy(epsilon=0.1)

    n_values = [1, 3, 5]
    episode_counts = [10, 30, 100, 300, 500]

    num_runs = 20
    num_evaluation_episodes = 20

    alpha = 0.1
    gamma = 1.0
    max_steps = 1000

    training_keys = jax.random.split(
        jax.random.key(42),
        num_runs,
    )

    evaluation_keys = jax.random.split(
        jax.random.key(123),
        num_runs * num_evaluation_episodes,
    ).reshape(num_runs, num_evaluation_episodes)

    results = {}

    for n in n_values:
        returns = np.zeros(
            (num_runs, len(episode_counts))
        )

        for j, num_episodes in enumerate(episode_counts):
            for run in range(num_runs):
                Q = n_step_sarsa_jit(
                    env,
                    policy,
                    training_keys[run],
                    num_episodes=num_episodes,
                    n=n,
                    alpha=alpha,
                    gamma=gamma,
                    max_steps=max_steps,
                    truncate=True,
                )

                episode_returns = evaluate_policy(
                    env,
                    policy,
                    Q,
                    evaluation_keys[run],
                    max_steps=max_steps,
                )

                returns[run, j] = float(
                    jnp.mean(episode_returns)
                )

        results[n] = returns

        print(
            f"n={n}: final mean return = "
            f"{returns[:, -1].mean():.2f}"
        )

    fig, ax = plt.subplots(figsize=(8, 5))

    for n, returns in results.items():
        mean = returns.mean(axis=0)
        stderr = (
            returns.std(axis=0, ddof=1)
            / np.sqrt(num_runs)
        )

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

    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Mean evaluation episode return")
    ax.set_title("n-step SARSA on Cliff Walking")
    ax.set_xticks(episode_counts)
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
