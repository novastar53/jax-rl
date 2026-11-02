import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax_rl.envs.cliff_walking import CliffWalking
from jax_rl.td.sarsa import sarsa_jit
from jax_rl.td.q_learning import q_learning_jit
from jax_rl.utils.policies import EpsilonGreedyPolicy


def main():
    num_runs = 100
    num_episodes = 500

    env = CliffWalking()
    key = jax.random.key(42)

    sarsa_rewards = []
    q_learning_rewards = []

    for run in range(num_runs):
        key, run_key = jax.random.split(key)

        sarsa_policy = EpsilonGreedyPolicy(epsilon=0.1)
        q_learning_policy = EpsilonGreedyPolicy(epsilon=0.1)

        _, rewards = sarsa_jit(
            env,
            sarsa_policy,
            run_key,
            num_episodes=num_episodes,
            alpha=0.5,
            gamma=1.0,
            max_steps=1000,
            return_history=True,
            truncate=True,
        )
        sarsa_rewards.append(rewards)

        _, rewards = q_learning_jit(
            env,
            q_learning_policy,
            run_key,
            num_episodes=num_episodes,
            alpha=0.5,
            gamma=1.0,
            max_steps=1000,
            return_history=True,
            truncate=True,
        )
        q_learning_rewards.append(rewards)

    sarsa_rewards = jnp.stack(sarsa_rewards)
    q_learning_rewards = jnp.stack(q_learning_rewards)

    mean_sarsa = jnp.mean(sarsa_rewards, axis=0)
    mean_q_learning = jnp.mean(q_learning_rewards, axis=0)

    episodes = jnp.arange(1, num_episodes + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        episodes,
        mean_sarsa,
        label="SARSA",
        linewidth=1.8,
    )

    ax.plot(
        episodes,
        mean_q_learning,
        label="Q-learning",
        linewidth=1.8,
    )

    ax.set_title("Cliff Walking: SARSA vs Q-learning")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Average reward per episode")
    ax.set_ylim(-150, 0)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    plt.show()

    print("\nMean reward over final 100 episodes:")
    print(
        f"SARSA: {float(jnp.mean(sarsa_rewards[:, -100:])):.2f}"
    )
    print(
        f"Q-learning: {float(jnp.mean(q_learning_rewards[:, -100:])):.2f}"
    )


if __name__ == "__main__":
    main()
