
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jax_rl.envs.bandit_env import KArmedBandit
from jax_rl.bandits.gradient_bandit import GradientBanditAgent


# Experiment configuration
k = 10
num_runs = 2000
num_steps = 1000
alpha = 0.1

reward_means = [0.0, 2.0, 4.0, 8.0]


def run_experiment(key, env, agent):
    key, env_key = jax.random.split(key)

    env_state, _ = env.reset(env_key)
    agent_state = agent.init()

    def training_step(carry, _):
        key, agent_state, env_state = carry

        key, action_key, reward_key = jax.random.split(key, 3)

        action = agent.select_action(action_key, agent_state)

        env_state, _, reward, _, _ = env.step(
            reward_key, env_state, action
        )

        agent_state = agent.update(
            agent_state, action, reward
        )

        return (key, agent_state, env_state), action

    initial_carry = (key, agent_state, env_state)

    _, actions = jax.lax.scan(
        training_step,
        initial_carry,
        xs=None,
        length=num_steps,
    )

    optimal_action = jnp.argmax(env_state)

    # Percentage of optimal actions in final 100 steps
    final_optimal_pct = 100 * jnp.mean(
        actions[-100:] == optimal_action
    )

    return final_optimal_pct


def main():
    key = jax.random.key(42)
    keys = jax.random.split(key, num_runs)

    results = {}

    for mu in reward_means:
        env = KArmedBandit(k=k, mean=mu)

        for use_baseline in [True, False]:
            agent = GradientBanditAgent(
                k=k,
                alpha=alpha,
                use_baseline=use_baseline,
            )

            batched_experiment = jax.jit(
                jax.vmap(
                    lambda run_key: run_experiment(
                        run_key, env, agent
                    )
                )
            )

            final_pct = batched_experiment(keys)

            results[(mu, use_baseline)] = np.asarray(final_pct)

            print(
                f"mu={mu:<4} "
                f"baseline={str(use_baseline):<5} "
                f"mean={float(jnp.mean(final_pct)):.2f}% "
                f"std={float(jnp.std(final_pct)):.2f}%"
            )

    # Plot 1: Average final performance vs reward offset
    means_with = []
    means_without = []

    for mu in reward_means:
        means_with.append(
            np.mean(results[(mu, True)])
        )
        means_without.append(
            np.mean(results[(mu, False)])
        )

    plt.figure(figsize=(9, 5))

    plt.plot(
        reward_means,
        means_with,
        marker="o",
        label="With baseline",
    )

    plt.plot(
        reward_means,
        means_without,
        marker="o",
        label="Without baseline",
    )

    plt.xlabel("Reward offset (μ)")
    plt.ylabel("Mean % Optimal Actions (Last 100 Steps)")
    plt.title("Effect of Reward Offset on Gradient Bandits")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 2: Distribution of final performance
    fig, axes = plt.subplots(
        2, 4, figsize=(16, 8),
        sharex=True,
        sharey=True,
    )

    bins = np.linspace(0, 100, 21)

    for col, mu in enumerate(reward_means):
        for row, use_baseline in enumerate([True, False]):
            ax = axes[row, col]
            data = results[(mu, use_baseline)]

            ax.hist(
                data,
                bins=bins,
                edgecolor="black",
                alpha=0.75,
            )

            ax.axvline(
                np.mean(data),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(data):.1f}%",
            )

            ax.set_title(
                f"μ={mu}, "
                f"{'baseline' if use_baseline else 'no baseline'}"
            )
            ax.set_xlim(0, 100)
            ax.legend()

            if row == 1:
                ax.set_xlabel("% Optimal Actions")
            if col == 0:
                ax.set_ylabel("Number of Runs")

    fig.suptitle(
        "Distribution of Final Performance by Reward Offset"
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
