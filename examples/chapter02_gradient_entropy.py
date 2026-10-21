import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jax_rl.envs.bandit_env import KArmedBandit
from jax_rl.bandits.gradient_bandit import GradientBanditAgent


# Experiment configuration
k = 10
num_runs = 2000
num_steps = 1000
alpha = 0.1

reward_means = [0.0, 8.0]
baseline_options = [True, False]

# Successful = optimal action chosen at least 90%
# of the time during the final 100 steps.
success_threshold = 90.0
evaluation_window = 100


def run_experiment(key, env, agent):
    key, env_key = jax.random.split(key)

    env_state, _ = env.reset(env_key)
    agent_state = agent.init()

    optimal_action = jnp.argmax(env_state)

    def training_step(carry, _):
        key, agent_state, env_state = carry

        key, action_key, reward_key = jax.random.split(key, 3)

        # Measure policy entropy BEFORE the action/update
        log_probs = jax.nn.log_softmax(agent_state.H)
        probs = jnp.exp(log_probs)

        entropy = -jnp.sum(probs * log_probs)

        # Select action
        action = agent.select_action(action_key, agent_state)

        # Environment interaction
        env_state, _, reward, _, _ = env.step(
            reward_key, env_state, action
        )

        # Update preferences
        agent_state = agent.update(
            agent_state, action, reward
        )

        return (
            key, agent_state, env_state
        ), (
            action, entropy
        )

    initial_carry = (key, agent_state, env_state)

    _, (actions, entropies) = jax.lax.scan(
        training_step,
        initial_carry,
        xs=None,
        length=num_steps,
    )

    return actions, entropies, optimal_action


def main():
    key = jax.random.key(42)
    keys = jax.random.split(key, num_runs)

    results = {}

    for mu in reward_means:
        env = KArmedBandit(k=k, mean=mu)

        for use_baseline in baseline_options:
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

            actions, entropies, optimal_actions = (
                batched_experiment(keys)
            )

            # Convert to NumPy for analysis and plotting
            actions = np.asarray(actions)
            entropies = np.asarray(entropies)
            optimal_actions = np.asarray(optimal_actions)

            # Evaluate each run over its final 100 steps
            final_optimal_pct = 100 * np.mean(
                actions[:, -evaluation_window:]
                == optimal_actions[:, None],
                axis=1,
            )

            successful = (
                final_optimal_pct >= success_threshold
            )

            results[(mu, use_baseline)] = {
                "entropies": entropies,
                "final_optimal_pct": final_optimal_pct,
                "successful": successful,
            }

            print(
                f"mu={mu:<4.1f} "
                f"baseline={str(use_baseline):<5} "
                f"mean optimal={np.mean(final_optimal_pct):5.1f}% "
                f"success rate={100 * np.mean(successful):5.1f}% "
                f"final entropy={np.mean(entropies[:, -1]):.3f}"
            )

    # --------------------------------------------------
    # Plot 1: Mean entropy for all configurations
    # --------------------------------------------------

    fig, axes = plt.subplots(
        1, 2, figsize=(15, 5.5)
    )

    steps = np.arange(1, num_steps + 1)

    ax = axes[0]

    for (mu, use_baseline), result in results.items():
        mean_entropy = np.mean(
            result["entropies"], axis=0
        )

        label = (
            f"μ={mu:g}, "
            f"{'baseline' if use_baseline else 'no baseline'}"
        )

        ax.plot(
            steps,
            mean_entropy,
            label=label,
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean policy entropy")
    ax.set_title("Mean Policy Entropy During Training")
    ax.set_ylim(0, np.log(k) + 0.1)
    ax.grid(alpha=0.3)
    ax.legend()

    # --------------------------------------------------
    # Plot 2: Success vs failure, reward offset mu=8
    # --------------------------------------------------

    ax = axes[1]
    mu = 8.0

    for use_baseline in baseline_options:
        result = results[(mu, use_baseline)]

        entropies = result["entropies"]
        successful = result["successful"]

        for is_success, status in [
            (True, "successful"),
            (False, "unsuccessful"),
        ]:
            mask = successful == is_success
            count = np.sum(mask)

            if count == 0:
                continue

            mean_entropy = np.mean(
                entropies[mask],
                axis=0,
            )

            label = (
                f"{'Baseline' if use_baseline else 'No baseline'}"
                f" — {status} (n={count})"
            )

            ax.plot(
                steps,
                mean_entropy,
                label=label,
            )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean policy entropy")
    ax.set_title("Entropy by Final Outcome (μ=8)")
    ax.set_ylim(0, np.log(k) + 0.1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(
        "Gradient Bandits: Policy Entropy and Premature Concentration",
        fontsize=13,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
