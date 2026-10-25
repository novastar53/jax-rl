import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jax_rl.envs.bernoulli_bandit import BernoulliBandit
from jax_rl.bandits.beta_bernoulli import BetaBernoulliAgent
from jax_rl.bandits.sample_average import SampleAverageAgent
from jax_rl.bandits.epsilon_greedy import EpsilonGreedyPolicy
from jax_rl.bandits.ucb import UCBPolicy
from jax_rl.bandits.gradient_bandit import GradientBanditAgent


# Experiment configuration
k = 10
num_runs = 2000
num_steps = 10000

env = BernoulliBandit(k=k)

configurations = {
    "Thompson Sampling": (
        BetaBernoulliAgent(k=k),
        None,
    ),
    "UCB (c=0.25)": (
        SampleAverageAgent(k=k, alpha=None, initial_q=0.0),
        UCBPolicy(c=0.25),
    ),
    "UCB (c=0.5)": (
        SampleAverageAgent(k=k, alpha=None, initial_q=0.0),
        UCBPolicy(c=0.50),
    ),
    "UCB (c=1.0)": (
        SampleAverageAgent(k=k, alpha=None, initial_q=0.0),
        UCBPolicy(c=1.0),
    ),
    "UCB (c=2.0)": (
        SampleAverageAgent(k=k, alpha=None, initial_q=0.0),
        UCBPolicy(c=2.0),
    ),
    "Gradient (α=0.1)": (
        GradientBanditAgent(
            k=k, alpha=0.1, use_baseline=True
        ),
        None,
    ),
    "Gradient (α=0.05)": (
        GradientBanditAgent(
            k=k, alpha=0.05, use_baseline=True
        ),
        None,
    ),
    "ε-greedy (ε=0.1)": (
        SampleAverageAgent(k=k, alpha=None, initial_q=0.0),
        EpsilonGreedyPolicy(epsilon=0.1),
    ),
}


def run_experiment(env_key, train_key, agent, policy):
    # Initialize environment and agent
    env_state, _ = env.reset(env_key)
    agent_state = agent.init()

    optimal_value = jnp.max(env_state)
    optimal_action = jnp.argmax(env_state)

    def training_step(carry, t):
        key, agent_state = carry

        key, action_key, reward_key = jax.random.split(key, 3)

        # Action selection
        if isinstance(agent, BetaBernoulliAgent):
            action = agent.select_action(action_key, agent_state)

        elif isinstance(agent, GradientBanditAgent):
            action = agent.select_action(action_key, agent_state)

        elif isinstance(policy, UCBPolicy):
            action = policy.select_action(
                action_key,
                agent_state.Q,
                agent_state.N,
                t,
            )

        else:
            action = policy.select_action(
                action_key,
                agent_state.Q,
            )

        # Environment interaction
        _, _, reward, _, _ = env.step(
            reward_key, env_state, action
        )

        # Update agent state
        agent_state = agent.update(
            agent_state, action, reward
        )

        # Performance metrics
        is_optimal = action == optimal_action
        pseudo_regret = optimal_value - env_state[action]

        new_carry = (key, agent_state)
        output = (reward, is_optimal, pseudo_regret)

        return new_carry, output

    initial_carry = (train_key, agent_state)

    _, (rewards, optimal, regrets) = jax.lax.scan(
        training_step,
        initial_carry,
        xs=jnp.arange(1, num_steps + 1),
    )

    return rewards, optimal, regrets


def main():
    key = jax.random.key(42)

    # Independent environment and training randomness
    env_seed, train_seed = jax.random.split(key)

    # All algorithms receive the same bandit problems
    env_keys = jax.random.split(env_seed, num_runs)

    results = {}

    for i, (name, (agent, policy)) in enumerate(
        configurations.items()
    ):
        # Separate training randomness for each algorithm
        algorithm_key = jax.random.fold_in(train_seed, i)

        train_keys = jax.random.split(
            algorithm_key, num_runs
        )

        batched_experiment = jax.jit(
            jax.vmap(
                lambda env_key, train_key: run_experiment(
                    env_key, train_key, agent, policy
                )
            )
        )

        rewards, optimal, regrets = batched_experiment(
            env_keys, train_keys
        )

        # Average reward per training step
        mean_rewards = jnp.mean(
            rewards.astype(jnp.float32),
            axis=0,
        )

        # Percentage of optimal actions
        optimal_action_pct = 100 * jnp.mean(
            optimal.astype(jnp.float32),
            axis=0,
        )

        # Mean cumulative pseudo-regret
        cumulative_regret = jnp.mean(
            jnp.cumsum(regrets, axis=1),
            axis=0,
        )

        # Per-run final performance for further analysis
        final_optimal_pct = 100 * jnp.mean(
            optimal[:, -100:].astype(jnp.float32),
            axis=1,
        )

        final_regret = jnp.sum(regrets, axis=1)

        results[name] = {
            "mean_rewards": np.asarray(mean_rewards),
            "optimal_action_pct": np.asarray(
                optimal_action_pct
            ),
            "cumulative_regret": np.asarray(
                cumulative_regret
            ),
            "final_optimal_pct": np.asarray(
                final_optimal_pct
            ),
            "final_regret": np.asarray(final_regret),
        }

        print(f"\n{name}")
        print("-" * 45)
        print(
            f"Average reward (last 100 steps): "
            f"{float(jnp.mean(mean_rewards[-100:])):.3f}"
        )
        print(
            f"Optimal actions (last 100 steps): "
            f"{float(jnp.mean(optimal_action_pct[-100:])):.2f}%"
        )
        print(
            f"Cumulative pseudo-regret: "
            f"{float(cumulative_regret[-1]):.2f}"
        )

    # Plot results
    steps = np.arange(1, num_steps + 1)

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5)
    )

    metrics = [
        ("mean_rewards", "Average Reward"),
        ("optimal_action_pct", "% Optimal Actions"),
        ("cumulative_regret", "Cumulative Pseudo-Regret"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        for name, result in results.items():
            ax.plot(
                steps,
                result[metric],
                label=name,
                alpha=0.5,
            )

        ax.set_xlabel("Steps")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(
        "Bandit Algorithms Comparison — Bernoulli Bandits"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
