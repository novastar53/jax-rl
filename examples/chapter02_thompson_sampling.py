
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax_rl.envs.bernoulli_bandit import BernoulliBandit
from jax_rl.bandits.beta_bernoulli import BetaBernoulliAgent
from jax_rl.bandits.sample_average import SampleAverageAgent
from jax_rl.bandits.epsilon_greedy import EpsilonGreedyPolicy


# Experiment configuration
k = 10
num_runs = 2000
num_steps = 1000

env = BernoulliBandit(k=k)

configurations = {
    "Thompson Sampling": (
        BetaBernoulliAgent(k=k),
        None,
    ),
    "ε-greedy (ε=0.1)": (
        SampleAverageAgent(k=k, alpha=None, initial_q=0.0),
        EpsilonGreedyPolicy(epsilon=0.1),
    ),
}


def run_experiment(key, agent, policy):
    key, env_key = jax.random.split(key)

    env_state, _ = env.reset(env_key)
    agent_state = agent.init()

    def training_step(carry, _):
        key, agent_state, env_state = carry

        key, action_key, reward_key = jax.random.split(key, 3)

        # Select action
        if isinstance(agent, BetaBernoulliAgent):
            action = agent.select_action(action_key, agent_state)
        else:
            action = policy.select_action(
                action_key, agent_state.Q
            )

        # Interact with environment
        env_state, _, reward, _, _ = env.step(
            reward_key, env_state, action
        )

        # Update the agent
        agent_state = agent.update(
            agent_state, action, reward
        )

        # Expected instantaneous pseudo-regret
        optimal_value = jnp.max(env_state)
        action_value = env_state[action]
        regret = optimal_value - action_value

        return (key, agent_state, env_state), (
            action, reward, regret
        )

    initial_carry = (key, agent_state, env_state)

    _, (actions, rewards, regrets) = jax.lax.scan(
        training_step,
        initial_carry,
        xs=None,
        length=num_steps,
    )

    return actions, rewards, regrets, env_state


def main():
    key = jax.random.key(42)
    keys = jax.random.split(key, num_runs)

    results = {}

    for name, (agent, policy) in configurations.items():
        batched_experiment = jax.jit(
            jax.vmap(
                lambda run_key: run_experiment(
                    run_key, agent, policy
                )
            )
        )

        actions, rewards, regrets, env_states = (
            batched_experiment(keys)
        )

        # Average reward per time step
        mean_rewards = jnp.mean(rewards, axis=0)

        # Percentage of optimal actions
        optimal_actions = jnp.argmax(env_states, axis=1)

        optimal_action_pct = 100 * jnp.mean(
            actions == optimal_actions[:, None],
            axis=0,
        )

        # Mean cumulative pseudo-regret
        cumulative_regret = jnp.mean(
            jnp.cumsum(regrets, axis=1),
            axis=0,
        )

        results[name] = {
            "mean_rewards": mean_rewards,
            "optimal_action_pct": optimal_action_pct,
            "cumulative_regret": cumulative_regret,
        }

        print(f"\n{name}")
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

    # Plot experiment results
    steps = jnp.arange(1, num_steps + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        ("mean_rewards", "Average reward"),
        ("optimal_action_pct", "% Optimal actions"),
        ("cumulative_regret", "Cumulative pseudo-regret"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        for name, result in results.items():
            ax.plot(
                steps,
                result[metric],
                label=name,
            )

        ax.set_xlabel("Steps")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(
        "Thompson Sampling vs ε-Greedy — Bernoulli Bandits"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
