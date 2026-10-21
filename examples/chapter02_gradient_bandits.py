import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax_rl.envs.bandit_env import KArmedBandit
from jax_rl.bandits.gradient_bandit import GradientBanditAgent


# Experiment configuration
k = 10
num_runs = 2000
num_steps = 1000

# Figure 2.5 uses q*(a) ~ N(4, 1)
env = KArmedBandit(k=k, mean=0.0)

# Agent configurations
configurations = {
    "α=0.1, with baseline": GradientBanditAgent(
        k=k, alpha=0.1, use_baseline=True
    ),
    "α=0.4, with baseline": GradientBanditAgent(
        k=k, alpha=0.4, use_baseline=True
    ),
    "α=0.1, without baseline": GradientBanditAgent(
        k=k, alpha=0.1, use_baseline=False
    ),
    "α=0.4, without baseline": GradientBanditAgent(
        k=k, alpha=0.4, use_baseline=False
    ),
}


def run_experiment(key, agent):
    # Initialize environment and agent
    key, env_key = jax.random.split(key)

    env_state, _ = env.reset(env_key)
    agent_state = agent.init()

    def training_step(carry, _):
        key, agent_state, env_state = carry

        key, action_key, reward_key = jax.random.split(key, 3)

        # Sample action from learned softmax policy
        action = agent.select_action(action_key, agent_state)

        # Interact with environment
        env_state, _, reward, _, _ = env.step(
            reward_key, env_state, action
        )

        # Update action preferences and reward baseline
        agent_state = agent.update(
            agent_state, action, reward
        )

        return (key, agent_state, env_state), (action, reward)

    initial_carry = (key, agent_state, env_state)

    _, (actions, rewards) = jax.lax.scan(
        training_step,
        initial_carry,
        xs=None,
        length=num_steps,
    )

    return actions, rewards, env_state


def main():
    key = jax.random.key(42)
    keys = jax.random.split(key, num_runs)

    results = {}

    for name, agent in configurations.items():
        batched_experiment = jax.jit(
            jax.vmap(
                lambda run_key: run_experiment(run_key, agent)
            )
        )

        actions, rewards, env_states = batched_experiment(keys)

        # Find the optimal action in each bandit
        optimal_actions = jnp.argmax(env_states, axis=1)

        # Shape: (num_runs, num_steps)
        is_optimal = actions == optimal_actions[:, None]

        # Percentage of optimal actions over the final 100 steps
        # Shape: (num_runs,)
        final_optimal_pct = 100 * jnp.mean(
            is_optimal[:, -100:],
            axis=1,
        )

        results[name] = final_optimal_pct


    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    bins = jnp.linspace(0, 100, 21)

    for ax, (name, final_pct) in zip(
        axes.flat, results.items()
    ):
        ax.hist(
            final_pct,
            bins=bins,
            edgecolor="black",
            alpha=0.75,
        )

        ax.axvline(
            jnp.mean(final_pct),
            color="red",
            linestyle="--",
            label=f"Mean: {jnp.mean(final_pct):.1f}%",
        )

        ax.set_title(name)
        ax.set_xlabel("% Optimal Actions (Last 100 Steps)")
        ax.set_ylabel("Number of Runs")
        ax.set_xlim(0, 100)
        ax.legend()

    plt.suptitle(
        "Gradient Bandits — Distribution of Final Performance"
    )
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
