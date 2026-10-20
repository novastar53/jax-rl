
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax_rl.envs.bandit_env import KArmedBandit
from jax_rl.bandits.sample_average import SampleAverageAgent
from jax_rl.bandits.epsilon_greedy import EpsilonGreedyPolicy
from jax_rl.bandits.ucb import UCBPolicy


# Experiment configuration
k = 10
num_runs = 2000
num_steps = 1000

env = KArmedBandit(k=k)

configurations = {
    "UCB (c=2)": (
        SampleAverageAgent(k=k, alpha=None, initial_q=0.0),
        UCBPolicy(c=2.0),
    ),
    "ε-greedy (ε=0.1)": (
        SampleAverageAgent(k=k, alpha=None, initial_q=0.0),
        EpsilonGreedyPolicy(epsilon=0.1),
    ),
}


def run_experiment(key, agent, policy):
    # Separate environment initialization from training randomness
    key, env_key = jax.random.split(key)

    agent_state = agent.init()
    env_state, _ = env.reset(env_key)

    def training_step(carry, t):
        key, agent_state, env_state = carry

        key, action_key, reward_key = jax.random.split(key, 3)

        # Select action using the appropriate policy
        if isinstance(policy, UCBPolicy):
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

        # Interact with environment
        env_state, _, reward, _, _ = env.step(
            reward_key, env_state, action
        )

        # Update action-value estimates
        agent_state = agent.update(agent_state, action, reward)

        return (key, agent_state, env_state), (action, reward)

    initial_carry = (key, agent_state, env_state)

    _, (actions, rewards) = jax.lax.scan(
        training_step,
        initial_carry,
        xs=jnp.arange(1, num_steps + 1),
    )

    return actions, rewards, env_state


def main():
    key = jax.random.key(42)
    keys = jax.random.split(key, num_runs)

    results = {}

    for name, (agent, policy) in configurations.items():

        batched_experiment = jax.jit(
            jax.vmap(
                lambda run_key: run_experiment(run_key, agent, policy)
            )
        )

        actions, rewards, env_states = batched_experiment(keys)

        # Average reward at each step across all runs
        mean_rewards = jnp.mean(rewards, axis=0)

        # Percentage of optimal actions (additional diagnostic)
        optimal_actions = jnp.argmax(env_states, axis=1)

        optimal_action_pct = 100 * jnp.mean(
            actions == optimal_actions[:, None],
            axis=0,
        )

        results[name] = {
            "mean_rewards": mean_rewards,
            "optimal_action_pct": optimal_action_pct,
        }

        print(
            f"{name}: "
            f"Average reward over last 100 steps = "
            f"{float(jnp.mean(mean_rewards[-100:])):.3f}"
        )

    # Reproduce Figure 2.4: Average reward vs steps
    plt.figure(figsize=(10, 6))

    steps = jnp.arange(1, num_steps + 1)

    for name, result in results.items():
        plt.plot(
            steps,
            result["mean_rewards"],
            label=name,
        )

    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.title("UCB vs ε-Greedy — Sutton & Barto Figure 2.4")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
