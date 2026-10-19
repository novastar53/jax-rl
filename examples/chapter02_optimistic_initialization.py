import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax_rl.envs.bandit_env import KArmedBandit
from jax_rl.bandits.sample_average import SampleAverageAgent
from jax_rl.bandits.epsilon_greedy import EpsilonGreedyPolicy


k = 10
num_runs = 2000
num_steps = 1000

# Initialize environment
env = KArmedBandit(k=k)

# Define agent-policy configurations
configurations = {
    "Optimistic greedy (Q₀=5, ε=0)": (
        SampleAverageAgent(k=k, alpha=0.1, initial_q=5.0),
        EpsilonGreedyPolicy(epsilon=0.0),
    ),
    "ε-greedy (Q₀=0, ε=0.1)": (
        SampleAverageAgent(k=k, alpha=0.1, initial_q=0.0),
        EpsilonGreedyPolicy(epsilon=0.1),
    ),
}


def run_experiment(key, agent, policy):
    key, env_key = jax.random.split(key)

    agent_state = agent.init()
    env_state, _ = env.reset(env_key)

    def training_step(carry, _):
        key, agent_state, env_state = carry

        key, action_key, reward_key = jax.random.split(key, 3)

        # Select action
        action = policy.select_action(action_key, agent_state.Q)

        # Interact with environment
        env_state, _, reward, _, _ = env.step(
            reward_key, env_state, action
        )

        # Update action-value estimates
        agent_state = agent.update(agent_state, action, reward)

        return (key, agent_state, env_state), (action, reward)

    initial_carry = (key, agent_state, env_state)

    final_carry, (actions, rewards) = jax.lax.scan(
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

    for name, (agent, policy) in configurations.items():

        batched_experiment = jax.jit(
            jax.vmap(
                lambda run_key: run_experiment(run_key, agent, policy)
            )
        )

        actions, rewards, env_states = batched_experiment(keys)

        # True optimal action in each bandit
        optimal_actions = jnp.argmax(env_states, axis=1)

        # Percentage of optimal actions at each step
        optimal_action_pct = 100 * jnp.mean(
            actions == optimal_actions[:, None],
            axis=0,
        )

        results[name] = optimal_action_pct

    # Plot results
    plt.figure(figsize=(10, 6))

    for name, optimal_action_pct in results.items():
        plt.plot(
            jnp.arange(1, num_steps + 1),
            optimal_action_pct,
            label=name,
        )

    plt.xlabel("Steps")
    plt.ylabel("% Optimal action")
    plt.title("Optimistic Initial Values — Sutton & Barto Figure 2.3")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()