import jax
import jax.numpy as jnp

from jax_rl.envs.bandit_env import KArmedBandit
from jax_rl.bandits.sample_average import SampleAverageAgent
from jax_rl.bandits.epsilon_greedy import EpsilonGreedyPolicy


def main():
    # Configuration
    k = 10
    num_steps = 1000
    epsilon = 0.1

    key = jax.random.key(42)

    # Initialize environment, agent, and policy
    env = KArmedBandit(k=k)
    agent = SampleAverageAgent(k=k)
    policy = EpsilonGreedyPolicy(epsilon=epsilon)

    # Initialize states
    key, env_key = jax.random.split(key)
    env_state, observation = env.reset(env_key)
    agent_state = agent.init()

    # Track training history
    actions = []
    rewards = []

    # Training loop
    for t in range(num_steps):
        key, action_key, reward_key = jax.random.split(key, 3)

        # Select action
        action = policy.select_action(action_key, agent_state.Q)

        # Interact with environment
        env_state, observation, reward, terminated, info = env.step(
            reward_key, env_state, action
        )

        # Update action-value estimates
        agent_state = agent.update(agent_state, action, reward)

        # Record results
        actions.append(action)
        rewards.append(reward)

    # Display results
    print("\nFinal Action-Value Estimates")
    print("-" * 55)
    print(f"{'Action':<10}{'True Q*':<15}{'Estimated Q':<15}{'Count':<10}")
    print("-" * 55)

    for a in range(k):
        print(
            f"{a:<10}"
            f"{float(env_state[a]):<15.3f}"
            f"{float(agent_state.Q[a]):<15.3f}"
            f"{int(agent_state.N[a]):<10}"
        )

    print("-" * 55)

    optimal_action = int(jnp.argmax(env_state))
    most_selected_action = int(jnp.argmax(agent_state.N))

    print(f"\nOptimal action: {optimal_action}")
    print(f"Most selected action: {most_selected_action}")
    print(f"Optimal action identified: {optimal_action == most_selected_action}")

    # Additional statistics
    rewards = jnp.array(rewards)
    actions = jnp.array(actions)

    print(f"\nAverage reward: {float(jnp.mean(rewards)):.3f}")
    print(
        f"Percentage of optimal actions: "
        f"{float(jnp.mean(actions == optimal_action) * 100):.1f}%"
    )


if __name__ == "__main__":
    main()