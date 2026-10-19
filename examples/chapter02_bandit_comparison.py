import jax
import jax.numpy as jnp

from jax_rl.envs.bandit_env import KArmedBandit
from jax_rl.bandits.sample_average import SampleAverageAgent
from jax_rl.bandits.epsilon_greedy import EpsilonGreedyPolicy

k = 10
num_runs = 2000
num_steps = 1000
epsilons = [0.0, 0.01, 0.1]


key = jax.random.key(42)

# Initialize environment, agent, and policy
env = KArmedBandit(k=k)
agent = SampleAverageAgent(k=k)


def run_experiment(key, policy):
    key, env_key = jax.random.split(key)
    agent_state = agent.init()
    env_state, observation = env.reset(env_key)

    def training_step(carry, _):
        key, agent_state, env_state = carry 

        key, action_key, reward_key = jax.random.split(key, 3)
        action = policy.select_action(action_key, agent_state.Q)

        # Interact with environment
        env_state, _, reward, _, _ = env.step(
            reward_key, env_state, action
        )

        # Update action-value estimates
        agent_state = agent.update(agent_state, action, reward)

        return (key, agent_state, env_state), (action, reward)

    initial_carry = key, agent_state, env_state

    final_carry, (actions, rewards) = jax.lax.scan(
        training_step,
        initial_carry,
        xs=None,
        length=num_steps
    )

    return actions, rewards, env_state

key = jax.random.key(42)
keys = jax.random.split(key, num_runs)

results = {}

for epsilon in epsilons:
    policy = EpsilonGreedyPolicy(epsilon=epsilon)

    batched_experiment = jax.jit(
        jax.vmap(lambda key: run_experiment(key, policy))
    )

    actions, rewards, env_states = batched_experiment(keys)

    optimal_actions = jnp.argmax(env_states, axis=1)

    results[epsilon] = {
        "mean_rewards": jnp.mean(rewards, axis=0),
        "optimal_action_pct": 100 * jnp.mean(
            actions == optimal_actions[:, None],
            axis=0,
        ),
    }


import matplotlib.pyplot as plt

for epsilon in epsilons:
    plt.plot(
        results[epsilon]["mean_rewards"],
        label=f"epsilon={epsilon}"
    )

plt.xlabel("Steps")
plt.ylabel("Average reward")
plt.legend()
plt.show()

for epsilon in epsilons:
    plt.plot(
        results[epsilon]["optimal_action_pct"],
        label=f"epsilon={epsilon}"
    )

plt.xlabel("Steps")
plt.ylabel("% Optimal action")
plt.legend()
plt.show()