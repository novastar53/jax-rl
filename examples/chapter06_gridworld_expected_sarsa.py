import jax
import jax.numpy as jnp

from jax_rl.envs.gridworld import Gridworld
from jax_rl.td.expected_sarsa import expected_sarsa_jit
from jax_rl.utils.policies import EpsilonGreedyPolicy


def print_policy(Q, env):
    symbols = ("↑", "→", "↓", "←")
    greedy_actions = jnp.argmax(Q, axis=1)

    print("\nLearned greedy policy:")

    for row in range(env.size):
        cells = []

        for col in range(env.size):
            state = row * env.size + col

            if state in (0, env.num_states - 1):
                cells.append("T")
            else:
                action = int(greedy_actions[state])
                cells.append(symbols[action])

        print("  ".join(cells))


def main():
    env = Gridworld(size=4)
    policy = EpsilonGreedyPolicy(epsilon=0.1)

    Q = expected_sarsa_jit(
        env,
        policy,
        jax.random.key(42),
        num_episodes=10_000,
        alpha=0.1,
        gamma=1.0,
        max_steps=1000,
    )

    V = jnp.max(Q, axis=1)

    # Exact optimal values for the 4x4 Gridworld.
    V_optimal = jnp.array([
        [0.0, -1.0, -2.0, -3.0],
        [-1.0, -2.0, -3.0, -2.0],
        [-2.0, -3.0, -2.0, -1.0],
        [-3.0, -2.0, -1.0, 0.0],
    ])

    print("Learned action values:")
    print(Q)

    print_policy(Q, env)

    print("\nEstimated state values:")
    print(V.reshape(env.size, env.size))

    print("\nOptimal state values:")
    print(V_optimal)

    nonterminal = jnp.arange(1, env.num_states - 1)

    error = V[nonterminal] - V_optimal.reshape(-1)[nonterminal]
    rmse = jnp.sqrt(jnp.mean(error**2))

    print(f"\nRMSE against optimal values: {float(rmse):.4f}")


if __name__ == "__main__":
    main()
