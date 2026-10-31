import jax
import jax.numpy as jnp

from jax_rl.envs.gridworld import Gridworld
from jax_rl.td.sarsa import sarsa
from jax_rl.utils.policies import EpsilonGreedyPolicy


def main():
    env = Gridworld(size=4)
    policy = EpsilonGreedyPolicy(epsilon=0.1)

    Q = sarsa(
        env,
        policy,
        jax.random.key(42),
        num_episodes=10_000,
        alpha=0.1,
        gamma=1.0,
        max_steps=1000,
    )

    print("Learned action values:")
    print(Q)

    # Greedy action for each state.
    greedy_actions = jnp.argmax(Q, axis=1)

    print("\nGreedy policy:")

    symbols = ("↑", "→", "↓", "←")

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


if __name__ == "__main__":
    main()
