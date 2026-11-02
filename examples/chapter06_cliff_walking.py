import jax
import jax.numpy as jnp

from jax_rl.envs.cliff_walking import CliffWalking
from jax_rl.td.sarsa import sarsa
from jax_rl.td.q_learning import q_learning
from jax_rl.utils.policies import EpsilonGreedyPolicy


def print_policy(Q, env, title):
    symbols = ("↑", "→", "↓", "←")
    greedy_actions = jnp.argmax(Q, axis=1)

    print(f"\n{title}")

    for row in range(env.rows):
        cells = []

        for col in range(env.cols):
            state = row * env.cols + col

            if state == env.start_state:
                cells.append("S")
            elif state == env.goal_state:
                cells.append("G")
            elif row == env.rows - 1:
                cells.append("C")
            else:
                action = int(greedy_actions[state])
                cells.append(symbols[action])

        print("  ".join(cells))


def main():
    env = CliffWalking()
    policy = EpsilonGreedyPolicy(epsilon=0.1)

    key = jax.random.key(42)

    Q_sarsa = sarsa(
        env,
        policy,
        key,
        num_episodes=10_000,
        alpha=0.5,
        gamma=1.0,
        max_steps=1000,
    )

    Q_learning = q_learning(
        env,
        policy,
        key,
        num_episodes=10_000,
        alpha=0.5,
        gamma=1.0,
        max_steps=1000,
    )

    print_policy(Q_sarsa, env, "SARSA greedy policy:")
    print_policy(Q_learning, env, "Q-learning greedy policy:")


if __name__ == "__main__":
    main()
