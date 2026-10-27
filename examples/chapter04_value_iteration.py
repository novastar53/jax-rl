import jax
import jax.numpy as jnp

from jax_rl.envs.gridworld import Gridworld
from jax_rl.dp.finite_mdp import build_deterministic_model
from jax_rl.dp.policy_improvement import (
    greedy_policy_improvement,
)
from jax_rl.utils.plotting import LivePolicyIteration


def main():
    env = Gridworld(size=4)

    P, R = build_deterministic_model(
        env, jax.random.key(42)
    )

    num_states, num_actions, _ = P.shape

    V = jnp.zeros(num_states)

    # Initial greedy policy induced by V = 0.
    policy = greedy_policy_improvement(
        P, R, V, gamma=1.0
    )

    visualizer = LivePolicyIteration(
        size=4,
        interval=1.0,
    )

    try:
        visualizer.update(
            V,
            policy,
            iteration=0,
            phase="Initial values",
        )

        for iteration in range(1, 101):

            # One Bellman optimality sweep.
            Q = jnp.sum(
                P * (
                    R
                    + V[None, None, :]
                ),
                axis=2,
            )

            V_new = jnp.max(
                Q,
                axis=1,
            )

            delta = jnp.max(
                jnp.abs(V_new - V)
            )

            # Greedy policy induced by the new values.
            policy = greedy_policy_improvement(
                P,
                R,
                V_new,
                gamma=1.0,
            )

            visualizer.update(
                V_new,
                policy,
                iteration=iteration,
                phase=f"Bellman optimality sweep "
                      f"(Δ={float(delta):.6f})",
            )

            print(
                f"Iteration {iteration}: "
                f"delta = {float(delta):.6f}"
            )

            V = V_new

            if delta < 1e-6:
                print("Value function converged.")
                break

        print("\nFinal value function:")
        print(V.reshape(4, 4))

        print("\nFinal policy:")
        print(
            policy.reshape(
                4,
                4,
                num_actions,
            )
        )

    finally:
        visualizer.close()


if __name__ == "__main__":
    main()