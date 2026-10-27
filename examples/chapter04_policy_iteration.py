
import jax
import jax.numpy as jnp

from jax_rl.envs.gridworld import Gridworld
from jax_rl.dp.finite_mdp import build_deterministic_model
from jax_rl.dp.policy_evaluation import (
    iterative_policy_evaluation,
)
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

    policy = jnp.full(
        (num_states, num_actions),
        1.0 / num_actions,
    )

    V = jnp.zeros(num_states)

    visualizer = LivePolicyIteration(
        size=4, interval=1.0
    )

    try:
        # Show the initial random policy.
        visualizer.update(
            V, policy, iteration=0,
            phase="Initial random policy"
        )

        for iteration in range(1, 101):

            # Evaluate the current policy.
            V, sweeps = iterative_policy_evaluation(
                P,
                R,
                policy,
                gamma=1.0,
                theta=1e-6,
            )

            visualizer.update(
                V, policy, iteration,
                phase=f"Evaluated ({sweeps} sweeps)"
            )

            # Construct the improved policy.
            new_policy = greedy_policy_improvement(
                P, R, V, gamma=1.0
            )

            visualizer.update(
                V, new_policy, iteration,
                phase="Greedy improvement"
            )

            print(
                f"Iteration {iteration}: "
                f"{sweeps} Bellman sweeps"
            )

            if bool(jnp.allclose(new_policy, policy)):
                print("Policy stable.")
                break

            policy = new_policy

        else:
            # The last improved policy may not have
            # been evaluated yet.
            policy = new_policy
            V, _ = iterative_policy_evaluation(
                P, R, policy, gamma=1.0
            )
            visualizer.update(
                V, policy, 100,
                phase="Iteration limit reached"
            )

        print("\nFinal value function:")
        print(V.reshape(4, 4))

        print("\nFinal policy:")
        print(policy.reshape(4, 4, num_actions))

    finally:
        visualizer.close()


if __name__ == "__main__":
    main()
