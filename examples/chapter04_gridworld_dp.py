import jax
import jax.numpy as jnp

from jax_rl.envs.gridworld import Gridworld
from jax_rl.dp.finite_mdp import build_deterministic_model
from jax_rl.dp.policy_evaluation import bellman_sweep
from jax_rl.utils.plotting import LiveValueGrid


def main():
    env = Gridworld(size=4)
    P, R = build_deterministic_model(
        env, jax.random.key(42)
    )

    policy = jnp.full((env.num_states, env.num_actions), 0.25)

    V = jnp.zeros(env.num_states)
    theta = 1e-6

    visualizer = LiveValueGrid(size=4, interval=0.15)
    visualizer.update(V, 0)

    for iteration in range(1, 10001):
        V_new = bellman_sweep(
            P, R, policy, V, gamma=1.0
        )

        delta = jnp.max(jnp.abs(V_new - V))
        V = V_new

        # Show every sweep initially, then every 10 sweeps.
        if iteration <= 20 or iteration % 10 == 0:
            visualizer.update(V, iteration)

        if float(delta) < theta:
            visualizer.update(V, iteration)
            break

    print(f"Converged after {iteration} sweeps")
    print(V.reshape(4, 4))

    visualizer.close()


if __name__ == "__main__":
    main()