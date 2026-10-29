import jax
import jax.numpy as jnp

from jax_rl.envs.gridworld import Gridworld
from jax_rl.mc.episode import generate_episode
from jax_rl.dp.finite_mdp import build_deterministic_model
from jax_rl.dp.policy_evaluation import iterative_policy_evaluation
from jax_rl.utils.plotting import LiveMCConvergence
from jax_rl.mc.mc_prediction import mc_prediction



def main():
    env = Gridworld(size=4)
    key = jax.random.key(42)

    num_states = env.num_states
    num_actions = env.num_actions

    policy = jnp.full(
        (num_states, num_actions),
        1.0 / num_actions,
    )

    # Ground-truth values using dynamic programming.
    P, R = build_deterministic_model(
        env, jax.random.key(123)
    )

    V_true, _ = iterative_policy_evaluation(
        P, R, policy, gamma=1.0, theta=1e-6
    )

    # Monte Carlo estimates and first-visit counts.
    V = jnp.zeros(num_states, dtype=jnp.float32)
    N = jnp.zeros(num_states, dtype=jnp.int32)

    num_episodes = 10000
    gamma = 1.0
    update_every = 50

    visualizer = LiveMCConvergence(
        size=4, interval=0.01
    )

    # Exclude terminal states from RMSE.
    nonterminal = jnp.arange(1, num_states - 1)


    def compute_rmse(V):
        error = V[1:15] - V_true[1:15]
        return jnp.sqrt(jnp.mean(error**2))


    def on_episode(episode, V, N):
        visualizer.update(
            V,
            episode,
            compute_rmse(V),
        )


    try:
        visualizer.update(
            jnp.zeros(num_states),
            episode=0,
            rmse=compute_rmse(jnp.zeros(num_states)),
        )

        V, N = mc_prediction(
            env,
            policy,
            key,
            num_episodes=num_episodes,
            gamma=gamma,
            visit_mode="first",
            callback=on_episode,
            callback_every=update_every,
        )

        print("\nMonte Carlo estimate:")
        print(V.reshape(4, 4))

        print("\nExact policy values:")
        print(V_true.reshape(4, 4))

        print(f"\nFinal RMSE: {float(compute_rmse(V)):.4f}")

    finally:
        visualizer.close()

if __name__ == "__main__":
    main()
