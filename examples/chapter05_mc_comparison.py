import jax
import jax.numpy as jnp

from jax_rl.envs.gridworld import Gridworld
from jax_rl.dp.finite_mdp import build_deterministic_model
from jax_rl.dp.policy_evaluation import iterative_policy_evaluation
from jax_rl.mc.episode import generate_episode
from jax_rl.mc.mc_prediction import (
    compute_returns,
    update_mc_values,
)
from jax_rl.utils.plotting import LiveMCComparison


def main():
    size = 4
    num_episodes = 10_000
    gamma = 1.0
    update_every = 50

    env = Gridworld(size=size)
    key = jax.random.key(42)

    num_states = env.num_states
    num_actions = env.num_actions

    policy = jnp.full(
        (num_states, num_actions),
        1.0 / num_actions,
    )

    # Ground-truth values from dynamic programming.
    P, R = build_deterministic_model(
        env, jax.random.key(123)
    )

    V_true, _ = iterative_policy_evaluation(
        P,
        R,
        policy,
        gamma=gamma,
        theta=1e-6,
    )

    # Independent estimates, shared experience.
    V_first = jnp.zeros(num_states, dtype=jnp.float32)
    N_first = jnp.zeros(num_states, dtype=jnp.int32)

    V_every = jnp.zeros(num_states, dtype=jnp.float32)
    N_every = jnp.zeros(num_states, dtype=jnp.int32)

    # Our Gridworld has terminal states 0 and 15.
    nonterminal = jnp.arange(1, num_states - 1)

    def compute_rmse(V):
        error = V[nonterminal] - V_true[nonterminal]
        return jnp.sqrt(jnp.mean(error**2))

    visualizer = LiveMCComparison(
        size=size,
        interval=0.01,
    )

    try:
        visualizer.update(
            episode=0,
            V_first=V_first,
            V_every=V_every,
            rmse_first=compute_rmse(V_first),
            rmse_every=compute_rmse(V_every),
        )

        for episode in range(1, num_episodes + 1):
            key, episode_key = jax.random.split(key)

            # Generate one shared episode.
            states, _, rewards = generate_episode(
                env, policy, episode_key
            )

            # Compute returns only once.
            returns = compute_returns(rewards, gamma)

            # Update first-visit estimate.
            V_first, N_first = update_mc_values(
                V_first,
                N_first,
                states,
                returns,
                visit_mode="first",
            )

            # Update every-visit estimate.
            V_every, N_every = update_mc_values(
                V_every,
                N_every,
                states,
                returns,
                visit_mode="every",
            )

            # Update both visualizations together.
            if (
                episode % update_every == 0
                or episode == num_episodes
            ):
                visualizer.update(
                    episode=episode,
                    V_first=V_first,
                    V_every=V_every,
                    rmse_first=compute_rmse(V_first),
                    rmse_every=compute_rmse(V_every),
                )

        print("\nFirst-visit MC:")
        print(V_first.reshape(size, size))
        print(
            f"RMSE: {float(compute_rmse(V_first)):.4f}"
        )
        print(
            f"Total updates: {int(jnp.sum(N_first))}"
        )

        print("\nEvery-visit MC:")
        print(V_every.reshape(size, size))
        print(
            f"RMSE: {float(compute_rmse(V_every)):.4f}"
        )
        print(
            f"Total updates: {int(jnp.sum(N_every))}"
        )

        print("\nExact policy values:")
        print(V_true.reshape(size, size))

    finally:
        visualizer.close()


if __name__ == "__main__":
    main()
