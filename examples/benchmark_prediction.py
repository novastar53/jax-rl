import time

import jax
import jax.numpy as jnp

from jax_rl.envs.gridworld import Gridworld
from jax_rl.mc.mc_prediction import mc_prediction
from jax_rl.td.td_prediction import td0_prediction


def benchmark(name, fn):
    start = time.perf_counter()
    result = fn()

    # Ensure JAX has completed any pending work.
    jax.block_until_ready(result)

    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.3f} seconds")


def main():
    env = Gridworld(size=4)

    policy = jnp.full(
        (env.num_states, env.num_actions),
        1.0 / env.num_actions,
    )

    num_episodes = 1_000
    key = jax.random.key(42)

    benchmark(
        "Monte Carlo",
        lambda: mc_prediction(
            env,
            policy,
            key,
            num_episodes=num_episodes,
        )[0],
    )

    benchmark(
        "TD(0)",
        lambda: td0_prediction(
            env,
            policy,
            key,
            num_episodes=num_episodes,
        ),
    )


if __name__ == "__main__":
    main()
