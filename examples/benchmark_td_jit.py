import time

import jax
import jax.numpy as jnp

from jax_rl.envs.gridworld import Gridworld
from jax_rl.td.td_prediction import (
    td0_prediction,
    td0_prediction_jit,
)


def benchmark(fn):
    start = time.perf_counter()
    result = fn()
    jax.block_until_ready(result)
    return time.perf_counter() - start


def main():
    env = Gridworld(size=4)
    policy = jnp.full((16, 4), 0.25)

    key = jax.random.key(42)
    num_episodes = 1_000

    def reference():
        return td0_prediction(
            env,
            policy,
            key,
            num_episodes=num_episodes,
            alpha=0.1,
        )

    def compiled():
        return td0_prediction_jit(
            env,
            policy,
            key,
            num_episodes=num_episodes,
            alpha=0.1,
        )

    # Compile and warm up the optimized implementation.
    compiled()

    reference_time = benchmark(reference)

    # Repeat to reduce timing noise.
    jit_times = [
        benchmark(compiled)
        for _ in range(5)
    ]

    jit_time = min(jit_times)

    print(f"Reference TD(0): {reference_time:.4f} s")
    print(f"Compiled TD(0):  {jit_time:.4f} s")
    print(f"Speedup:         {reference_time / jit_time:.1f}x")


if __name__ == "__main__":
    main()
