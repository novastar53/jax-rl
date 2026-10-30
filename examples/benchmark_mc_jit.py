import time

import jax
import jax.numpy as jnp

from jax_rl.envs.gridworld import Gridworld
from jax_rl.mc.mc_prediction import mc_prediction
from jax_rl.mc.mc_prediction import mc_prediction_jit


def benchmark(fn):
    start = time.perf_counter()
    V, N = fn()

    V.block_until_ready()
    N.block_until_ready()

    return time.perf_counter() - start


def main():
    env = Gridworld(size=4)
    policy = jnp.full((env.num_states, env.num_actions), 0.25)

    key = jax.random.key(42)
    num_episodes = 1_000

    def reference():
        return mc_prediction(
            env,
            policy,
            key,
            num_episodes=num_episodes,
            visit_mode="first",
        )

    def compiled():
        return mc_prediction_jit(
            env,
            policy,
            key,
            num_episodes=num_episodes,
            visit_mode="first",
        )

    # Warm up the compiled implementation.
    compiled()[0].block_until_ready()

    reference_time = benchmark(reference)

    # Measure repeated execution after compilation.
    compiled_times = [benchmark(compiled) for _ in range(5)]
    compiled_time = min(compiled_times)

    print(f"Reference MC: {reference_time:.4f} s")
    print(f"Compiled MC:  {compiled_time:.4f} s")
    print(f"Speedup:      {reference_time / compiled_time:.1f}x")


if __name__ == "__main__":
    main()
