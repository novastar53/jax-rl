import time

import jax
import jax.numpy as jnp

from jax_rl.envs.gridworld import Gridworld
from jax_rl.td.td_prediction import (
    n_step_td_prediction,
    n_step_td_prediction_jit,
)


def benchmark(fn, repeats=5):
    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        times.append(time.perf_counter() - start)

    return min(times)


def main():
    env = Gridworld(size=4)
    policy = jnp.full(
        (env.num_states, env.num_actions),
        1.0 / env.num_actions,
    )

    key = jax.random.key(42)

    kwargs = dict(
        num_episodes=1_000,
        n=3,
        alpha=0.1,
        gamma=1.0,
        max_steps=1_000,
    )

    def run_reference():
        return n_step_td_prediction(
            env, policy, key, **kwargs
        )

    def run_jit():
        return n_step_td_prediction_jit(
            env, policy, key, **kwargs
        )

    # Compile and warm up before benchmarking.
    V_jit = run_jit()
    jax.block_until_ready(V_jit)

    # Check correctness.
    V_reference = run_reference()

    if not bool(jnp.allclose(V_reference, V_jit, atol=1e-5)):
        raise AssertionError("Reference and JIT results differ.")

    reference_time = benchmark(run_reference, repeats=3)
    jit_time = benchmark(run_jit, repeats=5)

    print(f"Episodes: {kwargs['num_episodes']}")
    print(f"n: {kwargs['n']}")
    print(f"Reference: {reference_time:.6f} s")
    print(f"JIT:       {jit_time:.6f} s")
    print(f"Speedup:   {reference_time / jit_time:.1f}x")
    print(f"Max |ΔV|:  {float(jnp.max(jnp.abs(V_reference - V_jit))):.8f}")


if __name__ == "__main__":
    main()
