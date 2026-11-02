import time

import jax

from jax_rl.envs.cliff_walking import CliffWalking
from jax_rl.td.sarsa import sarsa
from jax_rl.td.sarsa import sarsa_jit
from jax_rl.td.q_learning import q_learning
from jax_rl.td.q_learning import q_learning_jit
from jax_rl.utils.policies import EpsilonGreedyPolicy


def benchmark(fn, repeats=3):
    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        times.append(time.perf_counter() - start)

    return min(times)


def main():
    env = CliffWalking()
    policy = EpsilonGreedyPolicy(epsilon=0.1)
    key = jax.random.key(42)

    num_episodes = 300

    algorithms = [
        ("SARSA", sarsa, sarsa_jit),
        ("Q-learning", q_learning, q_learning_jit),
    ]

    for name, reference, compiled in algorithms:
        def run_reference():
            return reference(
                env, policy, key,
                num_episodes=num_episodes,
                alpha=0.5,
            )

        def run_compiled():
            return compiled(
                env, policy, key,
                num_episodes=num_episodes,
                alpha=0.5,
            )

        # Compile before timing.
        jax.block_until_ready(run_compiled())

        reference_time = benchmark(run_reference, repeats=1)
        compiled_time = benchmark(run_compiled)

        print(f"\n{name}")
        print(f"Reference: {reference_time:.4f} s")
        print(f"Compiled:  {compiled_time:.4f} s")
        print(f"Speedup:   {reference_time / compiled_time:.1f}x")


if __name__ == "__main__":
    main()
