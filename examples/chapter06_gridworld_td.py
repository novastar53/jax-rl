import jax
import jax.numpy as jnp

from jax_rl.envs.gridworld import Gridworld
from jax_rl.td.td_prediction import td0_prediction


def main():
    env = Gridworld(size=4)

    policy = jnp.full(
        (env.num_states, env.num_actions),
        1.0 / env.num_actions,
    )

    V = td0_prediction(
        env,
        policy,
        jax.random.key(42),
        num_episodes=10_000,
        alpha=0.1,
        gamma=1.0,
    )

    print("TD(0) estimated values:")
    print(V.reshape(4, 4))


if __name__ == "__main__":
    main()