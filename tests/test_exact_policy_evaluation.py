import jax
import jax.numpy as jnp
import numpy as np

from jax_rl.dp.policy_evaluation import bellman_sweep, exact_discounted_policy_evaluation
from jax_rl.envs.stochastic_cliff_walking import StochasticCliffWalking


def test_exact_discounted_evaluation_handles_absorbing_and_zero_policy_terminal():
    p = jnp.array([[[0.0, 1.0]], [[0.0, 1.0]]])
    r = jnp.array([[[0.0, -1.0]], [[0.0, 0.0]]])
    for policy in (jnp.array([[1.0], [0.0]]), jnp.ones((2, 1))):
        v = jax.jit(exact_discounted_policy_evaluation)(p, r, policy, 0.99)
        np.testing.assert_allclose(v, [-1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(v, bellman_sweep(p, r, policy, v, 0.99), atol=1e-6)


def test_cliff_uniform_policy_has_small_bellman_residual():
    p, r = StochasticCliffWalking().transition_model()
    policy = jnp.full((48, 4), 0.25).at[47].set(0.0)
    value = exact_discounted_policy_evaluation(p, r, policy, 0.99)
    np.testing.assert_allclose(value, bellman_sweep(p, r, policy, value, 0.99), atol=0.001)
