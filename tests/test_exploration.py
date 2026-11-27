import jax
import jax.numpy as jnp
import numpy as np

from jax_rl.utils.exploration import (
    count_uncertainty,
    gaussian_thompson_action,
    local_ucb_action,
    propagated_uncertainty,
)


def test_ucb_prioritizes_unvisited_and_breaks_ties():
    q = jnp.array([1e6, -100.0, -200.0])
    n = jnp.array([1, 0, 0])
    keys = jax.random.split(jax.random.key(1), 200)
    actions = jax.jit(jax.vmap(lambda k: local_ucb_action(k, q, n)))(keys)
    assert set(np.asarray(actions).tolist()) == {1, 2}


def test_ucb_uses_local_counts_and_scores():
    action = local_ucb_action(jax.random.key(1), jnp.array([0.0, 1.0]), jnp.array([1, 100]), 2.0)
    assert int(action) == 0


def test_thompson_zero_uncertainty_and_jit_vmap():
    keys = jax.random.split(jax.random.key(2), 10)
    sample = jax.jit(jax.vmap(lambda k: gaussian_thompson_action(
        k, jnp.array([1.0, 3.0]), jnp.array([0.0, 0.0])
    )))(keys)
    assert np.all(np.asarray(sample) == 1)


def test_thompson_uses_scaled_independent_gaussian_scores():
    key = jax.random.key(41)
    q = jnp.array([-1.0, 2.0, 0.5])
    u = jnp.array([4.0, 0.25, 1.0])
    scores = q + 1.5 * jnp.sqrt(u) * jax.random.normal(key, q.shape)
    assert int(gaussian_thompson_action(key, q, u, 1.5)) == int(jnp.argmax(scores))


def test_propagated_uncertainty_terminal_and_nonterminal():
    update = jax.jit(propagated_uncertainty)
    assert np.isclose(float(count_uncertainty(3, 4.0)), 1.0)
    assert np.isclose(float(update(3, 2.0, False, gamma=0.9)), 0.95 + 0.05 * (1 + 0.81 * 2))
    assert np.isclose(float(update(3, 2.0, True, gamma=0.9)), 1.0)
