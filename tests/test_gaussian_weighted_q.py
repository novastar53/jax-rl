import jax
import jax.numpy as jnp

from jax_rl.td.gaussian_weighted_q import (
    init_ess,
    update_ess,
    effective_sample_size,
    gaussian_weighted_max,
    init_gaussian_weighted_q,
    update_gaussian_weighted_q,
    update_weighted_variance 
)


def test_equal_weights():
    state = init_ess(num_states=2, num_actions=3)

    for n in range(1, 11):
        state = update_ess(state, 0, 1, alpha=1 / n)

    ess = effective_sample_size(state)

    assert jnp.isclose(ess[0, 1], 10.0)
    assert jnp.all(ess[1] == 0)


def test_unequal_weights():
    state = init_ess(num_states=1, num_actions=2)

    for alpha in [1.0, 0.5, 0.5]:
        state = update_ess(state, 0, 0, alpha)

    ess = effective_sample_size(state)

    assert jnp.isclose(ess[0, 0], 8 / 3)
    assert ess[0, 1] == 0


def test_jit():
    state = init_ess(num_states=2, num_actions=3)

    update = jax.jit(update_ess)
    state = update(state, 0, 1, 0.5)

    ess = effective_sample_size(state)

    assert jnp.isclose(ess[0, 1], 1.0)



import jax
import jax.numpy as jnp

from jax_rl.td.gaussian_weighted_q import (
    init_ess,
    update_ess,
    effective_sample_size,
    init_weighted_variance,
    update_weighted_variance,
    weighted_sample_variance,
)


def update_statistics(variance_state, ess_state, s, a, alpha, target):
    variance_state = update_weighted_variance(
        variance_state, ess_state, s, a, alpha, target
    )
    ess_state = update_ess(ess_state, s, a, alpha)

    return variance_state, ess_state


def test_equal_weights():
    variance_state = init_weighted_variance(1, 1)
    ess_state = init_ess(1, 1)

    targets = [10.0, 20.0, 15.0]

    for n, target in enumerate(targets, start=1):
        variance_state, ess_state = update_statistics(
            variance_state, ess_state, 0, 0, 1 / n, target
        )

    variance = weighted_sample_variance(variance_state, ess_state)
    ess = effective_sample_size(ess_state)

    assert jnp.isclose(variance_state.mean[0, 0], 15.0)
    assert jnp.isclose(variance[0, 0], 25.0)
    assert jnp.isclose(ess[0, 0], 3.0)


def test_unequal_weights():
    variance_state = init_weighted_variance(1, 1)
    ess_state = init_ess(1, 1)

    targets = [10.0, 20.0, 15.0]
    alphas = [1.0, 0.5, 0.5]

    for alpha, target in zip(alphas, targets):
        variance_state, ess_state = update_statistics(
            variance_state, ess_state, 0, 0, alpha, target
        )

    variance = weighted_sample_variance(variance_state, ess_state)
    ess = effective_sample_size(ess_state)

    assert jnp.isclose(variance_state.mean[0, 0], 15.0)
    assert jnp.isclose(variance[0, 0], 20.0)
    assert jnp.isclose(ess[0, 0], 8 / 3)


def test_single_observation():
    variance_state = init_weighted_variance(1, 1)
    ess_state = init_ess(1, 1)

    variance_state, ess_state = update_statistics(
        variance_state, ess_state, 0, 0, 1.0, 10.0
    )

    variance = weighted_sample_variance(variance_state, ess_state)

    assert jnp.isclose(variance_state.mean[0, 0], 10.0)
    assert variance[0, 0] == 0.0
    assert jnp.isfinite(variance[0, 0])


def test_independent_state_action_pairs():
    variance_state = init_weighted_variance(2, 3)
    ess_state = init_ess(2, 3)

    variance_state, ess_state = update_statistics(
        variance_state, ess_state, 0, 1, 1.0, 10.0
    )

    assert variance_state.mean[0, 1] == 10.0
    assert jnp.count_nonzero(variance_state.mean) == 1
    assert jnp.count_nonzero(ess_state.W) == 1


def test_jit():
    variance_state = init_weighted_variance(1, 1)
    ess_state = init_ess(1, 1)

    update = jax.jit(update_statistics)

    variance_state, ess_state = update(
        variance_state, ess_state, 0, 0, 1.0, 10.0
    )
    variance_state, ess_state = update(
        variance_state, ess_state, 0, 0, 0.5, 20.0
    )

    variance = weighted_sample_variance(variance_state, ess_state)

    assert jnp.isclose(variance_state.mean[0, 0], 15.0)
    assert jnp.isclose(variance[0, 0], 50.0)



def test_zero_uncertainty():
    q = jnp.array([1.0, 3.0, 2.0])
    se = jnp.zeros(3)

    value, probs = gaussian_weighted_max(
        jax.random.key(0), q, se
    )

    assert jnp.isclose(value, 3.0)
    assert jnp.allclose(probs, jnp.array([0.0, 1.0, 0.0]))


def test_equal_maxima():
    q = jnp.array([2.0, 2.0, 1.0])
    se = jnp.zeros(3)

    value, probs = gaussian_weighted_max(
        jax.random.key(0), q, se
    )

    assert jnp.isclose(value, 2.0)
    assert jnp.allclose(probs, jnp.array([0.5, 0.5, 0.0]))


def test_gaussian_probabilities():
    q = jnp.array([10.0, 8.0])
    se = jnp.array([4.0, 4.0])

    value, probs = gaussian_weighted_max(
        jax.random.key(42),
        q,
        se,
        num_samples=20_000,
    )

    expected_prob = jax.scipy.special.ndtr(
        (10.0 - 8.0) / jnp.sqrt(4.0**2 + 4.0**2)
    )

    expected_value = (
        expected_prob * 10.0
        + (1 - expected_prob) * 8.0
    )

    assert jnp.isclose(probs.sum(), 1.0)
    assert jnp.isclose(probs[0], expected_prob, atol=0.02)
    assert jnp.isclose(value, expected_value, atol=0.04)


def test_gaussian_weighted_max_jit():
    q = jnp.array([1.0, 2.0, 3.0])
    se = jnp.array([0.5, 0.5, 0.5])

    compute = jax.jit(gaussian_weighted_max, static_argnames=("num_samples",))

    value, probs = compute(
        jax.random.key(0),
        q,
        se,
        num_samples=100,
    )

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(probs))
    assert jnp.isclose(probs.sum(), 1.0)
    assert jnp.all(probs >= 0.0)
    assert q.min() <= value <= q.max()



def test_first_q_update():
    state = init_gaussian_weighted_q(2, 3)

    state = update_gaussian_weighted_q(
        state,
        key=jax.random.key(0),
        s=0,
        a=1,
        reward=5.0,
        next_s=1,
        terminated=True,
    )

    assert jnp.isclose(state.Q[0, 1], 5.0)
    assert state.N[0, 1] == 1
    assert jnp.isclose(state.ess.W[0, 1], 1.0)
    assert jnp.isclose(state.ess.H[0, 1], 1.0)


def test_nonterminal_bootstrap():
    state = init_gaussian_weighted_q(2, 2)

    # Give the next state two known action values.
    Q = state.Q.at[1].set(jnp.array([2.0, 4.0]))
    N = state.N.at[1].set(jnp.array([2, 2]))

    state = state._replace(Q=Q, N=N)

    state = update_gaussian_weighted_q(
        state,
        key=jax.random.key(0),
        s=0,
        a=0,
        reward=1.0,
        next_s=1,
        terminated=False,
        gamma=0.9,
    )

    # Zero estimated variance makes the weighted max exact.
    expected_target = 1.0 + 0.9 * 4.0

    assert jnp.isclose(state.Q[0, 0], expected_target)


def test_terminal_transition():
    state = init_gaussian_weighted_q(2, 2)

    Q = state.Q.at[1].set(jnp.array([100.0, 200.0]))
    state = state._replace(Q=Q)

    state = update_gaussian_weighted_q(
        state,
        key=jax.random.key(0),
        s=0,
        a=0,
        reward=-5.0,
        next_s=1,
        terminated=True,
        gamma=0.9,
    )

    assert jnp.isclose(state.Q[0, 0], -5.0)


def test_learning_rate_decay():
    state = init_gaussian_weighted_q(2, 2)

    state = update_gaussian_weighted_q(
        state,
        jax.random.key(0),
        0, 0, 10.0, 1, True,
    )

    state = update_gaussian_weighted_q(
        state,
        jax.random.key(1),
        0, 0, 20.0, 1, True,
    )

    alpha = 1.0 / (2.0 ** 0.8)
    expected = 10.0 + alpha * (20.0 - 10.0)

    assert state.N[0, 0] == 2
    assert jnp.isclose(state.Q[0, 0], expected)


def test_uncertainty_statistics():
    state = init_gaussian_weighted_q(2, 2)

    for i, reward in enumerate([10.0, 20.0]):
        state = update_gaussian_weighted_q(
            state,
            jax.random.key(i),
            0, 0, reward, 1, True,
        )

    variance = weighted_sample_variance(
        state.variance, state.ess
    )
    ess = effective_sample_size(state.ess)

    assert jnp.isclose(
        state.variance.mean[0, 0],
        state.Q[0, 0],
    )
    assert jnp.isclose(variance[0, 0], 50.0)
    assert 1.0 < ess[0, 0] <= 2.0


def test_complete_update_jit():
    state = init_gaussian_weighted_q(2, 2)

    update = jax.jit(
        update_gaussian_weighted_q,
        static_argnames=("num_samples",),
    )

    state = update(
        state,
        jax.random.key(0),
        0, 1, 5.0, 1, False,
        num_samples=100,
    )

    assert jnp.all(jnp.isfinite(state.Q))
    assert jnp.isclose(state.Q[0, 1], 5.0)
    assert state.N[0, 1] == 1
