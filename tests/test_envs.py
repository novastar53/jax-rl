
import jax
import jax.numpy as jnp
import pytest

from jax_rl.envs.gridworld import Action, Gridworld
from jax_rl.envs.cliff_walking import CliffWalking
from jax_rl.envs.roulette import Roulette
from jax_rl.envs.stochastic_gridworld import StochasticGridworld
from jax_rl.envs.stochastic_cliff_walking import StochasticCliffWalking
from jax_rl.envs.frozen_lake import FrozenLake



@pytest.mark.parametrize(
    "action, expected_state",
    [
        (Action.UP, 1),
        (Action.RIGHT, 6),
        (Action.DOWN, 9),
        (Action.LEFT, 4),
    ],
)
def test_gridworld_movement(action, expected_state):
    env = Gridworld(size=4)
    key = jax.random.key(42)

    new_state, observation, reward, terminated, info = env.step(
        key, jnp.array(5), action
    )

    assert int(new_state) == expected_state
    assert int(observation) == expected_state
    assert float(reward) == -1.0
    assert not bool(terminated)


@pytest.mark.parametrize(
    "state, action",
    [
        (1, Action.UP),      # Top boundary
        (4, Action.LEFT),    # Left boundary
        (11, Action.RIGHT),  # Right boundary
        (14, Action.DOWN),   # Bottom boundary
    ],
)
def test_gridworld_boundaries(state, action):
    env = Gridworld(size=4)
    key = jax.random.key(42)

    new_state, _, reward, terminated, _ = env.step(
        key, jnp.array(state), action
    )

    assert int(new_state) == state
    assert float(reward) == -1.0
    assert not bool(terminated)


@pytest.mark.parametrize(
    "state, action, expected_state",
    [
        (1, Action.LEFT, 0),
        (14, Action.RIGHT, 15),
    ],
)
def test_gridworld_enter_terminal(state, action, expected_state):
    env = Gridworld(size=4)
    key = jax.random.key(42)

    new_state, _, reward, terminated, _ = env.step(
        key, jnp.array(state), action
    )

    assert int(new_state) == expected_state
    assert float(reward) == -1.0
    assert bool(terminated)


@pytest.mark.parametrize("terminal_state", [0, 15])
def test_gridworld_terminal_absorbing(terminal_state):
    env = Gridworld(size=4)
    key = jax.random.key(42)

    for action in Action:
        new_state, _, reward, terminated, _ = env.step(
            key, jnp.array(terminal_state), action
        )

        assert int(new_state) == terminal_state
        assert float(reward) == 0.0
        assert bool(terminated)


def test_gridworld_jit():
    env = Gridworld(size=4)
    key = jax.random.key(42)

    jitted_step = jax.jit(env.step)

    new_state, _, reward, terminated, _ = jitted_step(
        key,
        jnp.array(5),
        jnp.array(Action.RIGHT),
    )

    assert int(new_state) == 6
    assert float(reward) == -1.0
    assert not bool(terminated)


from jax_rl.envs.bernoulli_bandit import BernoulliBandit


def test_bernoulli_bandit_reset():
    env = BernoulliBandit(k=10)
    key = jax.random.key(42)

    env_state, observation = env.reset(key)

    assert env_state.shape == (10,)
    assert jnp.all(env_state >= 0.0)
    assert jnp.all(env_state <= 1.0)
    assert observation == 0


def test_bernoulli_bandit_step():
    env = BernoulliBandit(k=10)
    key = jax.random.key(42)

    key, reset_key, step_key = jax.random.split(key, 3)

    env_state, _ = env.reset(reset_key)

    new_state, observation, reward, terminated, info = env.step(
        step_key, env_state, action=3
    )

    # Reward must be a scalar Bernoulli outcome
    assert reward.shape == ()
    assert reward in (0, 1)

    # Environment remains stationary
    assert jnp.array_equal(new_state, env_state)

    assert observation == 0
    assert not terminated


@pytest.mark.parametrize(
    "action, expected_reward",
    [
        (0, 0),
        (1, 1),
    ],
)
def test_bernoulli_bandit_deterministic_rewards(
    action, expected_reward
):
    env = BernoulliBandit(k=2)
    key = jax.random.key(42)

    # Arm 0 always fails; arm 1 always succeeds
    env_state = jnp.array([0.0, 1.0])

    _, _, reward, _, _ = env.step(
        key, env_state, action
    )

    assert int(reward) == expected_reward


def test_bernoulli_bandit_jit():
    env = BernoulliBandit(k=2)

    env_state = jnp.array([0.0, 1.0])
    key = jax.random.key(42)

    jitted_step = jax.jit(env.step)

    new_state, _, reward, terminated, _ = jitted_step(
        key,
        env_state,
        jnp.array(1),
    )

    assert int(reward) == 1
    assert jnp.array_equal(new_state, env_state)
    assert not bool(terminated)


from jax_rl.envs.contextual_bandit import (
    ContextualBandit,
    ContextualBanditState,
)


@pytest.fixture
def env():
    return ContextualBandit(num_contexts=3, k=4)


def test_reset(env):
    key = jax.random.key(42)
    env_state, observation = env.reset(key)

    assert isinstance(env_state, ContextualBanditState)
    assert env_state.q_star.shape == (3, 4)

    assert observation.shape == ()
    assert 0 <= int(observation) < 3
    assert jnp.array_equal(observation, env_state.context)


def test_reward_generation(env):
    key = jax.random.key(42)
    env_state, _ = env.reset(key)

    action = 2
    step_key = jax.random.key(123)

    reward_key, _ = jax.random.split(step_key)

    expected_reward = (
        env_state.q_star[env_state.context, action]
        + jax.random.normal(reward_key)
    )

    _, _, reward, _, _ = env.step(
        step_key, env_state, action
    )

    assert jnp.allclose(reward, expected_reward)


def test_state_transition(env):
    key = jax.random.key(42)
    env_state, _ = env.reset(key)

    step_key = jax.random.key(123)
    _, context_key = jax.random.split(step_key)

    expected_context = jax.random.choice(
        context_key,
        a=jnp.arange(env.num_contexts),
    )

    new_env_state, observation, reward, terminated, info = (
        env.step(step_key, env_state, action=1)
    )

    # True action values remain stationary.
    assert jnp.array_equal(
        new_env_state.q_star,
        env_state.q_star,
    )

    # The next context is sampled using the supplied key.
    assert jnp.array_equal(
        new_env_state.context,
        expected_context,
    )

    assert jnp.array_equal(observation, expected_context)

    assert reward.shape == ()
    assert terminated is False
    assert info == {}


def test_jit_compatibility(env):
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    key = jax.random.key(42)
    env_state, observation = reset_fn(key)

    assert env_state.q_star.shape == (3, 4)
    assert observation.shape == ()

    new_env_state, next_observation, reward, terminated, info = (
        step_fn(
            jax.random.key(123),
            env_state,
            jnp.array(2),
        )
    )

    assert new_env_state.q_star.shape == (3, 4)
    assert next_observation.shape == ()
    assert reward.shape == ()
    assert not bool(terminated)
    assert info == {}


from jax_rl.envs.cliff_walking import CliffWalking
from jax_rl.envs.gridworld import Action


def test_cliff_reset():
    env = CliffWalking()

    state, observation = env.reset(jax.random.key(42))

    assert int(state) == 36
    assert int(observation) == 36


def test_cliff_fall():
    env = CliffWalking()
    key = jax.random.key(42)

    next_state, observation, reward, terminated, _ = env.step(
        key,
        jnp.int32(36),
        Action.RIGHT,
    )

    assert int(next_state) == 36
    assert int(observation) == 36
    assert float(reward) == -100.0
    assert not bool(terminated)


def test_cliff_safe_move():
    env = CliffWalking()

    next_state, _, reward, terminated, _ = env.step(
        jax.random.key(42),
        jnp.int32(36),
        Action.UP,
    )

    assert int(next_state) == 24
    assert float(reward) == -1.0
    assert not bool(terminated)


def test_cliff_goal():
    env = CliffWalking()

    next_state, _, reward, terminated, _ = env.step(
        jax.random.key(42),
        jnp.int32(35),
        Action.DOWN,
    )

    assert int(next_state) == 47
    assert float(reward) == -1.0
    assert bool(terminated)


def test_cliff_boundaries():
    env = CliffWalking()

    next_state, _, reward, terminated, _ = env.step(
        jax.random.key(42),
        jnp.int32(0),
        Action.UP,
    )

    assert int(next_state) == 0
    assert float(reward) == -1.0
    assert not bool(terminated)


def test_cliff_goal_is_absorbing():
    env = CliffWalking()

    next_state, _, reward, terminated, _ = env.step(
        jax.random.key(42),
        jnp.int32(47),
        Action.LEFT,
    )

    assert int(next_state) == 47
    assert float(reward) == 0.0
    assert bool(terminated)


def test_cliff_step_jit():
    env = CliffWalking()

    compiled_step = jax.jit(env.step)

    result = compiled_step(
        jax.random.key(42),
        jnp.int32(36),
        jnp.int32(Action.RIGHT),
    )

    next_state, _, reward, terminated, _ = result

    assert int(next_state) == 36
    assert float(reward) == -100.0
    assert not bool(terminated)


from jax_rl.envs.maximization_bias import MaximizationBias


def test_maximization_bias_left():
    env = MaximizationBias()
    key = jax.random.key(42)

    state, observation = env.reset(key)

    state, observation, reward, terminated, _ = env.step(
        key, state, jnp.int32(0)
    )

    assert int(observation) == 1
    assert float(reward) == 0.0
    assert not bool(terminated)


def test_maximization_bias_right():
    env = MaximizationBias()
    key = jax.random.key(42)

    state, _ = env.reset(key)

    state, observation, reward, terminated, _ = env.step(
        key, state, jnp.int32(1)
    )

    assert int(observation) == 2
    assert float(reward) == 0.0
    assert bool(terminated)


def test_maximization_bias_reward_distribution():
    env = MaximizationBias()

    keys = jax.random.split(
        jax.random.key(42), 10_000
    )

    def sample(key):
        _, _, reward, _, _ = env.step(
            key, jnp.int32(1), jnp.int32(0)
        )
        return reward

    rewards = jax.vmap(sample)(keys)

    assert jnp.isclose(jnp.mean(rewards), -0.1, atol=0.05)
    assert jnp.isclose(jnp.std(rewards), 1.0, atol=0.05)


def test_maximization_bias_action_masks():
    env = MaximizationBias()

    assert int(jnp.sum(env.action_mask(0))) == 2
    assert int(jnp.sum(env.action_mask(1))) == 10



def test_roulette_shape_and_expected_values():
    env = Roulette()
    assert env.num_actions == 171
    assert env.num_states == 2
    assert len(env.bet_labels) == 170
    assert env.stop_action == 170
    assert jnp.isclose(float(env.expected_rewards[0]), -2 / 38)
    idx = env.bet_labels.index("five-number top line")
    assert jnp.isclose(float(env.expected_rewards[idx]), -3 / 38)
    assert jnp.all(jnp.asarray(env.expected_rewards) < 0)


def test_roulette_terminal_action():
    env = Roulette()
    state, obs = env.reset(jax.random.PRNGKey(0))
    state, obs, reward, done, _ = env.step(jax.random.PRNGKey(1), state, env.stop_action)
    assert int(state) == 1 and int(obs) == 1
    assert bool(done) and float(reward) == 0
    _, _, reward2, done2, _ = env.step(jax.random.PRNGKey(2), state, 0)
    assert bool(done2) and float(reward2) == 0


def test_roulette_all_rewards_shape_and_jit():
    env = Roulette()
    fn = jax.jit(env.sample_all_rewards)
    samples = fn(jax.random.PRNGKey(0), jnp.int32(0))
    assert samples.shape == (171,)
    assert float(samples[-1]) == 0
    assert jnp.all(jnp.asarray(fn(jax.random.PRNGKey(0), jnp.int32(1))) == 0)


def test_roulette_shared_pocket_same_sample_as_step():
    env = Roulette()
    key = jax.random.PRNGKey(5)
    all_rewards = env.sample_all_rewards(key, jnp.int32(0), shared_spin=True)
    for action in (0, 20, 100, 169):
        assert jnp.isclose(
            float(all_rewards[action]),
            float(env.step(key, jnp.int32(0), action)[2]),
        )


def test_transition_models_are_stochastic():
    for env in (
        Roulette(), StochasticGridworld(), StochasticCliffWalking(),
        FrozenLake(), FrozenLake(map_name="8x8"),
    ):
        P, R = env.transition_model()
        assert P.shape == R.shape == (env.num_states, env.num_actions, env.num_states)
        assert jnp.allclose(jnp.asarray(P.sum(axis=-1)), 1.0, atol=1e-6)
        assert jnp.all(jnp.asarray(P) >= -1e-7)
        assert jnp.all(jnp.isfinite(jnp.asarray(R)))


@pytest.mark.parametrize("action", [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT])
def test_stochastic_gridworld_reduces_to_gridworld(action):
    original = Gridworld()
    noisy = StochasticGridworld(slip_prob=0, reward_std=0)
    for state in (0, 5, 10, 15):
        got = noisy.step(jax.random.PRNGKey(0), jnp.int32(state), action)
        expected = original.step(jax.random.PRNGKey(0), jnp.int32(state), action)
        assert all(bool(jnp.array_equal(a, b)) for a, b in zip(got[:4], expected[:4]))


def test_stochastic_gridworld_slip_distribution():
    env = StochasticGridworld(slip_prob=0.2, reward_std=0)
    P, R = env.transition_model()
    # From center cell (row 1, col 1), UP goes to state 1, side turns to 4 / 6.
    assert jnp.isclose(float(P[5, Action.UP, 1]), .8)
    assert jnp.isclose(float(P[5, Action.UP, 4]), .1)
    assert jnp.isclose(float(P[5, Action.UP, 6]), .1)
    assert jnp.isclose(float(R[5, Action.UP, 1]), -1.)
    assert jnp.isclose(float(P[0, Action.UP, 0]), 1.)
    assert jnp.isclose(float(R[0, Action.UP, 0]), 0.)


def test_stochastic_cliff_reduces_to_cliff_walking():
    original = CliffWalking()
    noisy = StochasticCliffWalking(slip_prob=0, reward_std=0)
    for state, action in [(36, Action.RIGHT), (24, Action.DOWN), (47, Action.LEFT)]:
        got = noisy.step(jax.random.PRNGKey(0), jnp.int32(state), action)
        expected = original.step(jax.random.PRNGKey(0), jnp.int32(state), action)
        assert all(bool(jnp.array_equal(a, b)) for a, b in zip(got[:4], expected[:4]))


def test_stochastic_cliff_falling_penalty_in_model():
    env = StochasticCliffWalking(slip_prob=0.2, reward_std=0)
    P, R = env.transition_model()
    # From start, RIGHT falls into cliff and resets, LEFT stays at start,
    # DOWN clips to start.
    assert jnp.isclose(float(P[36, Action.RIGHT, 36]), .9)
    assert jnp.isclose(float(P[36, Action.RIGHT, 24]), .1)
    assert jnp.isclose(float(R[36, Action.RIGHT, 36]), (-100 * .8 - 1 * .1) / .9)


def test_frozen_lake_deterministic_and_absorbing():
    env = FrozenLake(is_slippery=False)
    state, _ = env.reset(jax.random.PRNGKey(0))
    assert int(state) == 0
    state, _, reward, done, _ = env.step(jax.random.PRNGKey(1), state, Action.RIGHT)
    assert int(state) == 1 and float(reward) == 0 and not bool(done)
    state, _, reward, done, _ = env.step(jax.random.PRNGKey(2), state, Action.DOWN)
    assert int(state) == 5 and float(reward) == 0 and bool(done)
    next_state, _, reward, done, _ = env.step(jax.random.PRNGKey(3), state, Action.RIGHT)
    assert int(next_state) == 5 and float(reward) == 0 and bool(done)


def test_frozen_lake_slippery_start_transition():
    env = FrozenLake()
    P, R = env.transition_model()
    # UP: UP and LEFT hit the boundary, RIGHT moves to state 1.
    assert jnp.isclose(float(P[0, Action.UP, 0]), 2/3)
    assert jnp.isclose(float(P[0, Action.UP, 1]), 1/3)
    assert jnp.isclose(float(R[0, Action.UP, 0]), 0)


def test_frozen_lake_goal_reward():
    env = FrozenLake(is_slippery=False)
    state, _, reward, done, _ = env.step(jax.random.PRNGKey(0), jnp.int32(14), Action.RIGHT)
    assert int(state) == 15 and float(reward) == 1. and bool(done)
    state, _, reward, done, _ = env.step(jax.random.PRNGKey(0), state, Action.LEFT)
    assert int(state) == 15 and float(reward) == 0 and bool(done)


def test_jit_vmap_of_step():
    key = jax.random.PRNGKey(0)
    for env, state in (
        (Roulette(), 0), (StochasticGridworld(), 5),
        (StochasticCliffWalking(), 36), (FrozenLake(), 0),
    ):
        fn = jax.jit(env.step)
        result = fn(key, jnp.int32(state), jnp.int32(0))
        assert len(result) == 5
        batched = jax.jit(jax.vmap(env.step))(
            jax.random.split(key, 4),
            jnp.full((4,), state, dtype=jnp.int32),
            jnp.zeros((4,), dtype=jnp.int32),
        )
        assert batched[2].shape == (4,)
