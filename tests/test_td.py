from jax_rl.envs.gridworld import Gridworld
import pytest
import jax
import jax.numpy as jnp

from jax_rl.td.td_prediction import td0_update


def test_td0_nonterminal_update():
    V = jnp.array([0.0, 2.0, 4.0])

    result = td0_update(
        V,
        state=0,
        reward=1.0,
        next_state=1,
        terminated=False,
        alpha=0.5,
        gamma=0.9,
    )

    # Target = 1 + 0.9 * 2 = 2.8
    # New V[0] = 0 + 0.5 * 2.8 = 1.4
    assert jnp.allclose(result, jnp.array([1.4, 2.0, 4.0]))


def test_td0_terminal_update():
    V = jnp.array([5.0, 10.0, 20.0])

    result = td0_update(
        V,
        state=0,
        reward=-1.0,
        next_state=2,
        terminated=True,
        alpha=0.1,
        gamma=1.0,
    )

    # Terminal: target = -1, no bootstrap
    # New V[0] = 5 + 0.1 * (-1 - 5) = 4.4
    assert jnp.allclose(result, jnp.array([4.4, 10.0, 20.0]))


def test_td0_zero_learning_rate():
    V = jnp.array([1.0, 2.0, 3.0])

    result = td0_update(
        V,
        state=0,
        reward=10.0,
        next_state=1,
        terminated=False,
        alpha=0.0,
    )

    assert jnp.array_equal(result, V)


from jax_rl.td.td_prediction import td0_prediction



class JaxFixedEpisodeEnv:
    """
    Deterministic two-transition environment:

        S0 --(+2)--> S1 --(+4)--> Terminal
    """

    def reset(self, key):
        state = jnp.array(0, dtype=jnp.int32)
        return state, state

    def step(self, key, env_state, action):
        next_state = env_state + 1

        reward = jnp.where(
            env_state == 0,
            2.0,
            4.0,
        )

        terminated = next_state == 2

        return (
            next_state,
            next_state,
            reward,
            terminated,
            {},
        )


class FixedEpisodeEnv:
    """
    Deterministic environment with two transitions:

        S0 --(+2)--> S1 --(+4)--> Terminal

    The environment state is a step counter.
    Observations are state indices.
    """

    def reset(self, key):
        return 0, 0

    def step(self, key, env_state, action):
        if env_state == 0:
            return 1, 1, 2.0, False, {}

        if env_state == 1:
            return 2, 2, 4.0, True, {}

        raise RuntimeError("Cannot step after termination")


def test_td0_prediction_single_episode():
    env = FixedEpisodeEnv()
    policy = jnp.ones((3, 1))

    V = td0_prediction(
        env,
        policy,
        jax.random.key(42),
        num_episodes=1,
        alpha=0.5,
        gamma=1.0,
    )

    # Initial V = [0, 0, 0]
    #
    # Transition 1:
    #   S0 -> S1, reward = 2
    #   target = 2 + V[S1] = 2
    #   V[S0] = 0 + 0.5 * 2 = 1
    #
    # Transition 2:
    #   S1 -> terminal, reward = 4
    #   target = 4
    #   V[S1] = 0 + 0.5 * 4 = 2

    expected = jnp.array([1.0, 2.0, 0.0])
    assert jnp.allclose(V, expected)


def test_td0_prediction_bootstraps_across_episodes():
    env = FixedEpisodeEnv()
    policy = jnp.ones((3, 1))

    V = td0_prediction(
        env,
        policy,
        jax.random.key(42),
        num_episodes=2,
        alpha=0.5,
        gamma=1.0,
    )

    # After episode 1: V = [1, 2, 0]
    #
    # Episode 2, transition 1:
    #   target = 2 + V[S1] = 4
    #   V[S0] = 1 + 0.5 * (4 - 1) = 2.5
    #
    # Episode 2, transition 2:
    #   target = 4
    #   V[S1] = 2 + 0.5 * (4 - 2) = 3

    expected = jnp.array([2.5, 3.0, 0.0])
    assert jnp.allclose(V, expected)


def test_td0_prediction_discounting():
    env = FixedEpisodeEnv()
    policy = jnp.ones((3, 1))

    V = td0_prediction(
        env,
        policy,
        jax.random.key(42),
        num_episodes=2,
        alpha=0.5,
        gamma=0.5,
    )

    # Episode 1: V = [1, 2, 0]
    #
    # Episode 2:
    #   target(S0) = 2 + 0.5 * 2 = 3
    #   V[S0] = 1 + 0.5 * (3 - 1) = 2
    #   V[S1] = 2 + 0.5 * (4 - 2) = 3

    expected = jnp.array([2.0, 3.0, 0.0])
    assert jnp.allclose(V, expected)


def test_td0_prediction_zero_episodes():
    env = FixedEpisodeEnv()
    policy = jnp.ones((3, 1))

    V = td0_prediction(
        env,
        policy,
        jax.random.key(42),
        num_episodes=0,
    )

    assert jnp.array_equal(V, jnp.zeros(3))


def test_td0_prediction_max_steps():
    env = FixedEpisodeEnv()
    policy = jnp.ones((3, 1))

    # The environment needs exactly two transitions
    # to terminate. One step is insufficient.
    with pytest.raises(RuntimeError, match="did not terminate"):
        td0_prediction(
            env,
            policy,
            jax.random.key(42),
            num_episodes=1,
            max_steps=1,
        )


def test_td0_prediction_exact_max_steps():
    env = FixedEpisodeEnv()
    policy = jnp.ones((3, 1))

    # Termination on the final permitted step is valid.
    V = td0_prediction(
        env,
        policy,
        jax.random.key(42),
        num_episodes=1,
        alpha=0.5,
        max_steps=2,
    )

    assert jnp.allclose(V, jnp.array([1.0, 2.0, 0.0]))



from jax_rl.td.td_prediction import (
    td0_prediction,
    td0_prediction_jit,
)


def test_td0_jit_matches_reference():
    env = JaxFixedEpisodeEnv()
    policy = jnp.ones((3, 1))
    key = jax.random.key(42)

    reference = td0_prediction(
        env,
        policy,
        key,
        num_episodes=10,
        alpha=0.1,
        gamma=0.9,
    )

    compiled = td0_prediction_jit(
        env,
        policy,
        key,
        num_episodes=10,
        alpha=0.1,
        gamma=0.9,
    )

    assert jnp.allclose(reference, compiled, atol=1e-6)


def test_td0_jit_max_steps():
    env = JaxFixedEpisodeEnv()
    policy = jnp.ones((3, 1))

    with pytest.raises(RuntimeError, match="did not terminate"):
        td0_prediction_jit(
            env,
            policy,
            jax.random.key(42),
            num_episodes=1,
            max_steps=1,
        )


def test_td0_jit_zero_episodes():
    env = JaxFixedEpisodeEnv()
    policy = jnp.ones((3, 1))

    result = td0_prediction_jit(
        env,
        policy,
        jax.random.key(42),
        num_episodes=0,
    )

    assert jnp.array_equal(result, jnp.zeros(3))


def test_td0_jit_gridworld_matches_reference():
    env = Gridworld(size=4)
    policy = jnp.full((16, 4), 0.25)
    key = jax.random.key(42)

    reference = td0_prediction(
        env,
        policy,
        key,
        num_episodes=100,
        alpha=0.1,
        gamma=1.0,
    )

    compiled = td0_prediction_jit(
        env,
        policy,
        key,
        num_episodes=100,
        alpha=0.1,
        gamma=1.0,
    )

    assert jnp.allclose(reference, compiled, atol=1e-5)



class SarsaTestEnv:
    """
    Deterministic two-step environment:

        S0 --(+2)--> S1 --(+4)--> Terminal

    The environment records the actions executed.
    """

    num_states = 3
    num_actions = 2

    def __init__(self):
        self.actions_taken = []

    def reset(self, key):
        return 0, 0

    def step(self, key, env_state, action):
        self.actions_taken.append(int(action))

        if env_state == 0:
            return 1, 1, 2.0, False, {}

        if env_state == 1:
            return 2, 2, 4.0, True, {}

        raise RuntimeError("Cannot step after termination")


class FixedActionPolicy:
    """Always selects a specified action."""

    def __init__(self, action):
        self.action = action

    def select_action(self, key, q_values):
        return jnp.int32(self.action)

    def action_probs(self, q_values):
        return jax.nn.one_hot(
            self.action,
            q_values.shape[0],
        )

class AlternatingActionPolicy:
    """
    Selects action 0 in state 0 and action 1 in state 1.

    In these tests, state 0 has equal Q-values initially,
    while state 1 has different Q-values after learning.
    """

    def __init__(self):
        self.calls = 0

    def select_action(self, key, q_values):
        action = self.calls % 2
        self.calls += 1
        return jnp.int32(action)

    def action_probs(self, q_values):
        return jnp.full(
        q_values.shape,
        1.0 / q_values.shape[0],
        )


from jax_rl.td.sarsa import sarsa


def test_sarsa_terminal_update():
    """A terminal transition must not bootstrap."""
    env = SarsaTestEnv()
    policy = FixedActionPolicy(action=0)

    Q = sarsa(
        env,
        policy,
        jax.random.key(42),
        num_episodes=1,
        alpha=0.5,
        gamma=1.0,
    )

    # S0 -> S1: Q[0,0] = 0.5 * 2 = 1
    # S1 -> terminal: Q[1,0] = 0.5 * 4 = 2
    expected = jnp.array([
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 0.0],
    ])

    assert jnp.allclose(Q, expected)


def test_sarsa_bootstraps_across_episodes():
    """Previously learned next-state values affect updates."""
    env = SarsaTestEnv()
    policy = FixedActionPolicy(action=0)

    Q = sarsa(
        env,
        policy,
        jax.random.key(42),
        num_episodes=2,
        alpha=0.5,
        gamma=1.0,
    )

    # After episode 1:
    # Q[0,0] = 1, Q[1,0] = 2
    #
    # Episode 2:
    # Q[0,0] = 1 + 0.5 * (2 + 2 - 1) = 2.5
    # Q[1,0] = 2 + 0.5 * (4 - 2) = 3
    expected = jnp.array([
        [2.5, 0.0],
        [3.0, 0.0],
        [0.0, 0.0],
    ])

    assert jnp.allclose(Q, expected)


def test_sarsa_discounting():
    """Gamma must discount the bootstrapped action value."""
    env = SarsaTestEnv()
    policy = FixedActionPolicy(action=0)

    Q = sarsa(
        env,
        policy,
        jax.random.key(42),
        num_episodes=2,
        alpha=0.5,
        gamma=0.5,
    )

    # Episode 1: Q[0,0] = 1, Q[1,0] = 2
    #
    # Episode 2:
    # Target at S0 = 2 + 0.5 * 2 = 3
    # Q[0,0] = 1 + 0.5 * (3 - 1) = 2
    # Q[1,0] = 3
    expected = jnp.array([
        [2.0, 0.0],
        [3.0, 0.0],
        [0.0, 0.0],
    ])

    assert jnp.allclose(Q, expected)


def test_sarsa_uses_next_selected_action():
    """The next action must be carried into the next step."""
    env = SarsaTestEnv()
    policy = AlternatingActionPolicy()

    Q = sarsa(
        env,
        policy,
        jax.random.key(42),
        num_episodes=1,
        alpha=0.5,
        gamma=1.0,
    )

    # First selection: action 0 at S0.
    # Second selection: action 1 at S1.
    assert env.actions_taken == [0, 1]
    assert policy.calls == 2

    expected = jnp.array([
        [1.0, 0.0],
        [0.0, 2.0],
        [0.0, 0.0],
    ])

    assert jnp.allclose(Q, expected)


def test_sarsa_zero_episodes():
    env = SarsaTestEnv()
    policy = FixedActionPolicy(action=0)

    Q = sarsa(
        env,
        policy,
        jax.random.key(42),
        num_episodes=0,
    )

    assert jnp.array_equal(Q, jnp.zeros((3, 2)))
    assert env.actions_taken == []


def test_sarsa_max_steps():
    env = SarsaTestEnv()
    policy = FixedActionPolicy(action=0)

    # The environment requires two transitions.
    with pytest.raises(RuntimeError, match="did not terminate"):
        sarsa(
            env,
            policy,
            jax.random.key(42),
            num_episodes=1,
            max_steps=1,
        )


def test_sarsa_exact_max_steps():
    env = SarsaTestEnv()
    policy = FixedActionPolicy(action=0)

    Q = sarsa(
        env,
        policy,
        jax.random.key(42),
        num_episodes=1,
        alpha=0.5,
        max_steps=2,
    )

    expected = jnp.array([
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 0.0],
    ])

    assert jnp.allclose(Q, expected)
    assert env.actions_taken == [0, 0]



class AlternatingPolicy:
    def __init__(self):
        self.calls = 0

    def select_action(self, key, q_values):
        action = self.calls % 2
        self.calls += 1
        return jnp.int32(action)


from jax_rl.td.q_learning import q_learning


def test_q_learning_terminal_update():
    env = SarsaTestEnv()
    policy = FixedActionPolicy(action=0)

    Q = q_learning(
        env,
        policy,
        jax.random.key(42),
        num_episodes=1,
        alpha=0.5,
        gamma=1.0,
    )

    # S0 -> S1, reward = 2
    # Q[0,0] = 0 + 0.5 * 2 = 1
    #
    # S1 -> terminal, reward = 4
    # Q[1,0] = 0 + 0.5 * 4 = 2

    expected = jnp.array([
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 0.0],
    ])

    assert jnp.allclose(Q, expected)


def test_q_learning_bootstraps_across_episodes():
    env = SarsaTestEnv()
    policy = FixedActionPolicy(action=0)

    Q = q_learning(
        env,
        policy,
        jax.random.key(42),
        num_episodes=2,
        alpha=0.5,
        gamma=1.0,
    )

    # After episode 1:
    # Q[0,0] = 1, Q[1,0] = 2
    #
    # Episode 2:
    # Target at S0 = 2 + max(Q[1]) = 4
    # Q[0,0] = 1 + 0.5 * (4 - 1) = 2.5
    # Q[1,0] = 2 + 0.5 * (4 - 2) = 3

    expected = jnp.array([
        [2.5, 0.0],
        [3.0, 0.0],
        [0.0, 0.0],
    ])

    assert jnp.allclose(Q, expected)


def test_q_learning_discounting():
    env = SarsaTestEnv()
    policy = FixedActionPolicy(action=0)

    Q = q_learning(
        env,
        policy,
        jax.random.key(42),
        num_episodes=2,
        alpha=0.5,
        gamma=0.5,
    )

    # Episode 1: Q[0,0] = 1, Q[1,0] = 2
    #
    # Episode 2:
    # Target at S0 = 2 + 0.5 * 2 = 3
    # Q[0,0] = 1 + 0.5 * (3 - 1) = 2
    # Q[1,0] = 3

    expected = jnp.array([
        [2.0, 0.0],
        [3.0, 0.0],
        [0.0, 0.0],
    ])

    assert jnp.allclose(Q, expected)


def test_q_learning_uses_greedy_target():
    """
    Q-learning must bootstrap from the maximum next-state
    action value, even when the behavior policy selects
    a different action.
    """
    env = SarsaTestEnv()
    policy = AlternatingPolicy()

    Q = q_learning(
        env,
        policy,
        jax.random.key(42),
        num_episodes=3,
        alpha=1.0,
        gamma=1.0,
    )

    # Episode 1: actions 0, 1
    # Q[0,0] = 2, Q[1,1] = 4
    #
    # Episode 2: actions 0, 1
    # Q[0,0] = 2 + max(Q[1]) = 6
    #
    # Episode 3: actions 0, 1
    # Q[0,0] remains 6

    expected = jnp.array([
        [6.0, 0.0],
        [0.0, 4.0],
        [0.0, 0.0],
    ])

    assert jnp.allclose(Q, expected)


def test_q_learning_selects_new_actions():
    env = SarsaTestEnv()
    policy = AlternatingPolicy()

    q_learning(
        env,
        policy,
        jax.random.key(42),
        num_episodes=1,
        alpha=0.5,
    )

    assert env.actions_taken == [0, 1]
    assert policy.calls == 2


def test_q_learning_zero_episodes():
    env = SarsaTestEnv()
    policy = FixedActionPolicy(action=0)

    Q = q_learning(
        env,
        policy,
        jax.random.key(42),
        num_episodes=0,
    )

    assert jnp.array_equal(
        Q,
        jnp.zeros((3, 2), dtype=jnp.float32),
    )
    assert env.actions_taken == []


def test_q_learning_max_steps():
    env = SarsaTestEnv()
    policy = FixedActionPolicy(action=0)

    with pytest.raises(RuntimeError, match="did not terminate"):
        q_learning(
            env,
            policy,
            jax.random.key(42),
            num_episodes=1,
            max_steps=1,
        )


def test_q_learning_exact_max_steps():
    env = SarsaTestEnv()
    policy = FixedActionPolicy(action=0)

    Q = q_learning(
        env,
        policy,
        jax.random.key(42),
        num_episodes=1,
        alpha=0.5,
        max_steps=2,
    )

    expected = jnp.array([
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 0.0],
    ])

    assert jnp.allclose(Q, expected)
    assert env.actions_taken == [0, 0]



def test_q_learning_off_policy_bootstrap():
    """
    The behavior policy takes action 0 at S1, but
    action 1 has the greater Q-value.

    Q-learning must use Q[S1, 1] in its target.
    """
    env = SarsaTestEnv()

    class NonGreedyPolicy:
        def __init__(self):
            self.calls = 0

        def select_action(self, key, q_values):
            # First episode: action 1 at both states.
            # Second episode: action 0 at both states.
            episode = self.calls // 2
            self.calls += 1
            return jnp.int32(1 if episode == 0 else 0)

    policy = NonGreedyPolicy()

    Q = q_learning(
        env,
        policy,
        jax.random.key(42),
        num_episodes=2,
        alpha=1.0,
        gamma=1.0,
    )

    # Episode 1:
    # Q[0,1] = 2
    # Q[1,1] = 4
    #
    # Episode 2:
    # Behavior selects action 0 at S0 and S1.
    # But at S1, max(Q[1]) = Q[1,1] = 4.
    #
    # Therefore:
    # Q[0,0] = 2 + 4 = 6
    # Q[1,0] = 4

    expected = jnp.array([
        [6.0, 2.0],
        [4.0, 4.0],
        [0.0, 0.0],
    ])

    assert jnp.allclose(Q, expected)
    assert env.actions_taken == [1, 1, 0, 0]


from jax_rl.envs.gridworld import Gridworld
from jax_rl.envs.cliff_walking import CliffWalking
from jax_rl.td.sarsa import sarsa
from jax_rl.td.q_learning import q_learning
from jax_rl.td.sarsa import sarsa_jit
from jax_rl.td.q_learning import q_learning_jit
from jax_rl.utils.policies import EpsilonGreedyPolicy


@pytest.mark.parametrize(
    "reference, compiled",
    [
        (sarsa, sarsa_jit),
        (q_learning, q_learning_jit),
    ],
)
def test_control_jit_matches_reference(reference, compiled):
    env = Gridworld(size=4)
    policy = EpsilonGreedyPolicy(epsilon=0.1)
    key = jax.random.key(42)

    Q_reference = reference(
        env,
        policy,
        key,
        num_episodes=20,
        alpha=0.1,
        gamma=1.0,
    )

    Q_compiled = compiled(
        env,
        policy,
        key,
        num_episodes=20,
        alpha=0.1,
        gamma=1.0,
    )

    assert jnp.allclose(
        Q_reference,
        Q_compiled,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "algorithm",
    [sarsa_jit, q_learning_jit],
)
def test_control_jit_zero_episodes(algorithm):
    env = Gridworld(size=4)
    policy = EpsilonGreedyPolicy(epsilon=0.1)

    Q, rewards = algorithm(
        env,
        policy,
        jax.random.key(42),
        num_episodes=0,
        return_history=True,
    )

    assert jnp.array_equal(
        Q,
        jnp.zeros((16, 4), dtype=jnp.float32),
    )
    assert rewards.shape == (0,)


@pytest.mark.parametrize(
    "algorithm",
    [sarsa_jit, q_learning_jit],
)
def test_control_jit_cliff_walking(algorithm):
    env = CliffWalking()
    policy = EpsilonGreedyPolicy(epsilon=0.1)

    Q, rewards = algorithm(
        env,
        policy,
        jax.random.key(42),
        num_episodes=20,
        alpha=0.5,
        max_steps=1000,
        return_history=True,
    )

    assert Q.shape == (48, 4)
    assert rewards.shape == (20,)
    assert bool(jnp.all(jnp.isfinite(Q)))
    assert bool(jnp.all(rewards <= -1))


@pytest.mark.parametrize(
    "algorithm",
    [sarsa_jit, q_learning_jit],
)
def test_control_jit_max_steps(algorithm):
    env = CliffWalking()
    policy = EpsilonGreedyPolicy(epsilon=0.1)

    with pytest.raises(RuntimeError, match="did not terminate"):
        algorithm(
            env,
            policy,
            jax.random.key(42),
            num_episodes=1,
            max_steps=1,
        )



def test_sarsa_jit_single_episode():
    env = Gridworld(size=4)
    policy = EpsilonGreedyPolicy(epsilon=0.1)
    key = jax.random.key(42)

    Q_reference = sarsa(
        env,
        policy,
        key,
        num_episodes=1,
        alpha=0.1,
        gamma=1.0,
    )

    Q_compiled = sarsa_jit(
        env,
        policy,
        key,
        num_episodes=1,
        alpha=0.1,
        gamma=1.0,
    )

    assert jnp.allclose(
        Q_reference,
        Q_compiled,
        atol=1e-6,
    )



class NeverTerminatingEnv:
    """
    Single-state environment that never terminates.

    Every transition returns to state 0 with reward -1.
    """

    num_states = 1
    num_actions = 1

    def reset(self, key):
        state = jnp.int32(0)
        return state, state

    def step(self, key, env_state, action):
        return (
            jnp.int32(0),
            jnp.int32(0),
            jnp.float32(-1.0),
            jnp.bool_(False),
            {},
        )



@pytest.mark.parametrize(
    "algorithm",
    [sarsa_jit, q_learning_jit],
)
def test_control_jit_truncation(algorithm):
    env = NeverTerminatingEnv()
    policy = FixedActionPolicy(action=0)

    Q, episode_returns = algorithm(
        env,
        policy,
        jax.random.key(42),
        num_episodes=3,
        alpha=0.1,
        gamma=1.0,
        max_steps=5,
        return_history=True,
        truncate=True,
    )

    expected_returns = jnp.array(
        [-5.0, -5.0, -5.0],
        dtype=jnp.float32,
    )

    assert jnp.array_equal(
        episode_returns,
        expected_returns,
    )

    assert Q.shape == (1, 1)
    assert jnp.all(jnp.isfinite(Q))




@pytest.mark.parametrize(
    "algorithm",
    [sarsa_jit, q_learning_jit],
)
def test_control_jit_truncation_disabled(algorithm):
    env = NeverTerminatingEnv()
    policy = FixedActionPolicy(action=0)

    with pytest.raises(
        RuntimeError,
        match="did not terminate",
    ):
        algorithm(
            env,
            policy,
            jax.random.key(42),
            num_episodes=3,
            max_steps=5,
            truncate=False,
        )


class FixedProbabilityPolicy:
    def __init__(self, probs):
        self.probs = jnp.asarray(probs, dtype=jnp.float32)

    def action_probs(self, q_values):
        return self.probs


from jax_rl.td.expected_sarsa import expected_sarsa_update



def test_expected_sarsa_update():
    Q = jnp.array([
        [2.0, 4.0, 6.0],
        [1.0, 5.0, 9.0],
    ])

    policy = FixedProbabilityPolicy(
        jnp.array([0.2, 0.3, 0.5])
    )

    result = expected_sarsa_update(
        Q,
        state=0,
        action=1,
        reward=2.0,
        next_state=1,
        terminated=False,
        policy=policy,
        alpha=0.5,
        gamma=1.0,
    )

    # Expected next-state value:
    # 0.2 * 1 + 0.3 * 5 + 0.5 * 9 = 6.2
    #
    # Target = 2 + 6.2 = 8.2
    # TD error = 8.2 - 4 = 4.2
    # Updated Q[0, 1] = 4 + 0.5 * 4.2 = 6.1

    expected = jnp.array([
        [2.0, 6.1, 6.0],
        [1.0, 5.0, 9.0],
    ])

    assert jnp.allclose(result, expected)


def test_expected_sarsa_discounting():
    Q = jnp.array([
        [2.0, 4.0, 6.0],
        [1.0, 5.0, 9.0],
    ])

    policy = FixedProbabilityPolicy(
        jnp.array([0.2, 0.3, 0.5])
    )

    result = expected_sarsa_update(
        Q,
        state=0,
        action=1,
        reward=2.0,
        next_state=1,
        terminated=False,
        policy=policy,
        alpha=0.5,
        gamma=0.5,
    )

    # Expected next-state value = 6.2
    # Target = 2 + 0.5 * 6.2 = 5.1
    # Updated Q[0, 1] = 4 + 0.5 * (5.1 - 4) = 4.55

    expected = jnp.array([
        [2.0, 4.55, 6.0],
        [1.0, 5.0, 9.0],
    ])

    assert jnp.allclose(result, expected)


def test_expected_sarsa_terminal_update():
    Q = jnp.array([
        [2.0, 4.0, 6.0],
        [1.0, 5.0, 9.0],
    ])

    policy = FixedProbabilityPolicy(
        jnp.array([0.2, 0.3, 0.5])
    )

    result = expected_sarsa_update(
        Q,
        state=0,
        action=1,
        reward=-2.0,
        next_state=1,
        terminated=True,
        policy=policy,
        alpha=0.5,
        gamma=1.0,
    )

    # Terminal target = -2
    # Updated Q[0, 1] = 4 + 0.5 * (-2 - 4) = 1

    expected = jnp.array([
        [2.0, 1.0, 6.0],
        [1.0, 5.0, 9.0],
    ])

    assert jnp.allclose(result, expected)


def test_expected_sarsa_zero_learning_rate():
    Q = jnp.array([
        [2.0, 4.0, 6.0],
        [1.0, 5.0, 9.0],
    ])

    policy = FixedProbabilityPolicy(
        jnp.array([0.2, 0.3, 0.5])
    )

    result = expected_sarsa_update(
        Q,
        state=0,
        action=1,
        reward=2.0,
        next_state=1,
        terminated=False,
        policy=policy,
        alpha=0.0,
        gamma=1.0,
    )

    assert jnp.array_equal(result, Q)



from jax_rl.td.expected_sarsa import expected_sarsa


def test_expected_sarsa_control_single_episode():
    env = SarsaTestEnv()
    policy = FixedActionPolicy(action=0)

    Q = expected_sarsa(
        env,
        policy,
        jax.random.key(42),
        num_episodes=1,
        alpha=0.5,
        gamma=1.0,
    )

    expected = jnp.array([
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 0.0],
    ])

    assert jnp.allclose(Q, expected)


def test_expected_sarsa_control_bootstrapping():
    env = SarsaTestEnv()
    policy = FixedActionPolicy(action=0)

    Q = expected_sarsa(
        env,
        policy,
        jax.random.key(42),
        num_episodes=2,
        alpha=0.5,
        gamma=1.0,
    )

    expected = jnp.array([
        [2.5, 0.0],
        [3.0, 0.0],
        [0.0, 0.0],
    ])

    assert jnp.allclose(Q, expected)


def test_expected_sarsa_control_discounting():
    env = SarsaTestEnv()
    policy = FixedActionPolicy(action=0)

    Q = expected_sarsa(
        env,
        policy,
        jax.random.key(42),
        num_episodes=2,
        alpha=0.5,
        gamma=0.5,
    )

    expected = jnp.array([
        [2.0, 0.0],
        [3.0, 0.0],
        [0.0, 0.0],
    ])

    assert jnp.allclose(Q, expected)


def test_expected_sarsa_control_action_selection():
    env = SarsaTestEnv()
    policy = AlternatingActionPolicy()

    Q = expected_sarsa(
        env,
        policy,
        jax.random.key(42),
        num_episodes=1,
        alpha=0.5,
    )

    assert env.actions_taken == [0, 1]
    assert policy.calls == 2

    assert Q.shape == (3, 2)


def test_expected_sarsa_control_zero_episodes():
    env = SarsaTestEnv()
    policy = FixedActionPolicy(action=0)

    Q = expected_sarsa(
        env,
        policy,
        jax.random.key(42),
        num_episodes=0,
    )

    assert jnp.array_equal(
        Q,
        jnp.zeros((3, 2), dtype=jnp.float32),
    )


def test_expected_sarsa_control_max_steps():
    env = SarsaTestEnv()
    policy = FixedActionPolicy(action=0)

    with pytest.raises(RuntimeError, match="did not terminate"):
        expected_sarsa(
            env,
            policy,
            jax.random.key(42),
            num_episodes=1,
            max_steps=1,
        )


def test_expected_sarsa_control_exact_max_steps():
    env = SarsaTestEnv()
    policy = FixedActionPolicy(action=0)

    Q = expected_sarsa(
        env,
        policy,
        jax.random.key(42),
        num_episodes=1,
        alpha=0.5,
        max_steps=2,
    )

    expected = jnp.array([
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 0.0],
    ])

    assert jnp.allclose(Q, expected)


from jax_rl.td.expected_sarsa import expected_sarsa, expected_sarsa_jit



@pytest.mark.parametrize("gamma", [0.5, 1.0])
@pytest.mark.parametrize("num_episodes", [1, 20])
def test_expected_sarsa_jit_matches_reference(gamma, num_episodes):
    env = Gridworld(size=4)
    policy = EpsilonGreedyPolicy(epsilon=0.1)
    key = jax.random.key(42)

    Q_reference = expected_sarsa(
        env,
        policy,
        key,
        num_episodes=num_episodes,
        alpha=0.1,
        gamma=gamma,
    )

    Q_compiled = expected_sarsa_jit(
        env,
        policy,
        key,
        num_episodes=num_episodes,
        alpha=0.1,
        gamma=gamma,
    )

    assert jnp.allclose(
        Q_reference,
        Q_compiled,
        atol=1e-5,
    )



def test_expected_sarsa_jit_zero_episodes():
    env = Gridworld(size=4)
    policy = EpsilonGreedyPolicy(epsilon=0.1)

    Q = expected_sarsa_jit(
        env,
        policy,
        jax.random.key(42),
        num_episodes=0,
    )

    expected = jnp.zeros(
        (env.num_states, env.num_actions),
        dtype=jnp.float32,
    )

    assert jnp.array_equal(Q, expected)



def test_expected_sarsa_jit_max_steps():
    env = CliffWalking()
    policy = EpsilonGreedyPolicy(epsilon=0.1)

    with pytest.raises(RuntimeError, match="did not terminate"):
        expected_sarsa_jit(
            env,
            policy,
            jax.random.key(42),
            num_episodes=1,
            max_steps=1,
        )



def test_expected_sarsa_jit_cliff_walking():
    env = CliffWalking()
    policy = EpsilonGreedyPolicy(epsilon=0.1)

    Q, episode_returns = expected_sarsa_jit(
        env,
        policy,
        jax.random.key(42),
        num_episodes=20,
        alpha=0.5,
        gamma=1.0,
        max_steps=1000,
        return_history=True,
        truncate=True,
    )

    assert Q.shape == (env.num_states, env.num_actions)
    assert episode_returns.shape == (20,)

    assert bool(jnp.all(jnp.isfinite(Q)))
    assert bool(jnp.all(jnp.isfinite(episode_returns)))
    assert bool(jnp.all(episode_returns <= -1))


class NStepTestEnv:
    """
    Deterministic trajectory:

    S0 --(+2)--> S1 --(+4)--> S2 (terminal)
    """

    num_states = 3
    num_actions = 1

    def reset(self, key):
        return jnp.int32(0), jnp.int32(0)

    def step(self, key, env_state, action):
        state = int(env_state)

        if state == 0:
            return (
                jnp.int32(1),
                jnp.int32(1),
                2.0,
                False,
                {},
            )

        if state == 1:
            return (
                jnp.int32(2),
                jnp.int32(2),
                4.0,
                True,
                {},
            )

        raise RuntimeError("Stepped after termination")


def n_step_test_policy():
    # One available action in each state.
    return jnp.ones((3, 1))


from jax_rl.td.td_prediction import n_step_td_prediction


class TestNStepTDPrediction:
    def setup_method(self):
        self.env = NStepTestEnv()
        self.policy = n_step_test_policy()
        self.key = jax.random.key(42)

    def test_one_step_single_episode(self):
        V = n_step_td_prediction(
            self.env,
            self.policy,
            self.key,
            num_episodes=1,
            n=1,
            alpha=0.5,
            gamma=1.0,
        )

        # S0: 0 + 0.5 * (2 + V(S1) - 0) = 1
        # S1: 0 + 0.5 * (4 - 0) = 2
        expected = jnp.array([1.0, 2.0, 0.0])

        assert jnp.allclose(V, expected)

    def test_two_step_single_episode(self):
        V = n_step_td_prediction(
            self.env,
            self.policy,
            self.key,
            num_episodes=1,
            n=2,
            alpha=0.5,
            gamma=1.0,
        )

        # S0: 0.5 * (2 + 4) = 3
        # S1: 0.5 * 4 = 2
        expected = jnp.array([3.0, 2.0, 0.0])

        assert jnp.allclose(V, expected)

    def test_n_larger_than_episode(self):
        V = n_step_td_prediction(
            self.env,
            self.policy,
            self.key,
            num_episodes=1,
            n=5,
            alpha=0.5,
            gamma=1.0,
        )

        # Both updates use complete returns.
        expected = jnp.array([3.0, 2.0, 0.0])

        assert jnp.allclose(V, expected)

    def test_discounting(self):
        V = n_step_td_prediction(
            self.env,
            self.policy,
            self.key,
            num_episodes=1,
            n=2,
            alpha=0.5,
            gamma=0.5,
        )

        # S0: 0.5 * (2 + 0.5*4) = 2
        # S1: 0.5 * 4 = 2
        expected = jnp.array([2.0, 2.0, 0.0])

        assert jnp.allclose(V, expected)

    def test_multiple_episodes(self):
        V = n_step_td_prediction(
            self.env,
            self.policy,
            self.key,
            num_episodes=2,
            n=2,
            alpha=0.5,
            gamma=1.0,
        )

        # Episode 1: V = [3, 2, 0]
        # Episode 2:
        # V(S0) = 3 + 0.5*(6 - 3) = 4.5
        # V(S1) = 2 + 0.5*(4 - 2) = 3
        expected = jnp.array([4.5, 3.0, 0.0])

        assert jnp.allclose(V, expected)

    def test_zero_episodes(self):
        V = n_step_td_prediction(
            self.env,
            self.policy,
            self.key,
            num_episodes=0,
            n=2,
        )

        assert jnp.array_equal(V, jnp.zeros(3))

    def test_exact_max_steps(self):
        V = n_step_td_prediction(
            self.env,
            self.policy,
            self.key,
            num_episodes=1,
            n=3,
            alpha=0.5,
            max_steps=2,
        )

        expected = jnp.array([3.0, 2.0, 0.0])
        assert jnp.allclose(V, expected)

    def test_max_steps_exceeded(self):
        with pytest.raises(RuntimeError, match="did not terminate"):
            n_step_td_prediction(
                self.env,
                self.policy,
                self.key,
                num_episodes=1,
                n=3,
                max_steps=1,
            )



class RepeatingNStepTestEnv:
    """
    S0 --(+2)--> S1 --(+4)--> S0
       --(+6)--> S2 --(+8)--> S3 (terminal)
    """

    num_states = 4
    num_actions = 1

    def reset(self, key):
        return (0, 0), jnp.int32(0)

    def step(self, key, env_state, action):
        t, state = env_state

        next_states = [1, 0, 2, 3]
        rewards = [2.0, 4.0, 6.0, 8.0]

        next_state = next_states[t]
        terminated = t == 3

        return (
            (t + 1, next_state),
            jnp.int32(next_state),
            rewards[t],
            terminated,
            {},
        )


def test_n_step_td_bootstraps_from_current_values():
    env = RepeatingNStepTestEnv()
    policy = jnp.ones((4, 1))

    V = n_step_td_prediction(
        env,
        policy,
        jax.random.key(42),
        num_episodes=2,
        n=2,
        alpha=0.5,
        gamma=1.0,
    )

    # Episode 1:
    # V(S0) = 3, then 3 + 0.5*(6+8-3) = 8.5
    # V(S1) = 0.5*(4+6) = 5
    # V(S2) = 0.5*8 = 4
    #
    # Episode 2:
    # First update to S0 bootstraps from its existing V(S0)=8.5:
    # G = 2+4+8.5 = 14.5
    # V(S0) = 8.5 + 0.5*(14.5-8.5) = 11.5
    #
    # Subsequent updates:
    # V(S1) = 5 + 0.5*(4+6+4-5) = 9.5
    # V(S0) = 11.5 + 0.5*(6+8-11.5) = 12.75
    # V(S2) = 4 + 0.5*(8-4) = 6

    expected = jnp.array([12.75, 9.5, 6.0, 0.0])

    assert jnp.allclose(V, expected)

from jax_rl.td.td_prediction import n_step_td_prediction, n_step_td_prediction_jit



@pytest.mark.parametrize("n", [1, 2, 3, 5, 10])
@pytest.mark.parametrize("gamma", [0.5, 1.0])
@pytest.mark.parametrize("num_episodes", [1, 20])
def test_n_step_td_jit_matches_reference(
    n, gamma, num_episodes
):
    env = Gridworld(size=4)
    policy = jnp.full(
        (env.num_states, env.num_actions),
        1.0 / env.num_actions,
    )
    key = jax.random.key(42)

    kwargs = dict(
        num_episodes=num_episodes,
        n=n,
        alpha=0.1,
        gamma=gamma,
        max_steps=1000,
    )

    V_reference = n_step_td_prediction(
        env, policy, key, **kwargs
    )

    V_compiled = n_step_td_prediction_jit(
        env, policy, key, **kwargs
    )

    assert jnp.allclose(
        V_reference,
        V_compiled,
        atol=1e-5,
        rtol=1e-5,
    ), (
        f"n={n}, gamma={gamma}, "
        f"num_episodes={num_episodes}\n"
        f"reference={V_reference}\n"
        f"compiled={V_compiled}"
    )



def test_n_step_td_jit_zero_episodes():
    env = Gridworld(size=4)
    policy = jnp.full(
        (env.num_states, env.num_actions),
        1.0 / env.num_actions,
    )

    V = n_step_td_prediction_jit(
        env,
        policy,
        jax.random.key(42),
        num_episodes=0,
        n=3,
    )

    assert jnp.array_equal(
        V,
        jnp.zeros((env.num_states,)),
    )



class JitNStepTestEnv:
    """
    S0 --(+2)--> S1 --(+4)--> S2 (terminal)
    """

    num_states = 3
    num_actions = 1

    def reset(self, key):
        state = jnp.int32(0)
        return state, state

    def step(self, key, env_state, action):
        next_state = jnp.minimum(
            env_state + 1,
            self.num_states - 1,
        )

        reward = jnp.where(
            env_state == 0,
            2.0,
            4.0,
        )

        terminated = next_state == 2

        return (
            next_state,
            next_state,
            reward,
            terminated,
            {},
        )



@pytest.mark.parametrize(
    "n, gamma, expected",
    [
        (1, 1.0, [1.0, 2.0, 0.0]),
        (2, 1.0, [3.0, 2.0, 0.0]),
        (5, 1.0, [3.0, 2.0, 0.0]),
        (2, 0.5, [2.0, 2.0, 0.0]),
    ],
)
def test_n_step_td_jit_terminal_flushing(
    n, gamma, expected
):
    env = JitNStepTestEnv()
    policy = jnp.ones((env.num_states, env.num_actions))

    V = n_step_td_prediction_jit(
        env,
        policy,
        jax.random.key(42),
        num_episodes=1,
        n=n,
        alpha=0.5,
        gamma=gamma,
        max_steps=10,
    )

    assert jnp.allclose(
        V,
        jnp.array(expected),
        atol=1e-6,
    )



def test_n_step_td_jit_exact_max_steps():
    env = JitNStepTestEnv()
    policy = jnp.ones((env.num_states, env.num_actions))

    V = n_step_td_prediction_jit(
        env,
        policy,
        jax.random.key(42),
        num_episodes=1,
        n=3,
        alpha=0.5,
        max_steps=2,
    )

    assert jnp.allclose(
        V,
        jnp.array([3.0, 2.0, 0.0]),
    )


def test_n_step_td_jit_max_steps_exceeded():
    env = JitNStepTestEnv()
    policy = jnp.ones((env.num_states, env.num_actions))

    with pytest.raises(
        RuntimeError,
        match="did not terminate",
    ):
        n_step_td_prediction_jit(
            env,
            policy,
            jax.random.key(42),
            num_episodes=1,
            n=3,
            max_steps=1,
        )



class JitRepeatingNStepTestEnv:
    """
    S0 --(+2)--> S1 --(+4)--> S0
       --(+6)--> S2 --(+8)--> S3 (terminal)
    """

    num_states = 4
    num_actions = 1

    def reset(self, key):
        return jnp.int32(0), jnp.int32(0)

    def step(self, key, env_state, action):
        next_states = jnp.array([1, 0, 2, 3])
        rewards = jnp.array([2.0, 4.0, 6.0, 8.0])

        next_state = next_states[env_state]
        reward = rewards[env_state]
        terminated = env_state == 3

        return (
            env_state + 1,
            next_state,
            reward,
            terminated,
            {},
        )


def test_n_step_td_jit_repeated_state_bootstrapping():
    env = JitRepeatingNStepTestEnv()
    policy = jnp.ones((env.num_states, env.num_actions))

    V = n_step_td_prediction_jit(
        env,
        policy,
        jax.random.key(42),
        num_episodes=2,
        n=2,
        alpha=0.5,
        gamma=1.0,
    )

    expected = jnp.array([12.75, 9.5, 6.0, 0.0])

    assert jnp.allclose(V, expected, atol=1e-6)


from jax_rl.td.sarsa import n_step_sarsa_return


class TestNStepSarsaReturn:
    def setup_method(self):
        self.rewards = [1.0, 2.0, 3.0]
        self.states = [0, 1, 2, 3]
        self.actions = [0, 1, 0, 1]

        self.Q = jnp.array([
            [10.0, 11.0],
            [20.0, 21.0],
            [30.0, 31.0],
            [40.0, 41.0],
        ])

    def test_one_step_return(self):
        result = n_step_sarsa_return(
            self.rewards,
            self.states,
            self.actions,
            self.Q,
            tau=0,
            T=5,
            n=1,
            gamma=0.5,
        )

        # R1 + gamma * Q(S1, A1)
        # 1 + 0.5 * 21 = 11.5
        assert jnp.isclose(result, 11.5)

    def test_two_step_return(self):
        result = n_step_sarsa_return(
            self.rewards,
            self.states,
            self.actions,
            self.Q,
            tau=0,
            T=5,
            n=2,
            gamma=0.5,
        )

        # R1 + gamma*R2 + gamma^2*Q(S2,A2)
        # 1 + 0.5*2 + 0.25*30 = 9.5
        assert jnp.isclose(result, 9.5)



def test_bootstrap_at_last_nonterminal_state(self):
    # n=2, so rewards has 2 slots and states/actions have 3.
    # R2 occupies slot 1; R3 occupies slot 0.
    rewards = [3.0, 2.0]
    states = [3, 1, 2]
    actions = [1, 1, 0]

    result = n_step_sarsa_return(
        rewards,
        states,
        actions,
        self.Q,
        tau=1,
        T=4,
        n=2,
        gamma=0.5,
    )

    # tau+n = 3 = T-1, so bootstrapping is required.
    # 2 + 0.5*3 + 0.25*Q(3, 1)
    # = 2 + 1.5 + 10.25 = 13.75
    assert jnp.isclose(result, 13.75)



def test_no_bootstrap_at_terminal_state(self):
    rewards = [3.0, 2.0]
    states = [3, 1, 2]
    actions = [1, 1, 0]

    result = n_step_sarsa_return(
        rewards,
        states,
        actions,
        self.Q,
        tau=1,
        T=3,
        n=2,
        gamma=0.5,
    )

    # tau+n = T, so no bootstrap.
    # 2 + 0.5*3 = 3.5
    assert jnp.isclose(result, 3.5)



    def test_n_exceeds_remaining_episode(self):
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        states = [0, 1, 2, 3, 0, 1]
        actions = [0, 1, 0, 1, 0, 1]

        result = n_step_sarsa_return(
            rewards,
            states,
            actions,
            self.Q,
            tau=3,
            T=5,
            n=5,
            gamma=0.5,
        )

        # R4 + gamma*R5, with no bootstrap.
        assert jnp.isclose(result, 6.5)

    def test_gamma_one(self):
        result = n_step_sarsa_return(
            self.rewards,
            self.states,
            self.actions,
            self.Q,
            tau=0,
            T=5,
            n=3,
            gamma=1.0,
        )

        # R1 + R2 + R3 + Q(S3,A3)
        assert jnp.isclose(result, 47.0)

    def test_gamma_zero(self):
        result = n_step_sarsa_return(
            self.rewards,
            self.states,
            self.actions,
            self.Q,
            tau=0,
            T=5,
            n=3,
            gamma=0.0,
        )

        assert jnp.isclose(result, 1.0)

    def test_circular_buffer_wraparound(self):
        n = 2

        # At tau=3, rewards R4 and R5 occupy slots 1 and 0.
        rewards = [5.0, 4.0]

        # State/action buffer length = n+1 = 3.
        # S5, A5 occupy slot 5 % 3 = 2.
        states = [3, 4, 2]
        actions = [0, 1, 1]

        result = n_step_sarsa_return(
            rewards,
            states,
            actions,
            self.Q,
            tau=3,
            T=7,
            n=n,
            gamma=0.5,
        )

        # R4 + gamma*R5 + gamma^2*Q(S5,A5)
        # 4 + 0.5*5 + 0.25*31 = 14.25
        assert jnp.isclose(result, 14.25)

    def test_full_episode_equals_monte_carlo_return(self):
        result = n_step_sarsa_return(
            self.rewards,
            self.states,
            self.actions,
            self.Q,
            tau=0,
            T=3,
            n=3,
            gamma=0.5,
        )

        # 1 + 0.5*2 + 0.25*3 = 2.75
        assert jnp.isclose(result, 2.75)

from jax_rl.td.sarsa import n_step_sarsa


class NStepSarsaActionTestEnv:
    num_states = 3
    num_actions = 2

    def __init__(self):
        self.actions_taken = []

    def reset(self, key):
        self.actions_taken.clear()
        return jnp.int32(0), jnp.int32(0)

    def step(self, key, env_state, action):
        state = int(env_state)
        self.actions_taken.append(int(action))

        if state == 0:
            return jnp.int32(1), jnp.int32(1), 2.0, False, {}

        if state == 1:
            return jnp.int32(2), jnp.int32(2), 4.0, True, {}

        raise RuntimeError("Stepped after termination")


class RepeatingNStepSarsaTestEnv:
    """
    Four transitions, including a repeated visit to S0:

    S0 --(+2)--> S1 --(+4)--> S0
       --(+6)--> S2 --(+8)--> S3 (terminal)
    """

    num_states = 4
    num_actions = 2

    def reset(self, key):
        return (0, 0), jnp.int32(0)

    def step(self, key, env_state, action):
        t, state = env_state

        next_states = [1, 0, 2, 3]
        rewards = [2.0, 4.0, 6.0, 8.0]

        next_state = next_states[t]

        return (
            (t + 1, next_state),
            jnp.int32(next_state),
            rewards[t],
            t == 3,
            {},
        )



class TestNStepSarsa:
    def setup_method(self):
        self.env = NStepTestEnv()
        self.policy = FixedActionPolicy(action=0)
        self.key = jax.random.key(42)

    @pytest.mark.parametrize(
        "n, expected",
        [
            (1, [1.0, 2.0, 0.0]),
            (2, [3.0, 2.0, 0.0]),
            (5, [3.0, 2.0, 0.0]),
        ],
    )
    def test_single_episode(self, n, expected):
        Q = n_step_sarsa(
            self.env,
            self.policy,
            self.key,
            num_episodes=1,
            n=n,
            alpha=0.5,
            gamma=1.0,
        )

        assert Q.shape == (3, 1)
        assert jnp.allclose(Q[:, 0], jnp.array(expected))

    def test_discounting(self):
        Q = n_step_sarsa(
            self.env,
            self.policy,
            self.key,
            num_episodes=1,
            n=2,
            alpha=0.5,
            gamma=0.5,
        )

        # S0: 0.5 * (2 + 0.5 * 4) = 2
        # S1: 0.5 * 4 = 2
        assert jnp.allclose(
            Q[:, 0],
            jnp.array([2.0, 2.0, 0.0]),
        )

    def test_multiple_episodes(self):
        Q = n_step_sarsa(
            self.env,
            self.policy,
            self.key,
            num_episodes=2,
            n=2,
            alpha=0.5,
            gamma=1.0,
        )

        # Episode 1: Q(S0,0)=3, Q(S1,0)=2
        # Episode 2: Q(S0,0)=4.5, Q(S1,0)=3
        assert jnp.allclose(
            Q[:, 0],
            jnp.array([4.5, 3.0, 0.0]),
        )

    def test_zero_episodes(self):
        Q = n_step_sarsa(
            self.env,
            self.policy,
            self.key,
            num_episodes=0,
            n=2,
        )

        assert jnp.array_equal(Q, jnp.zeros((3, 1)))

    def test_exact_max_steps(self):
        Q = n_step_sarsa(
            self.env,
            self.policy,
            self.key,
            num_episodes=1,
            n=3,
            alpha=0.5,
            max_steps=2,
        )

        assert jnp.allclose(
            Q[:, 0],
            jnp.array([3.0, 2.0, 0.0]),
        )

    def test_max_steps_exceeded(self):
        with pytest.raises(RuntimeError, match="did not terminate"):
            n_step_sarsa(
                self.env,
                self.policy,
                self.key,
                num_episodes=1,
                n=3,
                max_steps=1,
            )

    def test_action_selection(self):
        env = NStepSarsaActionTestEnv()
        policy = AlternatingActionPolicy()

        Q = n_step_sarsa(
            env,
            policy,
            self.key,
            num_episodes=1,
            n=2,
            alpha=0.5,
        )

        assert env.actions_taken == [0, 1]
        assert policy.calls == 2

        expected = jnp.array([
            [3.0, 0.0],
            [0.0, 2.0],
            [0.0, 0.0],
        ])

        assert jnp.allclose(Q, expected)

    def test_repeated_state_bootstrapping(self):
        env = RepeatingNStepSarsaTestEnv()

        Q = n_step_sarsa(
            env,
            self.policy,
            self.key,
            num_episodes=2,
            n=2,
            alpha=0.5,
            gamma=1.0,
        )

        # For a deterministic action-0 policy, this should
        # match the n-step TD repeating-state calculation.
        expected = jnp.array([
            [12.75, 0.0],
            [9.5, 0.0],
            [6.0, 0.0],
            [0.0, 0.0],
        ])

        assert jnp.allclose(Q, expected)


from jax_rl.td.sarsa import n_step_sarsa, n_step_sarsa_jit

@pytest.mark.parametrize("n", [1, 2, 3, 5])
@pytest.mark.parametrize("gamma", [0.5, 1.0])
@pytest.mark.parametrize("num_episodes", [1, 20])
def test_n_step_sarsa_jit_matches_reference(
    n, gamma, num_episodes
):
    env = Gridworld(size=4)
    policy = EpsilonGreedyPolicy(epsilon=0.1)
    key = jax.random.key(42)

    kwargs = dict(
        num_episodes=num_episodes,
        n=n,
        alpha=0.1,
        gamma=gamma,
        max_steps=1000,
    )

    Q_reference = n_step_sarsa(
        env, policy, key, **kwargs
    )

    Q_compiled = n_step_sarsa_jit(
        env, policy, key, **kwargs
    )

    assert jnp.allclose(
        Q_reference,
        Q_compiled,
        atol=1e-5,
        rtol=1e-5,
    ), (
        f"n={n}, gamma={gamma}, "
        f"num_episodes={num_episodes}\n"
        f"Reference:\n{Q_reference}\n"
        f"JIT:\n{Q_compiled}"
    )



class JitNStepSarsaActionTestEnv:
    """
    Two transitions with two available actions.

    S0 --(+2)--> S1 --(+4)--> S2 (terminal).
    Rewards and transitions do not depend on the action.
    """

    num_states = 3
    num_actions = 2

    def reset(self, key):
        state = jnp.int32(0)
        return state, state

    def step(self, key, env_state, action):
        next_state = jnp.minimum(env_state + 1, 2)

        reward = jnp.where(
            env_state == 0,
            2.0,
            4.0,
        )

        terminated = next_state == 2

        return (
            next_state,
            next_state,
            reward,
            terminated,
            {},
        )


@pytest.mark.parametrize(
    "n, gamma, expected",
    [
        (1, 1.0, [1.0, 2.0, 0.0]),
        (2, 1.0, [3.0, 2.0, 0.0]),
        (5, 1.0, [3.0, 2.0, 0.0]),
        (2, 0.5, [2.0, 2.0, 0.0]),
    ],
)
def test_n_step_sarsa_jit_terminal_flushing(
    n, gamma, expected
):
    env = JitNStepTestEnv()
    policy = EpsilonGreedyPolicy(epsilon=0.1)

    Q = n_step_sarsa_jit(
        env,
        policy,
        jax.random.key(42),
        num_episodes=1,
        n=n,
        alpha=0.5,
        gamma=gamma,
        max_steps=10,
    )

    assert Q.shape == (3, 1)
    assert jnp.allclose(
        Q[:, 0],
        jnp.array(expected),
        atol=1e-6,
    )



@pytest.mark.parametrize("action", [0, 1])
def test_n_step_sarsa_jit_action_values(action):
    env = JitNStepSarsaActionTestEnv()
    policy = FixedActionPolicy(action=action)

    Q = n_step_sarsa_jit(
        env,
        policy,
        jax.random.key(42),
        num_episodes=1,
        n=2,
        alpha=0.5,
        gamma=1.0,
    )

    expected = jnp.zeros((3, 2))
    expected = expected.at[0, action].set(3.0)
    expected = expected.at[1, action].set(2.0)

    assert jnp.allclose(Q, expected)



def test_n_step_sarsa_jit_repeated_state_bootstrapping():
    env = JitRepeatingNStepTestEnv()
    policy = EpsilonGreedyPolicy(epsilon=0.0)

    Q = n_step_sarsa_jit(
        env,
        policy,
        jax.random.key(42),
        num_episodes=2,
        n=2,
        alpha=0.5,
        gamma=1.0,
    )

    expected = jnp.array([
        [12.75],
        [9.5],
        [6.0],
        [0.0],
    ])

    assert jnp.allclose(Q, expected, atol=1e-6)



def test_n_step_sarsa_jit_zero_episodes():
    env = Gridworld(size=4)
    policy = EpsilonGreedyPolicy(epsilon=0.1)

    Q = n_step_sarsa_jit(
        env,
        policy,
        jax.random.key(42),
        num_episodes=0,
        n=3,
    )

    assert jnp.array_equal(
        Q,
        jnp.zeros((env.num_states, env.num_actions)),
    )


def test_n_step_sarsa_jit_exact_max_steps():
    env = JitNStepTestEnv()
    policy = EpsilonGreedyPolicy(epsilon=0.1)

    Q = n_step_sarsa_jit(
        env,
        policy,
        jax.random.key(42),
        num_episodes=1,
        n=3,
        alpha=0.5,
        max_steps=2,
    )

    assert jnp.allclose(
        Q[:, 0],
        jnp.array([3.0, 2.0, 0.0]),
    )


def test_n_step_sarsa_jit_max_steps_exceeded():
    env = JitNStepTestEnv()
    policy = EpsilonGreedyPolicy(epsilon=0.1)

    with pytest.raises(RuntimeError, match="did not terminate"):
        n_step_sarsa_jit(
            env,
            policy,
            jax.random.key(42),
            num_episodes=1,
            n=3,
            max_steps=1,
        )



class JitNeverTerminatingSarsaEnv:
    num_states = 1
    num_actions = 1

    def reset(self, key):
        return jnp.int32(0), jnp.int32(0)

    def step(self, key, env_state, action):
        return (
            env_state,
            jnp.int32(0),
            jnp.float32(1.0),
            jnp.bool_(False),
            {},
        )


def test_n_step_sarsa_jit_truncation_bootstraps():
    env = JitNeverTerminatingSarsaEnv()
    policy = EpsilonGreedyPolicy(epsilon=0.0)

    Q = n_step_sarsa_jit(
        env,
        policy,
        jax.random.key(42),
        num_episodes=2,
        n=3,
        alpha=0.5,
        gamma=1.0,
        max_steps=2,
        truncate=True,
    )

    # Episode 1:
    # G0 = 1 + 1 + Q(S2,A2) = 2
    # Q = 0 + 0.5 * 2 = 1
    # G1 = 1 + Q(S2,A2) = 2
    # Q = 1 + 0.5 * (2 - 1) = 1.5
    #
    # Episode 2:
    # G0 = 1 + 1 + 1.5 = 3.5
    # Q = 1.5 + 0.5 * (3.5 - 1.5) = 2.5
    # G1 = 1 + 2.5 = 3.5
    # Q = 2.5 + 0.5 * (3.5 - 2.5) = 3.0

    assert jnp.allclose(Q, jnp.array([[3.0]]))


from jax_rl.td.sarsa import n_step_importance_sampling_ratio


class FixedDistributionPolicy:
    def __init__(self, probs):
        self.probs = jnp.asarray(probs, dtype=jnp.float32)

    def action_probs(self, q_values):
        return self.probs

    def select_action(self, key, q_values):
        return jax.random.choice(
            key,
            a=len(self.probs),
            p=self.action_probs(q_values),
        )

from jax_rl.td.sarsa import off_policy_n_step_sarsa



class TestOffPolicyNStepSarsa:
    def setup_method(self):
        self.env = NStepSarsaActionTestEnv()
        self.behavior_policy = FixedDistributionPolicy([1.0, 0.0])
        self.key = jax.random.key(42)

    @pytest.mark.parametrize(
        "n, expected",
        [
            (1, [1.0, 2.0, 0.0]),
            (2, [3.0, 2.0, 0.0]),
            (5, [3.0, 2.0, 0.0]),
        ],
    )
    def test_identical_deterministic_policies(self, n, expected):
        Q = off_policy_n_step_sarsa(
            self.env,
            self.behavior_policy,
            self.behavior_policy,
            self.key,
            num_episodes=1,
            n=n,
            alpha=0.5,
            gamma=1.0,
        )

        assert jnp.allclose(
            Q[:, 0],
            jnp.array(expected),
        )

        assert jnp.allclose(Q[:, 1], 0.0)

    def test_importance_sampling_scales_update(self):
        target_policy = FixedDistributionPolicy([0.5, 0.5])

        Q = off_policy_n_step_sarsa(
            self.env,
            self.behavior_policy,
            target_policy,
            self.key,
            num_episodes=1,
            n=2,
            alpha=0.5,
            gamma=1.0,
        )

        # At tau=0:
        # G = 2 + 4 = 6
        # rho = pi(A1=0) / b(A1=0) = 0.5 / 1 = 0.5
        # Q(S0,0) = 0.5 * 0.5 * 6 = 1.5
        #
        # At tau=1:
        # G = 4
        # rho = 1 (empty product)
        # Q(S1,0) = 0.5 * 4 = 2

        expected = jnp.array([
            [1.5, 0.0],
            [2.0, 0.0],
            [0.0, 0.0],
        ])

        assert jnp.allclose(Q, expected)

    def test_zero_importance_ratio(self):
        target_policy = FixedDistributionPolicy([0.0, 1.0])

        Q = off_policy_n_step_sarsa(
            self.env,
            self.behavior_policy,
            target_policy,
            self.key,
            num_episodes=1,
            n=2,
            alpha=0.5,
        )

        # At tau=0, the intermediate action A1=0
        # has zero probability under the target policy.
        # Therefore rho=0 and Q(S0,0) is unchanged.
        #
        # At tau=1, the ratio is an empty product.
        # Q(S1,0) still receives its terminal update.

        expected = jnp.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 0.0],
        ])

        assert jnp.allclose(Q, expected)

    def test_one_step_bootstrap_action_correction(self):
        target_policy = FixedDistributionPolicy([0.5, 0.5])

        Q = off_policy_n_step_sarsa(
            self.env,
            self.behavior_policy,
            target_policy,
            self.key,
            num_episodes=1,
            n=1,
            alpha=0.5,
        )

        # The one-step return bootstraps from A1=0.
        # Its importance ratio is 0.5 / 1 = 0.5.
        #
        # Q(S0,0) = 0.5 * 0.5 * (2 + 0) = 0.5
        # Q(S1,0) = 0.5 * 4 = 2

        expected = jnp.array([
            [0.5, 0.0],
            [2.0, 0.0],
            [0.0, 0.0],
        ])

        assert jnp.allclose(Q, expected)

    def test_discounting(self):
        Q = off_policy_n_step_sarsa(
            self.env,
            self.behavior_policy,
            self.behavior_policy,
            self.key,
            num_episodes=1,
            n=2,
            alpha=0.5,
            gamma=0.5,
        )

        # G0 = 2 + 0.5 * 4 = 4
        # G1 = 4
        # Both importance ratios are 1.

        expected = jnp.array([
            [2.0, 0.0],
            [2.0, 0.0],
            [0.0, 0.0],
        ])

        assert jnp.allclose(Q, expected)

    def test_multiple_episodes(self):
        target_policy = FixedDistributionPolicy([0.5, 0.5])

        Q = off_policy_n_step_sarsa(
            self.env,
            self.behavior_policy,
            target_policy,
            self.key,
            num_episodes=2,
            n=2,
            alpha=0.5,
        )

        # Episode 1: Q(S0,0)=1.5, Q(S1,0)=2
        #
        # Episode 2:
        # Q(S0,0) = 1.5 + 0.5*0.5*(6-1.5)
        #         = 2.625
        # Q(S1,0) = 2 + 0.5*(4-2) = 3

        expected = jnp.array([
            [2.625, 0.0],
            [3.0, 0.0],
            [0.0, 0.0],
        ])

        assert jnp.allclose(Q, expected)

    def test_zero_episodes(self):
        Q = off_policy_n_step_sarsa(
            self.env,
            self.behavior_policy,
            self.behavior_policy,
            self.key,
            num_episodes=0,
            n=3,
        )

        assert jnp.array_equal(Q, jnp.zeros((3, 2)))

    def test_exact_max_steps(self):
        Q = off_policy_n_step_sarsa(
            self.env,
            self.behavior_policy,
            self.behavior_policy,
            self.key,
            num_episodes=1,
            n=3,
            alpha=0.5,
            max_steps=2,
        )

        assert jnp.allclose(
            Q[:, 0],
            jnp.array([3.0, 2.0, 0.0]),
        )

    def test_max_steps_exceeded(self):
        with pytest.raises(
            RuntimeError,
            match="did not terminate",
        ):
            off_policy_n_step_sarsa(
                self.env,
                self.behavior_policy,
                self.behavior_policy,
                self.key,
                num_episodes=1,
                n=3,
                max_steps=1,
            )

    @pytest.mark.parametrize("n", [0, -1])
    def test_invalid_n(self, n):
        with pytest.raises(ValueError, match="n must be positive"):
            off_policy_n_step_sarsa(
                self.env,
                self.behavior_policy,
                self.behavior_policy,
                self.key,
                n=n,
            )

    def test_invalid_max_steps(self):
        with pytest.raises(
            ValueError,
            match="max_steps must be positive",
        ):
            off_policy_n_step_sarsa(
                self.env,
                self.behavior_policy,
                self.behavior_policy,
                self.key,
                max_steps=0,
            )



@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_off_policy_n_step_sarsa_matches_on_policy(n):
    env = NStepSarsaActionTestEnv()
    policy = FixedDistributionPolicy([0.75, 0.25])
    key = jax.random.key(42)

    kwargs = dict(
        num_episodes=20,
        n=n,
        alpha=0.1,
        gamma=0.9,
        max_steps=10,
    )

    Q_on_policy = n_step_sarsa(
        env,
        policy,
        key,
        **kwargs,
    )

    Q_off_policy = off_policy_n_step_sarsa(
        env,
        policy,
        policy,
        key,
        **kwargs,
    )

    assert jnp.allclose(
        Q_on_policy,
        Q_off_policy,
        atol=1e-6,
    )



def test_n_step_ratio_uses_recorded_behavior_probability():
    n = 2

    states = [0, 1, 2]
    actions = [0, 0, 0]

    # The behavior probability of A1 was 0.8
    # when that action was originally selected.
    behavior_prob_buffer = [1.0, 0.8, 1.0]

    target_policy = FixedDistributionPolicy([0.5, 0.5])
    Q = jnp.zeros((3, 2))

    rho = n_step_importance_sampling_ratio(
        states,
        actions,
        behavior_prob_buffer,
        target_policy,
        Q,
        tau=0,
        T=2,
        n=n,
    )

    # A1 is the only action included.
    # rho = 0.5 / 0.8 = 0.625

    assert jnp.isclose(rho, 0.625)



def test_off_policy_environment_terminates():
    env = NStepSarsaActionTestEnv()
    key = jax.random.key(42)

    env_state, observation = env.reset(key)

    assert int(observation) == 0

    env_state, observation, reward, terminated, _ = env.step(
        key, env_state, jnp.int32(0)
    )

    assert int(observation) == 1
    assert reward == 2.0
    assert not bool(terminated)

    env_state, observation, reward, terminated, _ = env.step(
        key, env_state, jnp.int32(0)
    )

    assert int(observation) == 2
    assert reward == 4.0
    assert bool(terminated)
