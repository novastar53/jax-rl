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
