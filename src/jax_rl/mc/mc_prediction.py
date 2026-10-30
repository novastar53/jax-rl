import jax
from functools import partial
import jax.numpy as jnp

from jax_rl.mc.episode import generate_episode


@jax.jit
def compute_returns(rewards, gamma=1.0):
    def scan_step(G, reward):
        G = reward + gamma * G
        return G, G

    _, returns = jax.lax.scan(
        scan_step,
        jnp.array(0.0, dtype=rewards.dtype),
        rewards,
        reverse=True,
    )

    return returns


def update_mc_values(V, N, states, returns, visit_mode="first"):
    """
    Update MC state-value estimates from one complete episode.

    V: (S,) current state-value estimates
    N: (S,) current update counts
    states: (T,) states visited before each action
    returns: (T,) corresponding returns
    visit_mode: "first" or "every"

    Returns:
        Updated (V, N).
    """
    if visit_mode not in ("first", "every"):
        raise ValueError("visit_mode must be 'first' or 'every'")

    seen_states = set()

    for state, G in zip(states, returns):
        s = int(state)

        if visit_mode == "first":
            if s in seen_states:
                continue
            seen_states.add(s)

        count = N[s] + 1

        V = V.at[s].set(
            V[s] + (G - V[s]) / count
        )
        N = N.at[s].set(count)

    return V, N


def mc_prediction(
    env,
    policy,
    key,
    num_episodes=10000,
    gamma=1.0,
    visit_mode="first",
    callback=None,
    callback_every=50,
):
    """
    Monte Carlo state-value prediction.

    Returns:
        V: (S,) estimated state values
        N: (S,) update counts
    """
    if visit_mode not in ("first", "every"):
        raise ValueError("visit_mode must be 'first' or 'every'")

    if num_episodes < 0:
        raise ValueError("num_episodes must be nonnegative")

    if callback is not None and callback_every < 1:
        raise ValueError("callback_every must be positive")

    num_states = policy.shape[0]

    V = jnp.zeros(num_states, dtype=jnp.float32)
    N = jnp.zeros(num_states, dtype=jnp.int32)

    for episode in range(1, num_episodes + 1):
        key, episode_key = jax.random.split(key)

        states, _, rewards = generate_episode(
            env, policy, episode_key
        )

        returns = compute_returns(rewards, gamma)

        V, N = update_mc_values(
            V, N, states, returns, visit_mode
        )

        if callback is not None and (
            episode % callback_every == 0
            or episode == num_episodes
        ):
            callback(episode, V, N)

    return V, N



@partial(
    jax.jit,
    static_argnames=(
        "env",
        "num_episodes",
        "max_steps",
        "visit_mode",
    ),
)
def _mc_prediction_compiled(
    env,
    policy,
    key,
    num_episodes,
    gamma,
    max_steps,
    visit_mode,
):
    num_states, num_actions = policy.shape

    V_initial = jnp.zeros(num_states, dtype=jnp.float32)
    N_initial = jnp.zeros(num_states, dtype=jnp.int32)

    def run_episode(carry):
        key, V, N, failed = carry

        # Match mc_prediction(): allocate an episode key.
        key, episode_key = jax.random.split(key)

        # Match generate_episode(): split the episode key
        # again to obtain the environment reset key.
        episode_key, reset_key = jax.random.split(episode_key)

        env_state, state = env.reset(reset_key)

        # Fixed-size buffers for one episode.
        states = jnp.zeros(max_steps, dtype=jnp.int32)
        rewards = jnp.zeros(max_steps, dtype=jnp.float32)

        initial = (
            episode_key,
            env_state,
            state,
            states,
            rewards,
            jnp.int32(0),
            jnp.bool_(False),
        )

        def generate_condition(c):
            _, _, _, _, _, t, terminated = c
            return (t < max_steps) & (~terminated)

        def generate_step(c):
            (
                key,
                env_state,
                state,
                states,
                rewards,
                t,
                _,
            ) = c

            key, action_key, step_key = jax.random.split(
                key, 3
            )

            action = jax.random.choice(
                action_key,
                a=num_actions,
                p=policy[state],
            )

            (
                next_env_state,
                next_state,
                reward,
                terminated,
                _,
            ) = env.step(step_key, env_state, action)

            states = states.at[t].set(state)
            rewards = rewards.at[t].set(reward)

            return (
                key,
                next_env_state,
                next_state,
                states,
                rewards,
                t + 1,
                terminated,
            )

        (
            _episode_key,
            _,
            _,
            states,
            rewards,
            length,
            terminated,
        ) = jax.lax.while_loop(
            generate_condition,
            generate_step,
            initial,
        )

        # Calculate returns backward over valid transitions.
        returns = jnp.zeros(max_steps, dtype=jnp.float32)

        def return_condition(c):
            t, _, _ = c
            return t >= 0

        def return_step(c):
            t, G, returns = c

            G = rewards[t] + gamma * G
            returns = returns.at[t].set(G)

            return t - 1, G, returns

        _, _, returns = jax.lax.while_loop(
            return_condition,
            return_step,
            (length - 1, jnp.float32(0.0), returns),
        )

        # Update values in chronological order.
        seen = jnp.zeros(num_states, dtype=jnp.bool_)

        def update_condition(c):
            t, _, _, _ = c
            return t < length

        def update_step(c):
            t, V, N, seen = c

            s = states[t]
            G = returns[t]

            if visit_mode == "first":
                should_update = ~seen[s]
            else:
                should_update = jnp.bool_(True)

            def apply_update(values):
                V, N = values

                count = N[s] + 1
                V = V.at[s].set(
                    V[s] + (G - V[s]) / count
                )
                N = N.at[s].set(count)

                return V, N

            V, N = jax.lax.cond(
                should_update,
                apply_update,
                lambda values: values,
                (V, N),
            )

            seen = seen.at[s].set(True)

            return t + 1, V, N, seen

        _, V, N, _ = jax.lax.while_loop(
            update_condition,
            update_step,
            (jnp.int32(0), V, N, seen),
        )

        failed = failed | (~terminated)
        return key, V, N, failed

    def episode_step(carry, _):
        # Stop further training if an episode failed.
        carry = jax.lax.cond(
            carry[3],
            lambda c: c,
            run_episode,
            carry,
        )
        return carry, None

    initial_carry = (
        key,
        V_initial,
        N_initial,
        jnp.bool_(False),
    )

    (_, V, N, failed), _ = jax.lax.scan(
        episode_step,
        initial_carry,
        xs=None,
        length=num_episodes,
    )

    return V, N, failed


def mc_prediction_jit(
    env,
    policy,
    key,
    num_episodes=10_000,
    gamma=1.0,
    visit_mode="first",
    max_steps=1000,
):
    """
    JIT-compiled Monte Carlo state-value prediction.

    Uses sample-average updates.

    Returns:
        V: (S,) estimated state values.
        N: (S,) update counts.
    """
    if visit_mode not in ("first", "every"):
        raise ValueError(
            "visit_mode must be 'first' or 'every'"
        )

    if num_episodes < 0:
        raise ValueError("num_episodes must be nonnegative")

    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    V, N, failed = _mc_prediction_compiled(
        env,
        policy,
        key,
        num_episodes,
        gamma,
        max_steps,
        visit_mode,
    )

    if bool(failed):
        raise RuntimeError(
            f"Episode did not terminate within {max_steps} steps"
        )

    return V, N
