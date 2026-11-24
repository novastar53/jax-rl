import jax
import jax.numpy as jnp

from jax_rl.envs.maximization_bias import MaximizationBias
from jax_rl.td.gaussian_weighted_q import (
    init_gaussian_weighted_q,
    update_gaussian_weighted_q,
    effective_sample_size,
    weighted_sample_variance,
    gaussian_weighted_max,
)


def epsilon_greedy_action(key, q_values, action_mask, epsilon):
    decision_key, random_key, greedy_key = jax.random.split(key, 3)

    masked_q = jnp.where(action_mask, q_values, -jnp.inf)
    greedy_mask = action_mask & (q_values == jnp.max(masked_q))

    random_action = jax.random.categorical(
        random_key, jnp.where(action_mask, 0.0, -jnp.inf)
    )
    greedy_action = jax.random.categorical(
        greedy_key, jnp.where(greedy_mask, 0.0, -jnp.inf)
    )

    explore = jax.random.bernoulli(decision_key, p=epsilon)
    return jnp.where(explore, random_action, greedy_action)


def train_episode(env, learner, key, epsilon=0.1, gamma=1.0):
    reset_key, loop_key = jax.random.split(key)
    env_state, s = env.reset(reset_key)

    def cond(carry):
        return ~carry[4]

    def body(carry):
        key, env_state, s, learner, _, steps, went_left, total_reward = carry

        key, action_key, reward_key, update_key = jax.random.split(key, 4)

        action = epsilon_greedy_action(
            action_key, learner.Q[s], env.action_mask(s), epsilon
        )

        b_key = jax.random.fold_in(action_key, 1)
        b_action = jax.random.randint(b_key, (), 0, 10)

        action = jnp.where(s == 1, b_action, action)

        next_env_state, next_s, reward, terminated, _ = env.step(
            reward_key, env_state, action
        )

        learner = update_gaussian_weighted_q(
            learner,
            update_key,
            s,
            action,
            reward,
            next_s,
            terminated,
            gamma=gamma,
            next_action_mask=env.action_mask(next_s),
        )

        went_left = jnp.where(steps == 0, action == 0, went_left)

        return (
            key,
            next_env_state,
            next_s,
            learner,
            terminated,
            steps + 1,
            went_left,
            total_reward + reward,
        )

    carry = (
        loop_key,
        env_state,
        s,
        learner,
        jnp.bool_(False),
        jnp.int32(0),
        jnp.bool_(False),
        jnp.float32(0),
    )

    _, _, _, learner, _, steps, went_left, total_reward = (
        jax.lax.while_loop(cond, body, carry)
    )

    return learner, went_left, total_reward, steps



def train_episodes(
    env,
    learner,
    key,
    num_episodes=1000,
    epsilon=0.1,
    gamma=1.0,
):
    def scan_step(carry, _):
        learner, key = carry
        key, episode_key = jax.random.split(key)

        learner, went_left, reward, steps = train_episode(
            env,
            learner,
            episode_key,
            epsilon=epsilon,
            gamma=gamma,
        )

        return (
            (learner, key),
            (went_left, reward, steps),
        )

    (learner, _), (went_left, rewards, steps) = jax.lax.scan(
        scan_step,
        (learner, key),
        xs=None,
        length=num_episodes,
    )

    return learner, went_left, rewards, steps



def train_many(env, key, num_runs=200, num_episodes=1000):
    keys = jax.random.split(key, num_runs)

    def run(key):
        learner = init_gaussian_weighted_q(
            env.num_states, env.num_actions
        )

        return train_episodes(
            env,
            learner,
            key,
            num_episodes=num_episodes,
            epsilon=0.1,
            gamma=1.0,
        )

    return jax.vmap(run)(keys)



if __name__ == "__main__":
    env = MaximizationBias()

    train = jax.jit(
        lambda key: train_many(
            env, key, num_runs=200, num_episodes=5000
        )
    )

    learners, went_left, rewards, steps = train(
        jax.random.key(42)
    )

    print("Left, first 100 episodes:",
          100 * went_left[:, :100].mean())

    print("Left, last 100 episodes:",
          100 * went_left[:, -100:].mean())

    print("Mean Q(A):", learners.Q[:, 0, :2].mean(axis=0))
    print("Std Q(A):", learners.Q[:, 0, :2].std(axis=0))

    print("Mean reward:", rewards.mean())


    b_q = learners.Q[:, 1, :]
    b_n = learners.N[:, 1, :]

    print("Mean max Q(B):", b_q.max(axis=1).mean())
    print("Mean Q(B):", b_q.mean())
    print("Mean visits to B:", b_n.sum(axis=1).mean())
    print(
        "Mean actions with fewer than 2 visits:",
        (b_n < 2).sum(axis=1).mean(),
    )

    b_q = learners.Q[:, 1, :]
    b_n = learners.N[:, 1, :]

    variance = weighted_sample_variance(
        learners.variance, learners.ess
    )[:, 1, :]

    ess = effective_sample_size(learners.ess)[:, 1, :]

    se = jnp.sqrt(variance / jnp.maximum(ess, 1.0))
    se = jnp.where(b_n < 2, 1.0, se)

    keys = jax.random.split(jax.random.key(123), b_q.shape[0])

    weighted_values, _ = jax.vmap(
        lambda key, q, uncertainty: gaussian_weighted_max(
            key, q, uncertainty, num_samples=1000
        )
    )(keys, b_q, se)

    print("Mean Q(B):", b_q.mean())
    print("Mean max Q(B):", b_q.max(axis=1).mean())
    print("Mean weighted Q(B):", weighted_values.mean())
    print("Mean standard error:", se.mean())
    print("Mean Q(A, Left):", learners.Q[:, 0, 0].mean())


    left = learners.Q[:, 0, 0]
    right = learners.Q[:, 0, 1]

    greedy_left = jnp.where(
        left > right,
        1.0,
        jnp.where(left == right, 0.5, 0.0),
    )

    print(
        "Runs preferring Left:",
        100 * (left > right).mean(),
    )

    print(
        "Final epsilon-greedy Left rate:",
        100 * (0.05 + 0.9 * greedy_left.mean()),
    )

    print(
        "Q(A, Left) quantiles:",
        jnp.quantile(left, jnp.array([0.05, 0.5, 0.95])),
    )
