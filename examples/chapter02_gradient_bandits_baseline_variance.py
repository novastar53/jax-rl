import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


# Configuration
k = 10
num_samples = 100_000
reward_means = [0.0, 2.0, 4.0, 8.0]

key = jax.random.key(42)

# Fix a single bandit problem
key, env_key = jax.random.split(key)
q_base = jax.random.normal(env_key, shape=(k,))

# Fix the policy: all preferences initially zero
H = jnp.zeros(k)
probs = jax.nn.softmax(H)

# Generate shared action and reward randomness
key, action_key, noise_key = jax.random.split(key, 3)

actions = jax.random.choice(
    action_key,
    a=k,
    shape=(num_samples,),
    p=probs,
)

noise = jax.random.normal(noise_key, (num_samples,))

# Score-function gradients: ∇ log π(A)
one_hot = jax.nn.one_hot(actions, k)
score_gradients = one_hot - probs


def measure_gradient_variance(mu, use_baseline):
    # Same underlying action values, shifted by mu
    q_star = q_base + mu

    # Observed rewards
    rewards = q_star[actions] + noise

    # Exact expected reward under the fixed policy
    baseline = jnp.dot(probs, q_star) if use_baseline else 0.0

    # Stochastic gradient estimates
    gradients = (
        (rewards - baseline)[:, None] * score_gradients
    )

    # Mean gradient
    mean_gradient = jnp.mean(gradients, axis=0)

    # Total variance: sum of component variances
    total_variance = jnp.sum(
        jnp.var(gradients, axis=0)
    )

    return float(total_variance), np.asarray(mean_gradient)


def main():
    results = {
        "With baseline": [],
        "Without baseline": [],
    }

    print(f"{'μ':<6} {'Baseline':<18} {'Variance':<15} {'Gradient norm'}")
    print("-" * 60)

    for mu in reward_means:
        for use_baseline in [True, False]:
            variance, mean_grad = measure_gradient_variance(
                mu, use_baseline
            )

            name = (
                "With baseline"
                if use_baseline
                else "Without baseline"
            )

            results[name].append(variance)

            print(
                f"{mu:<6.1f} "
                f"{name:<18} "
                f"{variance:<15.4f} "
                f"{np.linalg.norm(mean_grad):.4f}"
            )

    # Plot gradient variance
    plt.figure(figsize=(9, 5))

    for name, variances in results.items():
        plt.plot(
            reward_means,
            variances,
            marker="o",
            label=name,
        )

    plt.xlabel("Reward offset (μ)")
    plt.ylabel("Total gradient variance")
    plt.title("Effect of Reward Offset on Policy-Gradient Variance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
