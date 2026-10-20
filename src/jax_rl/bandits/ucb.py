import jax
import jax.numpy as jnp

class UCBPolicy:
    def __init__(self, c=2.0):
        self.c = c


    def select_action(self, key, Q, N, t):
        k = Q.shape[0]
        bonus = jnp.where(
            N == 0,
            jnp.inf,
            self.c * jnp.sqrt(jnp.log(jnp.maximum(t, 1)) / jnp.maximum(N, 1))
        )
        scores = Q + bonus
        best_mask = scores == jnp.max(scores)
        probs = best_mask / jnp.sum(best_mask)
        action = jax.random.choice(key, a=k, p=probs)
        return action


