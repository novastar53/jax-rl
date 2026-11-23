"""American roulette as the continuing MDP from the Double Q-learning paper.

The default has 170 wagers plus one stop action, as in van Hasselt (2010).
Of the wagers, 160 are conventional roulette placements and 10 are extra
*synthetic* two-pocket wagers to reproduce the paper's action-space size.
They have the same -2/38 expected net reward as ordinary wagers; the exact
170-bet menu used by the paper is not specified in the available description.

Payouts are *net* profits on a $1 stake: losing = -1, winning a single
number = +35. The five-number top line has net odds 6:1.
"""

import jax
import jax.numpy as jnp

from .base import Environment


RED = frozenset({
    1, 3, 5, 7, 9, 12, 14, 16, 18,
    19, 21, 23, 25, 27, 30, 32, 34, 36,
})


def _standard_bets():
    bets = []

    def add(label, pockets, gross_return=None):
        pockets = tuple(sorted(pockets))
        if gross_return is None:
            gross_return = 36.0 / len(pockets)
        bets.append((label, pockets, float(gross_return)))

    for n in range(38):
        add(f"straight {n if n < 37 else '00'}", [n])

    for row in range(12):
        base = 3 * row + 1
        add(f"split {base}-{base+1}", [base, base + 1])
        add(f"split {base+1}-{base+2}", [base + 1, base + 2])
    for row in range(11):
        for col in range(3):
            n = 3 * row + 1 + col
            add(f"split {n}-{n+3}", [n, n + 3])
    for a, b in [(0, 37), (0, 1), (0, 2), (37, 2), (37, 3)]:
        add(f"split {a}-{b}", [a, b])

    for row in range(12):
        base = 3 * row + 1
        add(f"street {base}-{base+2}", range(base, base + 3))
    add("trio 0-1-2", [0, 1, 2])
    add("trio 00-2-3", [37, 2, 3])

    for row in range(11):
        for col in range(2):
            n = 3 * row + 1 + col
            add(f"corner {n}", [n, n+1, n+3, n+4])

    for row in range(11):
        base = 3 * row + 1
        add(f"six-line {base}-{base+5}", range(base, base+6))

    for group in range(3):
        base = 12 * group + 1
        add(f"dozen {group+1}", range(base, base + 12))
    for col in range(3):
        add(f"column {col+1}", range(1 + col, 37, 3))

    add("red", RED)
    add("black", set(range(1, 37)) - RED)
    add("odd", range(1, 37, 2))
    add("even", range(2, 37, 2))
    add("low", range(1, 19))
    add("high", range(19, 37))

    add("five-number top line", [0, 37, 1, 2, 3], gross_return=7.0)
    assert len(bets) == 160
    return bets


class Roulette(Environment):
    def __init__(self, num_bets=170, stake=1.0):
        if not 1 <= num_bets <= 170:
            raise ValueError("num_bets must be between 1 and 170")
        if stake <= 0:
            raise ValueError("stake must be positive")

        bets = _standard_bets()
        # Ten additional pocket-pair actions make the benchmark 170 wagers.
        # These are synthetic, not standard betting-table placements.
        for n in range(4, 14):
            bets.append((f"synthetic split 0-{n}", (0, n), 18.0))
        bets = bets[:num_bets]

        self.num_bets = num_bets
        self.num_actions = num_bets + 1
        self.num_states = 2  # betting, absorbing terminal
        self.start_state = 0
        self.terminal_state = 1
        self.stop_action = num_bets
        self.stake = float(stake)
        self.bet_labels = tuple(bet[0] for bet in bets)

        self.bet_mask = jnp.asarray([
            [pocket in bet[1] for pocket in range(38)] for bet in bets
        ])
        self.gross_returns = jnp.asarray([bet[2] for bet in bets])
        self.expected_rewards = self.stake * (
            self.bet_mask.sum(axis=1) * self.gross_returns / 38.0 - 1.0
        )

    def reset(self, key):
        state = jnp.int32(self.start_state)
        return state, state

    def _payouts(self, pockets):
        """Vector of net rewards for each betting action at provided pockets."""
        winner = self.bet_mask[jnp.arange(self.num_bets), pockets]
        return self.stake * jnp.where(winner, self.gross_returns - 1.0, -1.0)

    def step(self, key, env_state, action):
        pocket = jax.random.randint(key, (), 0, 38)
        bet_index = jnp.minimum(action, self.num_bets - 1)
        winner = self.bet_mask[bet_index, pocket]
        bet_reward = self.stake * jnp.where(
            winner, self.gross_returns[bet_index] - 1.0, -1.0
        )
        was_terminal = env_state == self.terminal_state
        stop = action == self.stop_action
        terminated = was_terminal | stop
        next_state = jnp.where(terminated, self.terminal_state, self.start_state)
        reward = jnp.where(terminated, 0.0, bet_reward)
        return next_state, next_state, reward, terminated, {"pocket": pocket}

    def sample_all_rewards(self, key, env_state, shared_spin=False):
        """Full-information reward samples for *all* actions.

        By default, different bets use independent simulated wheel spins.
        Set shared_spin=True to use one wheel result for every action.
        Both modes update every bet per iteration (not online interaction).
        """
        if shared_spin:
            pockets = jnp.full((self.num_bets,), jax.random.randint(key, (), 0, 38))
        else:
            pockets = jax.random.randint(key, (self.num_bets,), 0, 38)
        bet_rewards = self._payouts(pockets)
        rewards = jnp.concatenate([bet_rewards, jnp.zeros((1,), dtype=bet_rewards.dtype)])
        return jnp.where(env_state == self.terminal_state, jnp.zeros_like(rewards), rewards)

    def transition_model(self):
        """P[s,a,s'], conditional expected reward R[s,a,s']."""
        P = jnp.zeros((self.num_states, self.num_actions, self.num_states))
        R = jnp.zeros_like(P)
        P = P.at[0, :self.num_bets, 0].set(1.0)
        P = P.at[0, self.stop_action, 1].set(1.0)
        P = P.at[1, :, 1].set(1.0)
        R = R.at[0, :self.num_bets, 0].set(self.expected_rewards)
        return P, R
