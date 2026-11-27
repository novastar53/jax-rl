# jax-rl
[![tests](https://github.com/novastar53/jax-rl/actions/workflows/tests.yml/badge.svg)](https://github.com/novastar53/jax-rl/actions/workflows/tests.yml)

Reinforcement Learning Framework in Jax

## Tabular exploration and evaluation

`jax_rl.utils.exploration` provides JIT-compatible local UCB and Gaussian
Thompson action selection, a count-based uncertainty proxy, and the
propagated-uncertainty heuristic used in stochastic cliff-walking Experiment 8e.
The propagation rule is an exploration heuristic, not a theoretical UBE bound.
Ordinary and Double Q-learning are in `jax_rl.td.q_learning`; the Gaussian
weighted maximum and its estimator are in `jax_rl.td.gaussian_weighted_q`.

For model-based evaluation of a tabular policy, use
`jax_rl.dp.policy_evaluation.exact_discounted_policy_evaluation(P, R, policy,
gamma)`. This solves the discounted Bellman equations using transition
probabilities and conditional expected rewards. It is an evaluation utility;
agents do not need access to the transition model.
