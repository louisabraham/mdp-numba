# MDP-numba

A very simple implementation of Markov decision processes

## Representation

Markov decision processes are represented in the following way:

    - state: integer from [0, n-1]
    - outcome: tuple (probability, reward, next state)
    - action: list of outcomes such that the probabilities sum to 1
    - MDP: list of list of actions. MDP[i] is the list of actions that can be taken from state i.
    - deterministic policy: list of actions. policy[i] is the action taken at state i.


Here is what a MDP with 3 states, 2 actions per state and 2 possible outcomes per action looks like:

```
[[[(0.25, 0.3179936561068226, 2), (0.75, -0.8796836837078306, 2)],
  [(0.5, 0.975312945360006, 0), (0.5, 0.5480562187562397, 0)]],
 [[(0.25, -0.4749949879945312, 0), (0.75, -0.7877396315424006, 2)],
  [(0.5714285714285714, 0.4294856213773759, 2),
   (0.42857142857142855, 0.1974254354972962, 1)]],
 [[(0.3333333333333333, 0.4660143251911333, 2),
   (0.6666666666666666, 0.658637696587731, 2)],
  [(0.6666666666666666, 0.03194537372232942, 2),
   (0.3333333333333333, 0.48849384151314523, 1)]]]
```

## API

### Q-value computation

- `deterministic_q(policy, discount, iters)`: Given a deterministic policy, a discount factor and a number of iterations, compute the Q-value of each state
- `compute_q(actions, policy, discount, iters)` Compute the Q-value for each state and each action

### Utility functions

- `gen_actions(n_states, n_actions, n_outcomes)`: Generate a random MDP, used mainly for example and tweaking purposes
- `sample_policy(actions)`: Select one action per state uniformly at random
- `to_numba`, `from_numba`: Recursively convert back and forth from numba.typed.List to Python's pure lists


## Example

```python
from mdp_numba import gen_actions, sample_policy, compute_q, from_numba

mdp = gen_actions(n_states=5, n_actions=2, n_outcomes=2)
q = compute_q(mdp, policy=sample_policy(mdp), discount=0.9, iters=200)
print(from_numba(q))
```


## Improvements

I coded this quickly for a specific purpose, feel free to modify this code. Some ideas:

- implement a different stopping criterion in `deterministic_q`, for example based on the absolute difference between iterations. 
- to go even faster, you could try `njit(fastmath=True)`

