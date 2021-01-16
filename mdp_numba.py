import numpy as np
from numba import njit
from numba.typed import List

"""
Please read README.md
"""


@njit
def _gen_action(n_states, n_outcomes):
    # action = list of outcomes
    weight = np.empty(n_outcomes)
    for i in range(n_outcomes):
        weight[i] = np.random.randint(1, 5)
    weight /= weight.sum()
    ans = List()
    for i in range(n_outcomes):
        # outcome = weight, reward, state
        ans.append((weight[i], np.random.rand() * 2 - 1, np.random.randint(n_states)))
    return ans


@njit
def gen_actions(n_states, n_actions, n_outcomes):
    """
    Generate a random MDP, used mainly for example and tweaking purposes
    """
    # actions = list of list of actions
    ans = List()
    for _ in range(n_states):
        cur = List()
        for _ in range(n_actions):
            cur.append(_gen_action(n_states, n_outcomes))
        ans.append(cur)
    return ans


@njit
def sample_policy(actions):
    """
    Select one action per state uniformly at random
    """
    ans = List()
    for i in range(len(actions)):
        ans.append(actions[i][np.random.randint(len(actions[i]))])
    return ans


@njit
def deterministic_q(policy, discount, iters):
    """
    Given a deterministic policy, a discount factor and a number of iterations, compute the Q-value of each state
    A policy is list containing the action of each state
    """
    n = len(policy)
    q = np.zeros(n)
    for _ in range(iters):
        newq = np.zeros(n)
        for s, outcomes in enumerate(policy):
            for p, r, ss in outcomes:
                newq[s] += p * (r + discount * q[ss])
        q = newq
    return q


@njit
def compute_q(actions, policy, discount, iters):
    """
    Compute the Q-value for each state and each action
    Returns a list of lists of float
    q[i][j] is the Q-value at state i of action j
    """
    q = deterministic_q(policy, discount, iters)
    out = List()
    for s in range(len(actions)):
        cur = List()
        for ia, action in enumerate(actions[s]):
            acc = 0.0
            for p, r, ss in action:
                acc += p * (r + discount * q[ss])
            cur.append(acc)
        out.append(cur)
    return out


def to_numba(container):
    """
    Recursively convert list to numba.typed.List
    """
    if not isinstance(container, list):
        return container
    ans = List()
    for x in container:
        ans.append(to_numba(x))
    return ans


def from_numba(container):
    """
    Recursively convert numba.typed.List to list
    """
    if isinstance(container, List):
        return list(map(from_numba, container))
    return container

