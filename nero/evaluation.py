"""Policies and the episode runner every evaluation in the project shares.

This is library code: the evaluation entry point, the latency benchmark, the
reward-model trainer and the figure-data script all import from here, so it
lives in the package rather than in any one of them.

Contents:

* ``run_episode`` / ``evaluate`` -- one episode, or all 20 held-out test sets;
* ``true_score`` -- the environment's reward for a hypothetical placement, read
  from the throughput table (used only by the oracle baselines and by offline
  verification, never by a deployable policy);
* baselines -- random, and the two Gavel greedy oracles;
* policy factories -- greedy PPO, oracle top-K, and the oracle-free learned
  top-K search.
"""

import random

import torch

from nero.agents.ppo import flatten_obs
from nero.envs.job_scheduling.eval import Eval_JobSchedulingEnv
from nero.paths import HEADS, TEST_SETS

EPISODES = 20


def true_score(env, j, s, a):
    colocated = [oj for oj in range(env.J) if oj != j and env.assignment[oj, s, a] > 0]
    if len(colocated) >= 2:
        return -float("inf"), 0.0
    prev_tp = {oj: env._estimate_job_throughput_given_combination(oj, s, a) for oj in colocated}
    env.assignment[j, s, a] = 1
    new_tp = env._estimate_job_throughput_given_combination(j, s, a)
    upd_tp = {oj: env._estimate_job_throughput_given_combination(oj, s, a) for oj in colocated}
    env.assignment[j, s, a] = 0
    delta = sum((upd_tp[oj] - prev_tp[oj]) * (1 - env.get_distribution_discount(oj, s))
                for oj in colocated)
    return new_tp, delta


def run_episode(env, policy_fn):
    obs, _ = env.reset()
    done = False
    total = 0.0
    while not done:
        action = policy_fn(env)
        if action is None:
            break
        s, a = action
        obs, rew, term, trunc, _ = env.step((s, a))
        total += sum(rew)
        done = term or trunc
    return total


def evaluate(policy_fn):
    env = Eval_JobSchedulingEnv(TEST_SETS)
    return [run_episode(env, policy_fn) for _ in range(EPISODES)]


def valid_actions(env):
    """Feasible slots for the current job (see ``Base_JobSchedulingEnv``)."""
    return env.feasible_slots()


def agent_device(agent):
    """Where an agent's weights live, so callers never mix devices."""
    return next(agent.actor.parameters()).device


def greedy_max_total(env):
    j = env.current_job_idx
    if j >= env.J:
        return None
    best, best_score = None, -float("inf")
    for (s, a) in valid_actions(env):
        nt, d = true_score(env, j, s, a)
        if nt + d > best_score:
            best_score = nt + d
            best = (s, a)
    return best


def greedy_max_throughput(env):
    j = env.current_job_idx
    if j >= env.J:
        return None
    best, best_score = None, -float("inf")
    for (s, a) in valid_actions(env):
        nt, _ = true_score(env, j, s, a)
        if nt > best_score:
            best_score = nt
            best = (s, a)
    return best


def make_random(seed):
    rng = random.Random(seed)

    def policy(env):
        if env.current_job_idx >= env.J:
            return None
        valid = valid_actions(env)
        return rng.choice(valid) if valid else None
    return policy


def make_ppo_greedy(agent):
    dev = agent_device(agent)

    def policy(env):
        if env.current_job_idx >= env.J:
            return None
        st = torch.tensor(flatten_obs(env._get_obs()), dtype=torch.float32, device=dev).unsqueeze(0)
        with torch.no_grad():
            _, probs = agent.actor(st)
        mask = torch.zeros_like(probs)
        for (s, a) in valid_actions(env):
            mask[0, s * env.A + a] = 1
        masked = probs * mask
        masked = masked / masked.sum(dim=-1, keepdim=True)
        return divmod(int(torch.argmax(masked, dim=-1).item()), env.A)
    return policy


def make_topk(agent, k):
    dev = agent_device(agent)

    def policy(env):
        if env.current_job_idx >= env.J:
            return None
        st = torch.tensor(flatten_obs(env._get_obs()), dtype=torch.float32, device=dev).unsqueeze(0)
        with torch.no_grad():
            _, probs = agent.actor(st)
        mask = torch.zeros_like(probs)
        for (s, a) in valid_actions(env):
            mask[0, s * env.A + a] = 1
        masked = probs * mask
        cands = torch.topk(masked.flatten(), k).indices.tolist()
        best, best_score = None, -float("inf")
        for a_idx in cands:
            s, a = divmod(int(a_idx), env.A)
            nt, d = true_score(env, env.current_job_idx, s, a)
            if nt == -float("inf"):
                continue
            if nt + d > best_score:
                best_score = nt + d
                best = (s, a)
        return best
    return policy


def make_learned_topk(k, mode="reward", beta=0.5, canonical="dedupe", agent=None,
                      learned_dir=None):
    """Oracle-free top-K: policy proposal + canonicalisation + learned scoring."""
    from nero.search.topk import LearnedTopKActor

    actor = LearnedTopKActor(k=k, mode=mode, beta=beta, canonical=canonical,
                             learned_dir=learned_dir or HEADS,
                             device="cpu", threads=0, agent=agent)
    return actor.decide
