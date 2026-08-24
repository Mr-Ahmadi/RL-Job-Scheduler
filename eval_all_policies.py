"""Reproducible evaluation of all scheduling policies on the 20 held-out job sets.

Covers the random and greedy (max-throughput / max-total oracle) baselines, the
trained inner PPO agent on its own (greedy and top-K), and the full two-tier
system.  The Sia baselines live in ``sia_baseline.py``.

Every policy is evaluated on a *fresh* eval environment for exactly 20 episodes,
so each episode consumes one distinct job set (set_000.json .. set_019.json) in
deterministic order. Results are written to curves/job_scheduling/ and compared
with any previously committed scores.
"""
import json
import os
import random
import numpy as np
import torch

from environment.ppo.core import PPOAgent, flatten_obs, device
from environment.job_scheduling.eval import Eval_JobSchedulingEnv

OUT = "curves/job_scheduling"
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
    env = Eval_JobSchedulingEnv("saved_job_sets")
    return [run_episode(env, policy_fn) for _ in range(EPISODES)]


def valid_actions(env):
    j = env.current_job_idx
    out = []
    for s in range(env.S):
        for a in range(env.A):
            col = [jj for jj in range(env.J) if jj != j and env.assignment[jj, s, a] > 0]
            if len(col) < 2:
                out.append((s, a))
    return out


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
    def policy(env):
        if env.current_job_idx >= env.J:
            return None
        st = torch.tensor(flatten_obs(env._get_obs()), dtype=torch.float32, device=device).unsqueeze(0)
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
    def policy(env):
        if env.current_job_idx >= env.J:
            return None
        st = torch.tensor(flatten_obs(env._get_obs()), dtype=torch.float32, device=device).unsqueeze(0)
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


def evaluate_two_tier():
    """Full two-tier system: frozen inner placement agent + trained outer agent.

    The outer agent decides *whether* to duplicate each job; the fine-tuned
    secondary inner agent decides *where* the duplicate goes.
    """
    from environment.subset_selector.eval import Eval_SubsetSelectorEnv
    from environment.subset_selector._requirements import flatten_obs_subset

    env = Eval_SubsetSelectorEnv("saved_job_sets")
    obs, _ = env.reset()
    state_dim = len(flatten_obs_subset(obs))

    agent = PPOAgent(state_dim, 2, lr_actor=3e-4, lr_critic=1e-3,
                     gamma=0.99, lamda=0.95, clip=0.2, epochs=8, batch_size=64,
                     checkpoint_dir="models/subset_selector/ppo")
    agent.load()
    agent.actor.eval()
    agent.critic.eval()

    def greedy(state):
        st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, probs = agent.actor(st)
        can_dup = state[-1] > 0.5
        mask = torch.tensor([[1.0, 1.0] if can_dup else [1.0, 0.0]],
                            dtype=torch.float32, device=device)
        masked = probs * mask
        masked = masked / masked.sum(dim=-1, keepdim=True)
        return int(torch.argmax(masked).item())

    rewards = []
    for _ in range(EPISODES):
        obs, _ = env.reset()
        st = flatten_obs_subset(obs)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(greedy(st))
            done = term or trunc
            st = flatten_obs_subset(obs) if not done else st
        rewards.append(info["total_reward"])
    return rewards


def report(name, rewards):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/evaluation_scores_{name}.json"
    ref = None
    if os.path.exists(path):
        ref = json.load(open(path))
    with open(path, "w") as f:
        json.dump(rewards, f)
    m, s = np.mean(rewards), np.std(rewards)
    msg = f"{name:22s} mean={m:8.4f} ± {s:6.4f} (pop)  n={len(rewards)}"
    if ref is not None and len(ref) == len(rewards):
        diff = m - np.mean(ref)
        msg += f"  vs-committed diff={diff:+.4f}"
    print(msg)


if __name__ == "__main__":
    eval_env = Eval_JobSchedulingEnv("saved_job_sets")
    obs, _ = eval_env.reset()
    state_dim = len(flatten_obs(obs))
    action_dim = eval_env.S * eval_env.A

    agent = PPOAgent(state_dim, action_dim, lr_actor=5e-4, lr_critic=1e-3,
                     gamma=0.99, lamda=0.95, clip=0.2, epochs=10, batch_size=128,
                     checkpoint_dir="models/job_scheduling/ppo")
    agent.load()
    agent.actor.eval()
    agent.critic.eval()

    torch.manual_seed(42)

    report("gavel_max_total", evaluate(greedy_max_total))
    report("gavel_max_throughput", evaluate(greedy_max_throughput))
    report("random", evaluate(make_random(42)))
    report("ppo", evaluate(make_ppo_greedy(agent)))
    report("ppo_topk_k3", evaluate(make_topk(agent, 3)))
    report("ppo_topk_k5", evaluate(make_topk(agent, 5)))
    report("subset_selector", evaluate_two_tier())
