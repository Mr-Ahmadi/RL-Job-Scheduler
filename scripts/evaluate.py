"""Reproducible evaluation of all scheduling policies on the 20 held-out job sets.

Covers the random and greedy (max-throughput / max-total oracle) baselines, the
trained inner PPO agent on its own (greedy and top-K), the oracle-free learned
top-K search (``nero/search/topk.py``), the continually-updated online policy
(``scripts/online_learning.py``), and the full two-tier system.  The Sia baselines live
in ``scripts/sia_baseline.py``.

``ppo_topk_k*`` reads the true reward at decision time and is therefore an
offline upper bound; ``ppo_topk_learned_k*`` is the deployable version of the
same search, scoring candidates with learned heads only.

All numbers here are on the 20 test sets in ``data/saved_job_sets/``, which nothing is
trained or tuned on.  ``--validation N`` instead scores a few policies on randomly
generated job sets; that mode exists for choosing between configurations and its
numbers are not reported as results.

Every policy is evaluated on a *fresh* eval environment for exactly 20 episodes,
so each episode consumes one distinct job set (set_000.json .. set_019.json) in
deterministic order. Results are written to results/job_scheduling/ and compared
with any previously committed scores.
"""
import json
import os

import numpy as np
import torch

from nero.agents.loading import load_inner_agent
from nero.evaluation import (EPISODES, evaluate, greedy_max_throughput, greedy_max_total,
                             make_learned_topk, make_ppo_greedy, make_random, make_topk)
from nero.agents.ppo import PPOAgent, flatten_obs, device
from nero.envs.job_scheduling.eval import Eval_JobSchedulingEnv
from nero.paths import HEADS, INNER, ONLINE, OUTER, SCORES, TEST_SETS

OUT = str(SCORES)
LEARNED_DIR = str(HEADS)


def evaluate_online_policy(directory=str(ONLINE)):
    """The policy left behind by ``scripts/online_learning.py`` (greedy, no search)."""

    agent, _, _ = load_inner_agent(device=device)
    agent.actor.load_state_dict(torch.load(f"{directory}/actor.pth", map_location=device))
    agent.critic.load_state_dict(torch.load(f"{directory}/critic.pth", map_location=device))
    agent.actor.eval()
    agent.critic.eval()
    return evaluate(make_ppo_greedy(agent))


def evaluate_two_tier_topk(secondary=False, set_dir=str(TEST_SETS), episodes=None):
    """Two-tier system with the oracle-free search doing the placements."""
    from nero.search.two_tier import LearnedTopK_SubsetSelectorEnv
    from nero.envs.subset_selector.common import flatten_obs_subset

    episodes = EPISODES if episodes is None else episodes
    cpu_agent, _, _ = load_inner_agent(device="cpu")
    env = LearnedTopK_SubsetSelectorEnv(set_dir, agent=cpu_agent, secondary=secondary)
    obs, _ = env.reset()

    agent = PPOAgent(len(flatten_obs_subset(obs)), 2, lr_actor=3e-4, lr_critic=1e-3,
                     gamma=0.99, lamda=0.95, clip=0.2, epochs=8, batch_size=64,
                     checkpoint_dir=str(OUTER))
    agent.load()
    agent.actor.eval()
    agent.critic.eval()

    def greedy(state):
        st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, probs = agent.actor(st)
        mask = torch.tensor([[1.0, 1.0] if state[-1] > 0.5 else [1.0, 0.0]],
                            dtype=torch.float32, device=device)
        m = probs * mask
        return int(torch.argmax(m / m.sum(dim=-1, keepdim=True)).item())

    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        st = flatten_obs_subset(obs)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(greedy(st))
            done = term or trunc
            st = flatten_obs_subset(obs) if not done else st
        rewards.append(info["total_reward"])
    return rewards


def evaluate_two_tier():
    """Full two-tier system: frozen inner placement agent + trained outer agent.

    The outer agent decides *whether* to duplicate each job; the fine-tuned
    secondary inner agent decides *where* the duplicate goes.
    """
    from nero.envs.subset_selector.eval import Eval_SubsetSelectorEnv
    from nero.envs.subset_selector.common import flatten_obs_subset

    env = Eval_SubsetSelectorEnv(str(TEST_SETS))
    obs, _ = env.reset()
    state_dim = len(flatten_obs_subset(obs))

    agent = PPOAgent(state_dim, 2, lr_actor=3e-4, lr_critic=1e-3,
                     gamma=0.99, lamda=0.95, clip=0.2, epochs=8, batch_size=64,
                     checkpoint_dir=str(OUTER))
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


def _two_tier_scores(env_factory, sets, key, n):
    """Run a two-tier env over ``sets``, pairing each episode back by job list."""
    from nero.envs.subset_selector.common import flatten_obs_subset

    env = env_factory()
    obs, _ = env.reset()
    agent = PPOAgent(len(flatten_obs_subset(obs)), 2, lr_actor=3e-4, lr_critic=1e-3,
                     gamma=0.99, lamda=0.95, clip=0.2, epochs=8, batch_size=64,
                     checkpoint_dir=str(OUTER))
    agent.load()
    agent.actor.eval()
    agent.critic.eval()

    def greedy(state):
        st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, probs = agent.actor(st)
        mask = torch.tensor([[1.0, 1.0] if state[-1] > 0.5 else [1.0, 0.0]],
                            dtype=torch.float32, device=device)
        m = probs * mask
        return int(torch.argmax(m / m.sum(dim=-1, keepdim=True)).item())

    scores = {}
    for _ in range(n):
        obs, _ = env.reset()
        originals = tuple(m for (m, _s), d in zip(env.env.problem.jobs,
                                                  env.env.duplicate_of) if d == -1)
        idx = key.get(originals)
        st = flatten_obs_subset(obs)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(greedy(st))
            done = term or trunc
            st = flatten_obs_subset(obs) if not done else st
        if idx is not None and idx not in scores:
            scores[idx] = info["total_reward"]
    if len(scores) != n:
        return None
    return np.array([scores[i] for i in range(n)])


def evaluate_two_tier_fresh(sets):
    """Two-tier system on the fresh sets, paired back to them by content.

    ``Eval_SubsetSelectorEnv`` only reads job sets from a directory and consumes
    one extra on construction, so the sets are written out, and each episode is
    matched to its set by the job list it actually scheduled rather than by
    position.
    """
    import tempfile

    from nero.envs.subset_selector.eval import Eval_SubsetSelectorEnv

    with tempfile.TemporaryDirectory() as tmp:
        key = {}
        for i, jl in enumerate(sets):
            names = [m for (m, _) in jl]
            json.dump(names, open(os.path.join(tmp, f"set_{i:03d}.json"), "w"))
            key[tuple(names)] = i

        out = {}
        out["subset_selector"] = _two_tier_scores(
            lambda: Eval_SubsetSelectorEnv(tmp), sets, key, len(sets))

        if os.path.exists(f"{LEARNED_DIR}/reward_model.pth"):
            from nero.search.two_tier import LearnedTopK_SubsetSelectorEnv
            cpu_agent, _, _ = load_inner_agent(device="cpu")
            out["subset_selector_topk"] = _two_tier_scores(
                lambda: LearnedTopK_SubsetSelectorEnv(tmp, agent=cpu_agent),
                sets, key, len(sets))
            out["subset_selector_topk_both"] = _two_tier_scores(
                lambda: LearnedTopK_SubsetSelectorEnv(tmp, agent=cpu_agent,
                                                      secondary=True),
                sets, key, len(sets))
    return {k: v for k, v in out.items() if v is not None}


def evaluate_validation(n_sets=40, seed=20260920):
    """Model selection on randomly generated job sets -- never on the test sets.

    The 20 sets in ``data/saved_job_sets/`` are the *test* set: everything is reported
    on them and nothing is tuned on them.  Any choice between configurations --
    the learned top-K's K, its scoring mode, its canonicalisation mode -- is made
    here instead, on job sets drawn from the training distribution with a seed
    used for no training and no reporting.

    This is a selection tool, not a second benchmark; its numbers are not the
    headline results.
    """
    import random as _random

    from nero.envs.problem import JobTable
    from nero.envs.job_scheduling.train import Train_JobSchedulingEnv

    rng = _random.Random(seed)
    types = [(t.model, 1) for t in JobTable]
    sets = [rng.choices(types, k=rng.randint(20, 90)) for _ in range(n_sets)]

    cpu_agent, _, _ = load_inner_agent(device="cpu")
    env = Train_JobSchedulingEnv()

    def run(policy_fn):
        out = []
        for jl in sets:
            env.reset(list(jl))
            total, done = 0.0, False
            while not done:
                a = policy_fn(env)
                if a is None:
                    break
                _, rew, term, trunc, _ = env.step(a)
                total += sum(rew)
                done = term or trunc
            out.append(total)
        return np.array(out)

    arms = [
        ("ppo", make_ppo_greedy(cpu_agent)),
        ("gavel_max_total", greedy_max_total),
        ("ppo_topk_k5", make_topk(cpu_agent, 5)),
        ("ppo_topk_learned_k5", make_learned_topk(5, agent=cpu_agent)),
        ("ppo_topk_learned_k3", make_learned_topk(3, agent=cpu_agent)),
        ("ppo_topk_learned_k5_expand",
         make_learned_topk(5, canonical="expand", agent=cpu_agent)),
        ("ppo_topk_learned_k5_reward_value",
         make_learned_topk(5, mode="reward_value", agent=cpu_agent)),
        ("ppo_topk_learned_k5_blend", make_learned_topk(5, mode="blend", agent=cpu_agent)),
        ("ppo_topk_learned_k5_q", make_learned_topk(5, mode="q", agent=cpu_agent)),
    ]
    res = {}
    for name, fn in arms:
        res[name] = run(fn)
        print(f"{name:28s} mean={res[name].mean():8.4f} ± {res[name].std(ddof=1):6.4f} "
              f"n={n_sets}")

    for name, vals in evaluate_two_tier_fresh(sets).items():
        res[name] = vals
        print(f"{name:28s} mean={vals.mean():8.4f} ± {vals.std(ddof=1):6.4f} n={n_sets}")

    print("\npaired against ppo_topk_learned_k5:")
    for other in ("subset_selector_topk_both", "subset_selector_topk", "ppo_topk_k5",
                  "subset_selector", "gavel_max_total", "ppo"):
        if other not in res:
            continue
        d = res["ppo_topk_learned_k5"] - res[other]
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  vs {other:20s} {d.mean():+.4f}  paired SE {se:.4f}  t={d.mean() / se:+.2f}")

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/validation_scores.json", "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in res.items()}, f)
    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validation", type=int, nargs="?", const=40, default=None,
                    metavar="N",
                    help="model selection on N randomly generated job sets instead "
                         "of reporting on the 20 test sets; used to choose between "
                         "configurations without touching the test set")
    cli = ap.parse_args()
    if cli.validation:
        evaluate_validation(cli.validation)
        raise SystemExit(0)

    eval_env = Eval_JobSchedulingEnv(str(TEST_SETS))
    obs, _ = eval_env.reset()
    state_dim = len(flatten_obs(obs))
    action_dim = eval_env.S * eval_env.A

    agent = PPOAgent(state_dim, action_dim, lr_actor=5e-4, lr_critic=1e-3,
                     gamma=0.99, lamda=0.95, clip=0.2, epochs=10, batch_size=128,
                     checkpoint_dir=str(INNER))
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

    # --- oracle-free policies (nothing below reads the throughput table) ---
    DEFAULT_DIR = LEARNED_DIR
    if os.path.exists(f"{DEFAULT_DIR}/reward_model.pth"):
        cpu_agent, _, _ = load_inner_agent(device="cpu")
        for k in (3, 5):
            report(f"ppo_topk_learned_k{k}",
                   evaluate(make_learned_topk(k, agent=cpu_agent)))
        # ablations: how the candidate set is built, and what scores it
        report("ppo_topk_learned_k5_nocanon",
               evaluate(make_learned_topk(5, canonical="none", agent=cpu_agent)))
        report("ppo_topk_learned_k5_expand",
               evaluate(make_learned_topk(5, canonical="expand", agent=cpu_agent)))
        for mode in ("reward_value", "q", "blend"):
            report(f"ppo_topk_learned_k5_{mode}",
                   evaluate(make_learned_topk(5, mode=mode, agent=cpu_agent)))
        # the two-tier system with the search doing the placements
        report("subset_selector_topk", evaluate_two_tier_topk(secondary=False))
        report("subset_selector_topk_both", evaluate_two_tier_topk(secondary=True))
    else:
        print(f"(skipping learned top-K: run `python -m scripts.train_reward_model` "
              f"to create {DEFAULT_DIR}/)")

    if os.path.exists(str(ONLINE / "actor.pth")):
        report("ppo_online", evaluate_online_policy())
    else:
        print("(skipping ppo_online: run `python -m scripts.online_learning` first)")
