"""Bootstrap the top-K reward model online, the way a real deployment would.

``scripts/train_reward_model.py`` fits the reward model offline against the full Gavel
throughput tables: every visited state is labelled at *all* ~45 feasible slots,
and every job type is seen. That is a fair model of a cluster that has been
profiled in advance, but it is the most generous assumption in this project.

This script removes it. The PPO policy is trained offline as usual (it never
needs the tables at decision time), and then the reward model starts from either

  ``--init cold``     random weights -- the cluster has never been profiled; or
  ``--init partial``  weights fitted offline on a *subset* of job families, with
                      the rest never seen (``scripts/train_reward_model.py --exclude-models``)

and learns on live traffic under deployment rules:

  * one label per decision -- the realised reward of the slot actually played,
    never the ~45 counterfactuals the offline trainer enjoys;
  * no throughput-table access at decision time (the search is the same one
    ``scripts/benchmark.py`` guards);
  * the PPO policy stays frozen throughout, so the curve measures the reward
    model alone.

The deployment stream is randomly generated job sets. Evaluation is the project
standard: the 20 held-out sets in ``data/saved_job_sets/``, greedy, and those episodes
are never learned from.

Usage:
    python -m scripts.online_reward_model --init cold
    python -m scripts.online_reward_model --init partial --learned-dir models/job_scheduling/learned_topk_partial
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from nero.agents.loading import load_inner_agent
from nero.search.canonicalization import feature_dim, slot_features
from nero.search.heads import DEFAULT_DIR, SlotRewardModel
from nero.search.topk import LearnedTopKActor
from nero.envs.job_scheduling.eval import Eval_JobSchedulingEnv
from nero.envs.job_scheduling.train import Train_JobSchedulingEnv
from nero.paths import MODELS, SCORES, TEST_SETS

CURVE_DIR = str(SCORES)


class LabelBuffer:
    """Realised (features, reward) pairs -- one per decision, as online allows."""

    def __init__(self, capacity, fdim):
        self.feat = np.zeros((capacity, fdim), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.pos = 0
        self.capacity = capacity
        self.seen = 0

    def add(self, feat, reward):
        i = self.pos
        self.feat[i] = feat
        self.reward[i] = reward
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.seen += 1

    def sample(self, n, rng):
        idx = rng.integers(0, self.size, size=min(n, self.size))
        return self.feat[idx], self.reward[idx]


def evaluate(actor, episodes=20):
    """Project-standard test: the 20 held-out sets, greedy, no learning."""
    from nero.evaluation import run_episode

    env = Eval_JobSchedulingEnv(str(TEST_SETS))
    return [run_episode(env, actor.decide) for _ in range(episodes)]


def model_fidelity(model, env_sets=3):
    """Diagnostic only: how close the model is to the oracle on held-out states.

    Uses the tables to produce ground truth, so it is never part of the loop --
    it just lets the curve be read in terms of model quality as well as reward.
    """
    from nero.search.canonicalization import dense_rewards

    env = Eval_JobSchedulingEnv(str(TEST_SETS))
    agree, n, errs = 0, 0, []
    for _ in range(env_sets):
        env.reset()
        while env.current_job_idx < env.J:
            j = env.current_job_idx
            slots = env.feasible_slots(j)
            if not slots:
                break
            r, _ = dense_rewards(env, j)
            with torch.no_grad():
                pred = model.predict(torch.from_numpy(slot_features(env, j, slots))).numpy()
            true = np.array([r[s * env.A + a] for (s, a) in slots])
            errs.append(np.abs(pred - true).mean())
            agree += int(np.argmax(pred) == np.argmax(true))
            n += 1
            env.step(slots[int(np.argmax(true))])
    return agree / max(n, 1), float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default="cold", choices=("cold", "partial", "full"),
                    help="cold: random weights; partial/full: load --learned-dir")
    ap.add_argument("--episodes", type=int, default=400, help="deployment episodes")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--explore-eps", type=float, default=0.10,
                    help="fraction of decisions played off-policy, to cover slots "
                         "the search would never try")
    ap.add_argument("--update-every", type=int, default=1,
                    help="gradient steps are taken every N episodes")
    ap.add_argument("--grad-steps", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--capacity", type=int, default=50000)
    ap.add_argument("--eval-every", type=int, default=25, help="episodes between tests")
    ap.add_argument("--learned-dir", default=DEFAULT_DIR)
    ap.add_argument("--out-dir", default=str(MODELS / "job_scheduling" / "online_reward_model"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)

    probe = Eval_JobSchedulingEnv(str(TEST_SETS))
    probe.reset()
    fdim = feature_dim(probe)

    agent, _, _ = load_inner_agent(device="cpu")      # frozen policy, trained offline
    model = SlotRewardModel(fdim, args.out_dir, lr=args.lr)
    if args.init != "cold":
        src = SlotRewardModel(fdim, args.learned_dir)
        src.load()
        model.load_state_dict(src.state_dict())
    model.eval()

    actor = LearnedTopKActor(k=args.k, mode="reward", agent=agent, device="cpu",
                             threads=0, reward_model=model)

    buf = LabelBuffer(args.capacity, fdim)
    env = Train_JobSchedulingEnv()
    history = []

    def snapshot(ep):
        scores = evaluate(actor)
        top1, mae = model_fidelity(model)
        rec = dict(episode=ep, labels=buf.seen, heldout_mean=float(np.mean(scores)),
                   heldout_std=float(np.std(scores, ddof=1)),
                   argmax_agreement=top1, mae=mae)
        history.append(rec)
        print(f"  ep {ep:4d} | labels {buf.seen:6d} | held-out {rec['heldout_mean']:7.4f} "
              f"| model agree {top1:.3f} MAE {mae:.2e}", flush=True)
        return rec

    print(f"init={args.init}  policy=frozen  K={args.k}  "
          f"explore_eps={args.explore_eps}")
    print("deployment stream: randomly generated job sets; "
          "test: 20 held-out sets, greedy\n")
    snapshot(0)

    for ep in range(1, args.episodes + 1):
        env.reset()
        while env.current_job_idx < env.J:
            j = env.current_job_idx
            feasible = env.feasible_slots(j)
            if not feasible:
                break
            if py_rng.random() < args.explore_eps:
                slot = py_rng.choice(feasible)     # coverage for the reward model
            else:
                slot = actor.decide(env)
                if slot is None:
                    break
            # the one label a live cluster gets: what this placement actually paid
            feat = slot_features(env, j, [slot])[0]
            _, rew, term, trunc, _ = env.step(slot)
            buf.add(feat, float(sum(rew)))
            if term or trunc:
                break

        if ep % args.update_every == 0 and buf.size >= args.batch_size:
            model.train()
            for _ in range(args.grad_steps):
                f, r = buf.sample(args.batch_size, rng)
                pred = model.predict(torch.from_numpy(f))
                loss = F.mse_loss(pred, torch.from_numpy(r))
                model.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                model.optimizer.step()
            model.eval()

        if ep % args.eval_every == 0:
            snapshot(ep)

    model.save()
    tag = f"_{args.tag}" if args.tag else f"_{args.init}"
    os.makedirs(CURVE_DIR, exist_ok=True)
    with open(f"{CURVE_DIR}/online_reward_model{tag}.json", "w") as f:
        json.dump(dict(config=vars(args), history=history), f, indent=2)
    print(f"\nsaved -> {args.out_dir}/reward_model.pth")
    print(f"curve -> {CURVE_DIR}/online_reward_model{tag}.json")


if __name__ == "__main__":
    main()
