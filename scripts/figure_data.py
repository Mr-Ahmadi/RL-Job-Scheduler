"""Measurements for the report figures that are not already on disk.

* ``report_reward_scatter.json`` -- the learned reward model's prediction against
  the true reward, for every feasible slot along greedy rollouts on the 20
  held-out job sets.  The table is read here only to produce the ground-truth
  axis of the figure; the model itself never sees it.
* ``report_latency.json`` -- the decision-path medians printed by
  ``scripts/benchmark.py``, in the form the figure expects.

Usage:  python -m scripts.figure_data
"""

import json
import os
import time

import numpy as np
import torch

from nero.agents.loading import load_inner_agent
from nero.search.canonicalization import dense_rewards, feasible_slots, slot_features
from nero.search.heads import DEFAULT_DIR, SlotRewardModel
from nero.search.topk import LearnedTopKActor
from nero.envs.job_scheduling.eval import Eval_JobSchedulingEnv
from nero.evaluation import make_topk
from nero.paths import SCORES, TEST_SETS

CURVES = str(SCORES)
WARMUP, N = 30, 400


def reward_scatter(episodes=20):
    from nero.search.canonicalization import feature_dim

    probe = Eval_JobSchedulingEnv(str(TEST_SETS))
    probe.reset()
    model = SlotRewardModel(feature_dim(probe), DEFAULT_DIR)
    model.load()
    model.eval()

    agent, _, _ = load_inner_agent(device="cpu")
    actor = LearnedTopKActor(k=5, agent=agent, device="cpu", threads=0)
    env = Eval_JobSchedulingEnv(str(TEST_SETS))

    true_all, pred_all = [], []
    for _ in range(episodes):
        env.reset()
        while env.current_job_idx < env.J:
            j = env.current_job_idx
            slots = feasible_slots(env, j)
            if not slots:
                break
            r, _ = dense_rewards(env, j)
            feats = torch.from_numpy(slot_features(env, j, slots))
            with torch.no_grad():
                pred = model.predict(feats).numpy()
            true_all.extend(float(r[s * env.A + a]) for (s, a) in slots)
            pred_all.extend(float(p) for p in pred)
            env.step(actor.decide(env))

    true = np.asarray(true_all)
    pred = np.asarray(pred_all)
    mae = float(np.abs(pred - true).mean())
    r2 = float(1.0 - ((pred - true) ** 2).sum() / ((true - true.mean()) ** 2).sum())
    print(f"reward model: n={len(true)}  MAE={mae:.3e}  R2={r2:.6f}  "
          f"max|err|={np.abs(pred - true).max():.3e}")
    json.dump({"true": [float(x) for x in true], "pred": [float(x) for x in pred],
               "mae": mae, "r2": r2},
              open(f"{CURVES}/report_reward_scatter.json", "w"))


def _bench(fn):
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(N):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)


def latency():
    from nero.search.canonicalization import canonical_classes
    from nero.search.fast_obs import NextObsBuilder
    from nero.deployment import OnlineActor

    torch.set_num_threads(1)
    cpu_agent, _, _ = load_inner_agent(device="cpu")
    actor = LearnedTopKActor(k=5, mode="reward", agent=cpu_agent, device="cpu", threads=1)
    # K=1 short-circuits the scoring step, so it is the same policy argmax as the
    # deployed greedy actor but over the vectorised feasibility scan: the gap
    # between it and K=5 is the search overhead proper
    greedy1 = LearnedTopKActor(k=1, mode="reward", agent=cpu_agent, device="cpu", threads=1)
    blend = LearnedTopKActor(k=5, mode="blend", agent=cpu_agent, device="cpu", threads=1)
    deploy = OnlineActor(threads=1)

    env = Eval_JobSchedulingEnv(str(TEST_SETS))
    env.reset()
    for _ in range(env.J // 2):        # half-full cluster
        env.step(actor.decide(env))

    builder = NextObsBuilder()
    base = env._get_obs()
    s0, a0 = feasible_slots(env)[3]
    oracle_topk = make_topk(cpu_agent, 5)

    out = {
        "canon": dict(label="canonicalisation", median=_bench(lambda: canonical_classes(env)), oracle=False),
        "nextobs": dict(label="next-obs delta", median=_bench(lambda: builder.next_obs(env, base, s0, a0)), oracle=False),
        "greedy": dict(label="PPO greedy (deployed)", median=_bench(lambda: deploy.decide(env)), oracle=False),
        "greedy_fast": dict(label="PPO greedy (same scan)", median=_bench(lambda: greedy1.decide(env)), oracle=False),
        "learned": dict(label="learned top-$K$", median=_bench(lambda: actor.decide(env)), oracle=False),
        "blend": dict(label="learned top-$K$ (blend)", median=_bench(lambda: blend.decide(env)), oracle=False),
        "oracle": dict(label="oracle top-$K$", median=_bench(lambda: oracle_topk(env)), oracle=True),
        "order": ["canon", "nextobs", "greedy", "greedy_fast", "learned", "blend", "oracle"],
    }
    for k in out["order"]:
        print(f"  {out[k]['label']:28s} {out[k]['median']:7.3f} ms")
    json.dump(out, open(f"{CURVES}/report_latency.json", "w"), indent=2)


if __name__ == "__main__":
    os.makedirs(CURVES, exist_ok=True)
    print("measuring reward-model accuracy ...")
    reward_scatter()
    print("measuring decision latency ...")
    latency()
