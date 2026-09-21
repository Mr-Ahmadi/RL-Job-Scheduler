"""Full online learning: a deployed scheduler that keeps updating on live traffic.

The experiment this script runs:

1. a stream of job sets is generated -- the first ``--shift-at`` episodes from the
   training mix, the rest from a **shifted** mix (heavier models, larger batch
   sizes), which is what a real cluster does to a scheduler trained last quarter;
2. the exact same stream is served by every arm under test:
   * **frozen** -- today's deployment: fixed policy, greedy, no search;
   * **frozen+search** -- Solution 4 with frozen heads (only with ``--use-topk``),
     which separates what the *search* buys from what *learning* buys;
   * **online** -- ``nero.online.learner.OnlineLearner``, which keeps updating
     its reward model, value head, Q-head, critic and policy behind a promotion
     gate;
3. every ``--probe-every`` updates the online learner's *policy alone* (no
   search, greedy) is measured on a fixed probe set drawn from the current mix,
   which isolates how much of the search has been amortised into the policy;
4. all arms are then re-evaluated on the 20 held-out job sets, to check that
   adapting to the new mix did not wreck performance on the old one.

Usage:
    python -m scripts.online_learning                          # policy-only online learning
    python -m scripts.online_learning --use-topk               # Solution 4 + online learning
    python -m scripts.online_learning --gate canary            # simulator-free promotion gate
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from nero.agents.loading import load_inner_agent
from nero.search.canonicalization import feature_dim
from nero.search.heads import (DEFAULT_DIR, SlotQHead, SlotRewardModel,
                                   SlotValueHead)
from nero.online.learner import OnlineLearner
from nero.envs.problem import JobTable, model_names
from nero.envs.job_scheduling.eval import Eval_JobSchedulingEnv
from nero.envs.job_scheduling.train import Train_JobSchedulingEnv
from nero.paths import ONLINE, SCORES, TEST_SETS

CURVE_DIR = str(SCORES)
ONLINE_DIR = str(ONLINE)

# the shifted mix over-weights the two heaviest model families and the largest
# batch sizes -- job types the training mix sees, but rarely and rarely together
SHIFT_WEIGHTS = {"ResNet-50": 4.0, "Transformer": 3.0,
                 "LM": 0.5, "Recommendation": 0.5, "ResNet-18": 0.5}


def job_types():
    return [(t.model, 1) for t in JobTable]


def _family(model_str):
    for name in model_names:
        if model_str.startswith(name):
            return name
    return None


def make_stream(n, shift_at, rng, min_jobs=20, max_jobs=90):
    """Deterministic stream of job sets; the mix shifts at ``shift_at``."""
    types = job_types()
    base_w = [1.0] * len(types)
    shift_w = []
    for (m, _) in types:
        w = SHIFT_WEIGHTS.get(_family(m), 1.0)
        # within a family, bias towards the larger batch sizes
        digits = [int(d) for d in "".join(
            c if c.isdigit() else " " for c in m).split()]
        if digits and digits[-1] >= 128:
            w *= 2.0
        shift_w.append(w)
    stream = []
    for i in range(n):
        w = base_w if i < shift_at else shift_w
        k = rng.randint(min_jobs, max_jobs)
        stream.append(rng.choices(types, weights=w, k=k))
    return stream


def load_heads(state_dim, action_dim, fdim, learned_dir, device):
    rm = SlotRewardModel(fdim, learned_dir)
    vh = SlotValueHead(state_dim, learned_dir)
    qh = SlotQHead(state_dim, action_dim, learned_dir)
    rm.load(map_location=device)
    qh.load(map_location=device)
    if os.path.exists(vh.checkpoint_file):
        vh.load(map_location=device)
    else:
        vh = None
    return rm.to(device), (vh.to(device) if vh is not None else None), qh.to(device)


def build(args, device, learn):
    agent, state_dim, action_dim = load_inner_agent(device=device)
    probe = Eval_JobSchedulingEnv(str(TEST_SETS))
    probe.reset()
    rm = vh = qh = None
    if args.use_topk or learn:
        try:
            rm, vh, qh = load_heads(state_dim, action_dim, feature_dim(probe),
                                    args.learned_dir, device)
        except FileNotFoundError:
            if args.use_topk:
                raise SystemExit(
                    "--use-topk needs models/job_scheduling/learned_topk/*.pth; "
                    "run `python -m scripts.train_reward_model` first")
    return OnlineLearner(
        agent, probe.S, probe.A, state_dim, action_dim, feature_dim(probe),
        reward_model=rm, value_head=vh, q_head=qh,
        lr_actor=args.lr_actor, lr_critic=args.lr_critic,
        clip=args.clip, ppo_epochs=args.ppo_epochs, kl_target=args.kl_target,
        update_every=args.update_every, use_topk=args.use_topk, k=args.k,
        mode=args.mode, beta=args.beta, explore_eps=args.explore_eps,
        distill_coef=args.distill_coef, gate=args.gate,
        gate_tolerance=args.gate_tolerance, gate_patience=args.gate_patience,
        monitor_episodes=args.monitor_episodes,
        canary_frac=args.canary_frac,
        sim_env=Train_JobSchedulingEnv() if args.gate == "sim" else None,
        device=str(device), seed=args.seed)


def heldout(learner, episodes=20, topk=None):
    env = Eval_JobSchedulingEnv(str(TEST_SETS))
    return [learner.run_episode(env, learn=False, greedy=True, topk=topk)
            for _ in range(episodes)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=400, help="deployment episodes")
    ap.add_argument("--shift-at", type=int, default=100,
                    help="episode at which the job mix drifts (-1 disables the shift)")
    ap.add_argument("--update-every", type=int, default=10, help="episodes per update")
    ap.add_argument("--probe-every", type=int, default=2,
                    help="policy-only probe every N updates (0 disables)")
    ap.add_argument("--probe-episodes", type=int, default=8)
    ap.add_argument("--gate-patience", type=int, default=3,
                    help="consecutive failed gates before the candidate is rolled back")
    ap.add_argument("--gate", default="sim", choices=("sim", "canary", "none"))
    ap.add_argument("--gate-tolerance", type=float, default=0.0)
    ap.add_argument("--monitor-episodes", type=int, default=8)
    ap.add_argument("--canary-frac", type=float, default=0.25,
                    help="live-traffic share served by the candidate (gate=canary)")
    ap.add_argument("--use-topk", action="store_true",
                    help="act with the Solution-4 learned top-K search")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mode", default="reward",
                    help="scoring mode for the top-K search (reward|reward_value|q|blend)")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--explore-eps", type=float, default=0.1)
    ap.add_argument("--distill-coef", type=float, default=0.5)
    ap.add_argument("--lr-actor", type=float, default=2e-4)
    ap.add_argument("--lr-critic", type=float, default=5e-4)
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--kl-target", type=float, default=0.02)
    ap.add_argument("--learned-dir", default=DEFAULT_DIR)
    ap.add_argument("--out-dir", default=ONLINE_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default=None, help="suffix for the written curve files")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    shift_at = args.episodes + 1 if args.shift_at < 0 else args.shift_at

    stream = make_stream(args.episodes, shift_at, random.Random(args.seed))
    print(f"stream: {args.episodes} episodes, mix shifts at {shift_at}, "
          f"gate={args.gate}, use_topk={args.use_topk}, device={device}")

    frozen = build(args, device, learn=False)
    online = build(args, device, learn=True)
    searched = build(args, device, learn=False) if args.use_topk else None

    env_f = Train_JobSchedulingEnv()
    env_o = Train_JobSchedulingEnv()
    env_s = Train_JobSchedulingEnv()
    env_p = Train_JobSchedulingEnv()
    # two fixed probe sets drawn with the same seed, one per mix, so the
    # policy-only probe is comparable before and after the drift
    probe_base = make_stream(args.probe_episodes, args.probe_episodes,
                             random.Random(args.seed + 1000))
    probe_shift = make_stream(args.probe_episodes, 0,
                              random.Random(args.seed + 1000))

    frozen_returns, online_returns, search_returns = [], [], []
    probes = []
    for i, jobs in enumerate(stream):
        shifted = i >= shift_at
        frozen_returns.append(frozen.run_episode(env_f, job_list=list(jobs),
                                                 learn=False, greedy=True,
                                                 topk=False))
        if searched is not None:
            search_returns.append(searched.run_episode(env_s, job_list=list(jobs),
                                                       learn=False, greedy=True))
        online_returns.append(online.run_episode(env_o, job_list=list(jobs), learn=True))
        if (i + 1) % args.update_every == 0:
            w = slice(-args.update_every, None)
            last = online.history[-1] if online.history else {}
            if args.probe_every and online.updates % args.probe_every == 0:
                pset = probe_shift if shifted else probe_base
                pol = float(np.mean([
                    online.run_episode(env_p, job_list=list(jl), learn=False,
                                       greedy=True, topk=False) for jl in pset]))
                ref = float(np.mean([
                    frozen.run_episode(env_p, job_list=list(jl), learn=False,
                                       greedy=True, topk=False) for jl in pset]))
                probes.append(dict(episode=i + 1, shifted=shifted,
                                   online_policy=pol, frozen_policy=ref))
            line = (f"ep {i + 1:4d}/{args.episodes}  "
                    f"frozen={np.mean(frozen_returns[w]):7.3f}  ")
            if searched is not None:
                line += f"frozen+search={np.mean(search_returns[w]):7.3f}  "
            line += (f"online={np.mean(online_returns[w]):7.3f}  "
                     f"promoted={str(last.get('promoted')):5s} "
                     f"kl={last.get('kl', float('nan')):.4f}")
            if probes and probes[-1]["episode"] == i + 1:
                line += (f"  policy-probe={probes[-1]['online_policy']:.3f}"
                         f" (frozen {probes[-1]['frozen_policy']:.3f})")
            print(line, flush=True)

    print("\nheld-out evaluation (20 fixed job sets, greedy):")
    frozen_heldout = heldout(frozen, topk=False)
    online_heldout = heldout(online, topk=False)
    print(f"  frozen policy   mean={np.mean(frozen_heldout):7.4f} "
          f"± {np.std(frozen_heldout, ddof=1):6.4f}")
    print(f"  online policy   mean={np.mean(online_heldout):7.4f} "
          f"± {np.std(online_heldout, ddof=1):6.4f}")
    search_heldout = None
    if searched is not None:
        search_heldout = heldout(searched)
        online_search_heldout = heldout(online)
        print(f"  frozen+search   mean={np.mean(search_heldout):7.4f} "
              f"± {np.std(search_heldout, ddof=1):6.4f}")
        print(f"  online+search   mean={np.mean(online_search_heldout):7.4f} "
              f"± {np.std(online_search_heldout, ddof=1):6.4f}")

    pre = slice(0, shift_at)
    post = slice(shift_at, None)
    summary = dict(
        config=vars(args),
        frozen=frozen_returns, online=online_returns,
        frozen_search=search_returns or None, probes=probes,
        frozen_heldout=frozen_heldout, online_heldout=online_heldout,
        pre_shift=dict(frozen=float(np.mean(frozen_returns[pre])),
                       online=float(np.mean(online_returns[pre]))),
        post_shift=dict(frozen=float(np.mean(frozen_returns[post])),
                        online=float(np.mean(online_returns[post]))) if shift_at <= args.episodes else None,
        history=online.history,
    )
    tag = f"_{args.tag}" if args.tag else ("_topk" if args.use_topk else "")
    os.makedirs(CURVE_DIR, exist_ok=True)
    with open(f"{CURVE_DIR}/online_learning{tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(f"{CURVE_DIR}/evaluation_scores_ppo_online{tag}.json", "w") as f:
        json.dump(online_heldout, f)
    online.save(args.out_dir)

    if summary["post_shift"]:
        d = summary["post_shift"]["online"] - summary["post_shift"]["frozen"]
        print(f"\npost-shift deployment mean: frozen={summary['post_shift']['frozen']:.3f} "
              f"online={summary['post_shift']['online']:.3f}  ({d:+.3f})")
    plot(summary, f"{CURVE_DIR}/online_learning{tag}.png", shift_at)
    print(f"wrote {CURVE_DIR}/online_learning{tag}.json and {args.out_dir}/")


def plot(summary, path, shift_at, window=10):
    """Two panels: what the cluster actually got, and what the policy learned.

    Styled by ``nero.plotting``, the same module the report figures use.
    """
    import nero.plotting
    from nero.plotting import BLUE, GREEN, INK_SOFT, RULE, VERM, value_axis_only

    nero.plotting.apply()
    import matplotlib.pyplot as plt

    def smooth(x):
        x = np.asarray(x, dtype=float)
        return x if len(x) < window else np.convolve(x, np.ones(window) / window,
                                                     mode="valid")

    probes = summary.get("probes") or []
    n_panels = 2 if probes else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(6.0, 2.6 * n_panels), dpi=150,
                             sharex=True, squeeze=False)

    ax = axes[0][0]

    def xs_for(v):
        # a series shorter than the window is plotted unsmoothed, from episode 0
        n = len(smooth(v))
        return np.arange(len(v) - n, len(v))

    if summary.get("frozen_search"):
        ax.plot(xs_for(summary["frozen_search"]), smooth(summary["frozen_search"]),
                color=VERM, linewidth=1.0, label="frozen $+$ search", zorder=4)
    ax.plot(xs_for(summary["online"]), smooth(summary["online"]), color=BLUE,
            linewidth=1.0, label="online learning", zorder=3)
    ax.plot(xs_for(summary["frozen"]), smooth(summary["frozen"]), color=INK_SOFT,
            linewidth=0.9, linestyle=(0, (4, 2)), label="frozen policy", zorder=2)
    ax.set_ylabel(f"Episode reward\n({window}-episode moving average)")
    ax.legend(loc="upper right", labelspacing=0.35)
    value_axis_only(ax, "y")

    if probes:
        # relative to the frozen policy on the same probe set, because the probe
        # set changes at the shift and absolute levels are not comparable
        ax2 = axes[1][0]
        px = [p["episode"] for p in probes]
        rel = [100.0 * (p["online_policy"] / p["frozen_policy"] - 1.0) for p in probes]
        ax2.axhline(0.0, color=INK_SOFT, linestyle=(0, (4, 2)), linewidth=0.8,
                    zorder=2)
        ax2.plot(px, rel, marker="o", markersize=2.4, linewidth=1.0, color=GREEN,
                 zorder=3, label="deployed policy alone (greedy, no search)")
        ax2.set_ylabel("Policy alone, vs frozen (\\%)"
                       if plt.rcParams["text.usetex"] else "Policy alone, vs frozen (%)")
        ax2.legend(loc="lower right")
        value_axis_only(ax2, "y")

    for row in axes:
        row[0].axvline(shift_at, color=RULE, linewidth=0.7, zorder=1)
    axes[-1][0].set_xlabel(f"Deployment episode (job mix shifts at {shift_at})")
    axes[0][0].set_title("Continual online learning under job-mix drift",
                         fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
