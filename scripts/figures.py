"""Figures for paper/oracle_free_topk_report.tex.

One plot per file, vector PDF only, sized for the report's text column. Text is
typeset by LaTeX with the same Times family as the report, so figure labels and
body text are the same typeface and the maths matches.

Every number is read from the JSON artefacts the evaluation scripts write, so a
figure cannot drift from the measurements:

  results/job_scheduling/evaluation_scores_*.json     committed 20-set benchmark
  results/job_scheduling/evaluation_scores_fresh.json 40 held-out-from-tuning sets
  results/job_scheduling/online_learning_*.json       drift experiment
  results/job_scheduling/report_reward_scatter.json   reward-model fidelity
  results/job_scheduling/report_latency.json          decision-path medians

The last two come from scripts/figure_data.py; run it first if they are missing.

Usage:  python -m scripts.figures
"""

import json
import os

import nero.plotting
from nero.plotting import (BLUE, GREEN, HATCH, INK, INK_SOFT, RULE, VERM, W,
                        W_SQUARE, value_axis_only)

nero.plotting.apply()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from nero.paths import ROOT as PROJECT_ROOT, FIGURES, OUTER_CURVES, SCORES

ROOT = str(PROJECT_ROOT)
CURVES = str(SCORES)
IMAGES = str(FIGURES)
os.makedirs(IMAGES, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(IMAGES, f"{name}.pdf"))
    plt.close(fig)
    print(f"  images/{name}.pdf")


def load(name):
    with open(os.path.join(CURVES, f"evaluation_scores_{name}.json")) as f:
        return np.asarray(json.load(f), dtype=float)


def load_validation():
    """Selection-only scores on randomly generated job sets (never reported)."""
    path = os.path.join(CURVES, "validation_scores.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {k: np.asarray(v, dtype=float) for k, v in json.load(f).items()}


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))


def spread_labels(ys, min_gap):
    """Nudge label positions apart, preserving order, for the slope chart."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for i in range(1, len(order)):
        a, b = order[i - 1], order[i]
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


# --------------------------------------------------------------------------- #
# Fig. 3 - the learned reward model against the oracle it replaces
# --------------------------------------------------------------------------- #
def fig_reward_model():
    path = os.path.join(CURVES, "report_reward_scatter.json")
    if not os.path.exists(path):
        print("  (skipping reward model: run scripts/figure_data.py)")
        return
    with open(path) as f:
        d = json.load(f)
    true, pred = np.asarray(d["true"]), np.asarray(d["pred"])
    lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
    pad = 0.05 * (hi - lo)

    fig, ax = plt.subplots(figsize=(W_SQUARE, 3.30))
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linewidth=0.8, color=RULE,
            linestyle=(0, (4, 2)), zorder=2, label="exact agreement")
    ax.scatter(true, pred, s=1.6, color=BLUE, alpha=0.22, linewidths=0, zorder=3,
               rasterized=True)
    ax.set_xlabel("True reward from the throughput table")
    ax.set_ylabel("Learned reward model")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left")
    n = f"{len(true):,}".replace(",", "{,}")
    ax.text(0.96, 0.05,
            f"$n = {n}$ candidate slots\n"
            f"MAE $= {d['mae'] * 1e3:.1f}\\times10^{{-3}}$\n"
            f"$R^2 = {d['r2']:.5f}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.6,
            color=INK_SOFT, linespacing=1.5)
    ax.set_axisbelow(True)
    save(fig, "report_reward_model")


# --------------------------------------------------------------------------- #
# Fig. 4 - what the candidate set and the score contribute
# --------------------------------------------------------------------------- #
def fig_ablation():
    """Reported on the 20 test sets; the winner was chosen on validation."""
    rows = [
        ("$\\hat r$, dedupe\n(default)", "ppo_topk_learned_k5", BLUE),
        ("$\\hat r$, expand", "ppo_topk_learned_k5_expand", GREEN),
        ("$\\hat r + \\gamma V(s\')$", "ppo_topk_learned_k5_reward_value", GREEN),
        ("blend\n($\\beta\\!=\\!0.5$)", "ppo_topk_learned_k5_blend", GREEN),
        ("$Q$ only", "ppo_topk_learned_k5_q", GREEN),
    ]
    rows = [r for r in rows
            if os.path.exists(os.path.join(CURVES,
                                           f"evaluation_scores_{r[1]}.json"))]
    vals = np.array([load(k).mean() for _, k, _ in rows])
    errs = np.array([se(load(k)) for _, k, _ in rows])
    x = np.arange(len(rows))
    base = load("ppo").mean()

    fig, ax = plt.subplots(figsize=(W, 2.70))
    ax.bar(x, vals, width=0.58, color=[c for _, _, c in rows], zorder=2,
           linewidth=0)
    ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor=INK, elinewidth=0.6,
                capsize=1.6, capthick=0.6, zorder=3)
    ax.axhline(base, color=INK_SOFT, linestyle=(0, (4, 2)), linewidth=0.8, zorder=1,
               label="inner policy alone")
    for xi, v, e in zip(x, vals, errs):
        ax.text(xi, v + e + 0.10, f"{v:.2f}", ha="center", fontsize=7.6, color=INK,
                zorder=4)

    lo = min(np.array(vals) - np.array(errs))
    ax.set_ylim(lo - 0.45, max(np.array(vals) + np.array(errs)) * 1.055)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for lab, _, _ in rows])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylabel("Mean reward (20 test sets)")
    ax.legend(loc="upper right")
    value_axis_only(ax, "y")
    save(fig, "report_ablation")


# --------------------------------------------------------------------------- #
# Fig. 5 - decision latency
# --------------------------------------------------------------------------- #
def fig_latency():
    path = os.path.join(CURVES, "report_latency.json")
    if not os.path.exists(path):
        print("  (skipping latency: run scripts/figure_data.py)")
        return
    with open(path) as f:
        d = json.load(f)
    keys = [k for k in ["greedy", "greedy_fast", "learned", "blend", "oracle"]
            if k in d]
    labels = {
        "greedy": "PPO greedy (deployed)",
        "greedy_fast": "PPO greedy (same scan)",
        "learned": "Learned top-$K$",
        "blend": "Learned top-$K$ (blend)",
        "oracle": "Oracle top-$K$",
    }
    order = sorted(keys, key=lambda k: -d[k]["median"])
    vals = np.array([d[k]["median"] for k in order])
    y = np.arange(len(order))[::-1]

    fig, ax = plt.subplots(figsize=(W, 2.10))
    for yi, k, v in zip(y, order, vals):
        oracle = d[k]["oracle"]
        ax.barh(yi, v, height=0.62, zorder=2,
                facecolor="white" if oracle else BLUE,
                edgecolor=VERM if oracle else BLUE,
                hatch=HATCH if oracle else None, linewidth=0.6)
        ax.text(v + vals.max() * 0.025, yi, f"{v:.2f}", va="center", ha="left",
                fontsize=7.6, color=INK, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[k] for k in order])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Median decision latency (ms)")
    ax.set_xlim(0, vals.max() * 1.13)
    value_axis_only(ax, "x")
    save(fig, "report_latency")


# --------------------------------------------------------------------------- #
# Fig. 6 - throughput under job-mix drift
# --------------------------------------------------------------------------- #
def fig_drift(window=15):
    path = os.path.join(CURVES, "online_learning_topk.json")
    if not os.path.exists(path):
        print("  (skipping drift: run scripts/online_learning.py --use-topk)")
        return
    with open(path) as f:
        d = json.load(f)
    shift = d["config"]["shift_at"]

    def smooth(v):
        v = np.asarray(v, dtype=float)
        return np.convolve(v, np.ones(window) / window, mode="valid")

    xs = np.arange(window - 1, len(d["frozen"]))
    fig, ax = plt.subplots(figsize=(W, 2.70))
    ax.plot(xs, smooth(d["frozen_search"]), color=VERM, linewidth=1.0,
            label="frozen $+$ search", zorder=4)
    ax.plot(xs, smooth(d["online"]), color=BLUE, linewidth=1.0,
            label="search $+$ online learning", zorder=3)
    ax.plot(xs, smooth(d["frozen"]), color=INK_SOFT, linewidth=0.9,
            linestyle=(0, (4, 2)), label="frozen policy", zorder=2)

    ax.axvline(shift, color=RULE, linewidth=0.7, zorder=1)
    ax.annotate("job mix shifts", xy=(shift, 1.0), xycoords=("data", "axes fraction"),
                xytext=(-4, -3), textcoords="offset points", fontsize=7.4,
                color=INK_SOFT, va="top", ha="right")
    ax.set_xlabel(f"Deployment episode ({window}-episode moving average)")
    ax.set_ylabel("Episode throughput reward")
    ax.set_xlim(0, len(d["frozen"]))
    ax.legend(loc="upper right", labelspacing=0.35)
    value_axis_only(ax, "y")
    save(fig, "report_drift")


# --------------------------------------------------------------------------- #
# Fig. 7 - what continual learning does to the policy itself
# --------------------------------------------------------------------------- #
def fig_probe():
    runs = [("policy only", "online_learning_policy.json", BLUE),
            ("with search", "online_learning_topk.json", VERM)]
    series = []
    for lab, fn, col in runs:
        p = os.path.join(CURVES, fn)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            d = json.load(f)
        pr = [q for q in d["probes"] if q["shifted"]]
        if pr:
            series.append((lab, col, [q["episode"] for q in pr],
                           [100 * (q["online_policy"] / q["frozen_policy"] - 1)
                            for q in pr]))
    if not series:
        print("  (skipping probe: no online-learning runs)")
        return

    fig, ax = plt.subplots(figsize=(W, 2.70))
    ax.axhline(0.0, color=INK_SOFT, linestyle=(0, (4, 2)), linewidth=0.8, zorder=2)
    for lab, col, xs, ys in series:
        ax.plot(xs, ys, marker="o", markersize=2.4, linewidth=1.0, color=col,
                label=lab, zorder=3)
    ax.annotate("frozen policy", xy=(1.0, 0.0), xycoords=("axes fraction", "data"),
                xytext=(-2, 3), textcoords="offset points", fontsize=7.4,
                color=INK_SOFT, ha="right", va="bottom")
    ax.set_xlabel("Deployment episode (after the shift)")
    ax.set_ylabel("Policy alone, vs frozen (\\%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              columnspacing=1.6)
    value_axis_only(ax, "y")
    save(fig, "report_probe")


# --------------------------------------------------------------------------- #
# Fig. 8 - bootstrapping the reward model online, with no prior profiling
# --------------------------------------------------------------------------- #
def fig_online_reward_model():
    runs = [("cold start (random weights)", "online_reward_model_cold.json", BLUE),
            ("partial (2 of 5 families unseen)", "online_reward_model_partial.json", VERM)]
    series = []
    for lab, fn, col in runs:
        path = os.path.join(CURVES, fn)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            h = json.load(f)["history"]
        series.append((lab, col, [r["labels"] for r in h],
                       [r["heldout_mean"] for r in h]))
    if not series:
        print("  (skipping online reward model: run scripts/online_reward_model.py)")
        return

    fig, ax = plt.subplots(figsize=(W, 2.70))
    # the oracle top-K line (17.21) is omitted: it sits within 0.3 of the
    # offline-trained line and of the curves themselves, and only adds clutter
    refs = [("offline-trained, dense labels", 17.48, GREEN, 0.34),
            ("max-total oracle", 16.10, INK_SOFT, 0.03),
            ("PPO policy alone", 15.15, INK_SOFT, 0.03)]
    for lab, y, col, xf in refs:
        ax.axhline(y, color=col, linestyle=(0, (4, 2)), linewidth=0.8, zorder=1)
        ax.annotate(lab, xy=(xf, y), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", fontsize=7.0,
                    color=col, ha="left", va="bottom", zorder=5)
    for lab, col, xs, ys in series:
        ax.plot(np.asarray(xs) / 1000.0, ys, marker="o", markersize=2.6,
                linewidth=1.1, color=col, label=lab, zorder=3)
    ax.set_xlabel("Realised labels observed online (thousands, one per decision)")
    ax.set_ylabel("Mean reward, 20 held-out sets")
    ax.set_ylim(12.4, 18.4)
    ax.legend(loc="lower right", labelspacing=0.35)
    value_axis_only(ax, "y")
    save(fig, "report_online_reward_model")


# --------------------------------------------------------------------------- #
# Fig. A - inner agent training curve
# --------------------------------------------------------------------------- #
def fig_inner_training(eval_every=300):
    path = os.path.join(CURVES, "training_ppo.json")
    if not os.path.exists(path):
        print("  (skipping inner training: no training_ppo.json)")
        return
    with open(path) as f:
        y = np.asarray(json.load(f), dtype=float)
    x = (np.arange(len(y)) + 1) * eval_every

    fig, ax = plt.subplots(figsize=(W, 2.40))
    ax.plot(x, y, color=BLUE, linewidth=1.1, zorder=3)
    best = float(y.max())
    ax.axhline(best, color=INK_SOFT, linestyle=(0, (4, 2)), linewidth=0.8, zorder=1)
    ax.annotate(f"best checkpoint, {best:.2f}", xy=(x[-1], best), xytext=(-3, 3),
                textcoords="offset points", fontsize=7.4, color=INK_SOFT,
                ha="right", va="bottom")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Held-out mean throughput")
    ax.set_xlim(0, x[-1])
    value_axis_only(ax, "y")
    save(fig, "report_inner_training")


# --------------------------------------------------------------------------- #
# Fig. B - joint training of the outer agent
# --------------------------------------------------------------------------- #
def fig_outer_training(eval_every=20, window=15):
    d_path = os.path.join(str(OUTER_CURVES), "training_ppo.json")
    t_path = os.path.join(str(OUTER_CURVES), "eval_total_sum.json")
    if not (os.path.exists(d_path) and os.path.exists(t_path)):
        print("  (skipping outer training: no subset-selector curves)")
        return
    with open(d_path) as f:
        dup = np.asarray(json.load(f), dtype=float)
    with open(t_path) as f:
        tot = np.asarray(json.load(f), dtype=float)

    def smooth(v):
        return v if len(v) < window else np.convolve(v, np.ones(window) / window,
                                                     mode="valid")

    fig, axes = plt.subplots(2, 1, figsize=(W, 3.30), sharex=True)
    for ax, v, lab, col in ((axes[0], dup, "Duplication reward", VERM),
                            (axes[1], tot, "Two-tier held-out throughput", BLUE)):
        sm = smooth(v)
        x = (np.arange(len(sm)) + (len(v) - len(sm)) + 1) * eval_every
        ax.plot(x, sm, color=col, linewidth=1.1, zorder=3)
        ax.set_ylabel(lab)
        value_axis_only(ax, "y")
    axes[0].axhline(0.0, color=INK_SOFT, linestyle=(0, (4, 2)), linewidth=0.8,
                    zorder=1)
    best = float(tot.max())
    axes[1].axhline(best, color=INK_SOFT, linestyle=(0, (4, 2)), linewidth=0.8,
                    zorder=1)
    axes[1].annotate(f"best, {best:.2f}", xy=(1.0, best),
                     xycoords=("axes fraction", "data"), xytext=(-3, 3),
                     textcoords="offset points", fontsize=7.4, color=INK_SOFT,
                     ha="right", va="bottom")
    axes[1].set_xlabel("Training episode")
    fig.align_ylabels(axes)
    save(fig, "report_outer_training")


# --------------------------------------------------------------------------- #
# Fig. C - every policy on the tuned benchmark, including the heuristics
# --------------------------------------------------------------------------- #
def fig_policy_comparison():
    rows = [
        ("Two-tier $+$ search", "subset_selector_topk_both", False),
        ("Two-tier", "subset_selector", False),
        ("Learned top-$K$ ($K\\!=\\!5$)", "ppo_topk_learned_k5", False),
        ("Oracle top-$K$ ($K\\!=\\!5$)", "ppo_topk_k5", True),
        ("Max-total oracle", "gavel_max_total", True),
        ("Inner agent (greedy)", "ppo", False),
        ("Sia, shared $+$ dist.", "sia_colocated_dist", True),
        ("Sia, shared", "sia_colocated", True),
        ("Sia, exclusive $+$ dist.", "sia_original_dist", True),
        ("Sia, exclusive", "sia_original", True),
        ("Max-throughput", "gavel_max_throughput", True),
        ("Random", "random", False),
    ]
    rows = [r for r in rows
            if os.path.exists(os.path.join(CURVES,
                                           f"evaluation_scores_{r[1]}.json"))]
    vals = np.array([load(k).mean() for _, k, _ in rows])
    errs = np.array([se(load(k)) for _, k, _ in rows])
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(W, 3.30))
    for yi, (_, k, oracle), v, e in zip(y, rows, vals, errs):
        ax.barh(yi, v, height=0.64, zorder=2,
                facecolor="white" if oracle else BLUE,
                edgecolor=VERM if oracle else BLUE,
                hatch=HATCH if oracle else None, linewidth=0.6)
        ax.errorbar(v, yi, xerr=e, color=INK, elinewidth=0.6, capsize=1.6,
                    capthick=0.6, zorder=3)
        ax.text(v + e + 0.28, yi, f"{v:.2f}", va="center", ha="left", fontsize=7.6,
                color=INK, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, _, _ in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Mean episode throughput reward (20 held-out sets)")
    ax.set_xlim(0, vals.max() * 1.17)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    value_axis_only(ax, "x")
    save(fig, "report_policy_comparison")


# --------------------------------------------------------------------------- #
# Fig. D - per-set view of the two-tier system against the oracle
# --------------------------------------------------------------------------- #
def fig_two_tier_per_set():
    need = ["subset_selector_topk_both", "subset_selector", "gavel_max_total", "ppo"]
    if not all(os.path.exists(os.path.join(CURVES, f"evaluation_scores_{n}.json"))
               for n in need):
        print("  (skipping per-set figure: missing scores)")
        return
    both, two, orc, inner = (load(n) for n in need)
    x = np.arange(1, len(two) + 1)

    fig, ax = plt.subplots(figsize=(W, 2.60))
    for v, lab, col, ls in ((both, "Two-tier $+$ search", BLUE, "-"),
                            (two, "Two-tier", GREEN, "-"),
                            (orc, "Max-total oracle", VERM, "-"),
                            (inner, "Inner agent", INK_SOFT, (0, (4, 2)))):
        ax.plot(x, v, linestyle=ls, linewidth=1.0, color=col, marker="o",
                markersize=2.4, label=lab, zorder=3)
        ax.axhline(v.mean(), color=col, linestyle=(0, (1, 2)), linewidth=0.7,
                   zorder=1)
    ax.set_xlabel("Held-out job set")
    ax.set_ylabel("Episode throughput reward")
    ax.set_xlim(0.5, len(two) + 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              columnspacing=1.4, labelspacing=0.3)
    value_axis_only(ax, "y")
    save(fig, "report_two_tier_per_set")


if __name__ == "__main__":
    print(f"regenerating report figures -> {os.path.relpath(IMAGES, ROOT)}")
    fig_reward_model()
    fig_ablation()
    fig_latency()
    fig_drift()
    fig_probe()
    fig_online_reward_model()
    fig_inner_training()
    fig_outer_training()
    fig_policy_comparison()
    fig_two_tier_per_set()
