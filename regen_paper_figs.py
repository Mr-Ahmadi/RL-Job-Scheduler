"""Publication-quality figure generation for the two-tier RL job scheduler paper.

Usage:
    python regen_paper_figs.py             # collect data, save figure JSONs, render all figures
    python regen_paper_figs.py --plot-only # re-render every figure from the saved figure JSONs

Data flow
---------
    raw evaluation / training JSONs (curves/*/*.json)
        -> curves/figures/figures.json     (canonical, self-contained figure data; committed)
        -> __paper/images/*.pdf | *.png    (vector PDF for the papers, PNG for README)

All plotting functions read exclusively from the saved figure JSON, never from the
raw eval JSONs.  This makes every figure fully regenerable: running with
``--plot-only`` reproduces byte-identical plots from the committed JSON alone.
"""
import json
import os
import argparse
from datetime import date

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
CURVES = os.path.join(ROOT, "curves")
FIGURE_DATA = os.path.join(CURVES, "figures")
IMAGES = os.path.join(ROOT, "__paper", "images")
os.makedirs(FIGURE_DATA, exist_ok=True)
os.makedirs(IMAGES, exist_ok=True)

# --------------------------------------------------------------------------- #
# Style: consistent with two-column ML venue formatting (sans-serif, ~7 pt)
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

W_SINGLE, W_DOUBLE = 3.45, 6.9     # inches (single- / double-column)
PNG_DPI = 300

# Color-blind-safe Okabe-Ito palette.
C_BLUE, C_SKY, C_GREEN, C_ORANGE, C_VERMILION, C_PINK = (
    "#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7")
C_GRAY, C_DGRAY, C_LGRAY, C_BLACK = "#8C8C8C", "#555555", "#B8B8B8", "#222222"

# --------------------------------------------------------------------------- #
# Policy registry (key -> human label)
# --------------------------------------------------------------------------- #
def _policy_specs():
    """(key, label, group, color, hatch, marker, raw_filename)."""
    return [
        ("ppo_two_tier",       "PPO two-tier (joint)",   "RL learned", C_BLUE,  None,   "o",  "subset_selector"),
        ("ppo_topk_k5",        "PPO top-K (K=5)",        "RL learned", C_SKY,   None,   "o",  "ppo_topk_k5"),
        ("ppo_topk_k3",        "PPO top-K (K=3)",        "RL learned", C_GREEN, None,   "o",  "ppo_topk_k3"),
        ("gavel_max_total",    "Max-total oracle",       "baseline",   C_ORANGE, "//",  "s",  "gavel_max_total"),
        ("ppo",                "PPO (greedy)",           "RL learned", C_VERMILION, None, "o", "ppo"),
        ("sia_colocated_dist", "Sia shared + dist.",     "baseline",   C_PINK,  None,   "^",  "sia_colocated_dist"),
        ("sia_colocated",      "Sia shared",             "baseline",   C_GRAY,  None,   "^",  "sia_colocated"),
        ("sia_original_dist",  "Sia exclusive + dist.",  "baseline",   C_DGRAY, None,   "^",  "sia_original_dist"),
        ("sia_original",       "Sia exclusive",          "baseline",   C_LGRAY, None,   "^",  "sia_original"),
        ("gavel_max_throughput", "Max-throughput",       "baseline",   C_BLACK, None,   "d",  "gavel_max_throughput"),
        ("random",             "Random",                 "baseline",   "#AAAAAA", None,  "x",  "random"),
    ]

# --------------------------------------------------------------------------- #
# Data collection: raw eval/training JSONs -> canonical figure JSON
# --------------------------------------------------------------------------- #
def load_raw(name):
    with open(os.path.join(CURVES, "job_scheduling", f"evaluation_scores_{name}.json")) as f:
        return np.asarray(json.load(f), dtype=float)


def collect():
    data = {"version": 1, "generated": date.today().isoformat(), "policies": {}, "order": [],
            "per_set": {}, "training": {}}

    for key, label, group, color, hatch, marker, raw in _policy_specs():
        vals = load_raw(raw)
        data["policies"][key] = {
            "label": label, "group": group, "color": color, "hatch": hatch,
            "marker": marker, "values": [float(v) for v in vals],
            "mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
            "se": float(vals.std(ddof=1) / np.sqrt(len(vals))),
        }
        data["order"].append(key)

    # Per-set comparison across the 20 held-out job sets.
    for src, dest in (("subset_selector", "two_tier"), ("ppo", "inner_ppo"),
                      ("gavel_max_total", "oracle"), ("random", "random")):
        data["per_set"][dest] = [float(v) for v in load_raw(src)]
    data["per_set"]["sets"] = list(range(1, len(data["per_set"]["two_tier"]) + 1))

    # Training curves.
    data["training"] = {
        "inner_ppo": [float(v) for v in json.load(
            open(os.path.join(CURVES, "job_scheduling", "training_ppo.json")))],
        "outer_ppo": [float(v) for v in json.load(
            open(os.path.join(CURVES, "subset_selector", "training_ppo.json")))],
        "outer_eval_total": [float(v) for v in json.load(
            open(os.path.join(CURVES, "subset_selector", "eval_total_sum.json")))],
        "window": {"inner": 5, "outer": 25},
    }

    with open(os.path.join(FIGURE_DATA, "figures.json"), "w") as f:
        json.dump(data, f, indent=2)
    return data


def load_figure_data():
    with open(os.path.join(FIGURE_DATA, "figures.json")) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def _new_axes(width, height=None, nrows=1, sharex=False):
    if height is None:
        height = width * 0.72
    fig, axs = plt.subplots(nrows, 1, figsize=(width, height * nrows), sharex=sharex)
    return fig, axs


def _save(fig, name):
    fig.savefig(os.path.join(IMAGES, f"{name}.pdf"))
    fig.savefig(os.path.join(IMAGES, f"{name}.png"), dpi=PNG_DPI)
    plt.close(fig)
    print(f"  wrote __paper/images/{name}.pdf/.png")


def _annotate_bars(ax, bars, values, dy=0.5, errs=None):
    """Label each bar, clearing its error-bar cap when one is drawn."""
    errs = [0.0] * len(values) if errs is None else errs
    for b, v, e in zip(bars, values, errs):
        ax.text(b.get_x() + b.get_width() / 2, v + e + dy, f"{v:.2f}",
                ha="center", va="bottom", fontsize=6.5)


def _jittered_points(ax, x, values, color, marker, size=14, alpha=0.5, seed=0):
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.16, 0.16, len(values))
    if marker in ("x", "+", "*"):
        ax.scatter(x + jitter, values, s=size, marker=marker, color=color,
                   linewidths=0.6, alpha=alpha, zorder=3)
    else:
        ax.scatter(x + jitter, values, s=size, marker=marker, facecolors="none",
                   edgecolors=color, linewidths=0.5, alpha=alpha, zorder=3)


def _smooth(y, w):
    w = max(int(w), 1)
    y = np.asarray(y, dtype=float)
    pad = np.pad(y, (w - 1, 0), mode="edge")
    kernel = np.ones(w) / w
    return np.convolve(pad, kernel, mode="valid")


def _smooth_band(y, w):
    mean = _smooth(y, w)
    sq = _smooth(np.asarray(y, dtype=float) ** 2, w)
    std = np.sqrt(np.clip(sq - mean ** 2, 0, None))
    return mean, std


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_policy_comparison(d):
    """Mean cumulative throughput of every policy (+/- SEM) with per-set dots."""
    order = d["order"]
    means = [d["policies"][k]["mean"] for k in order]
    ses = [d["policies"][k]["se"] for k in order]
    fig, ax = _new_axes(W_DOUBLE, 2.6)
    ax.grid(axis="y", zorder=0)

    colors = [d["policies"][k]["color"] for k in order]
    bars = ax.bar(range(len(order)), means, yerr=ses, capsize=2.2,
                  color=colors, edgecolor="white", linewidth=0.4,
                  zorder=2, error_kw=dict(elinewidth=0.7, ecolor="#333333"))
    for b, k in zip(bars, order):
        if d["policies"][k]["hatch"]:
            b.set_hatch(d["policies"][k]["hatch"])
            b.set_edgecolor(C_ORANGE)

    for i, k in enumerate(order):
        _jittered_points(ax, i, d["policies"][k]["values"], colors[i],
                         d["policies"][k]["marker"], seed=i)
    _annotate_bars(ax, bars, means, dy=0.25, errs=ses)

    ax.axhline(d["policies"]["gavel_max_total"]["mean"], ls=(0, (4, 2)),
               color=C_ORANGE, lw=0.8, zorder=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([d["policies"][k]["label"] for k in order], rotation=42,
                       ha="right", fontsize=6.5)
    ax.set_ylabel("Cumulative throughput")
    ax.set_ylim(0, max(means) * 1.22)
    _save(fig, "policy_comparison")

    # README copy (bitmap only, same path as before).
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(range(len(order)), means, yerr=ses, capsize=3,
           color=colors, edgecolor="white")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([d["policies"][k]["label"] for k in order], rotation=22, ha="right")
    ax.set_ylabel("Mean cumulative throughput")
    ax.set_ylim(0, max(means) * 1.2)
    fig.tight_layout()
    fig.savefig(os.path.join(CURVES, "job_scheduling", "policy_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wrote curves/job_scheduling/policy_comparison.png")


def fig_violin_comparison(d):
    """Inner-agent policies: per-set distributions with mean markers."""
    keys = ["ppo", "gavel_max_total", "gavel_max_throughput",
            "sia_colocated", "sia_colocated_dist", "sia_original", "sia_original_dist"]
    labels = [d["policies"][k]["label"] for k in keys]
    order = sorted(range(len(keys)), key=lambda i: d["policies"][keys[i]]["mean"])
    keys = [keys[i] for i in order]
    labels = [labels[i] for i in order]

    values = [d["policies"][k]["values"] for k in keys]
    means = [d["policies"][k]["mean"] for k in keys]
    colors = [d["policies"][k]["color"] for k in keys]

    fig, ax = _new_axes(W_SINGLE, 2.5)
    ax.grid(axis="y", zorder=0)
    parts = ax.violinplot(values, positions=range(len(keys)), showextrema=False,
                          widths=0.72)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(0.30)
        body.set_edgecolor(c)
        body.set_linewidth(0.7)
    for i, (v, c, m, k) in enumerate(zip(values, colors, means, keys)):
        _jittered_points(ax, i, v, c, d["policies"][k]["marker"], size=11, seed=i)
        ax.plot(i, m, marker="D", ms=4, color=c, zorder=4, mec="white", mew=0.4)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=6.2)
    ax.set_ylabel("Cumulative throughput")
    _save(fig, "agents_violin_comparison")


def fig_two_tier_per_set(d):
    """Per-set two-tier vs. inner greedy PPO vs. max-total oracle."""
    ps = d["per_set"]
    x = np.array(ps["sets"])
    fig, ax = _new_axes(W_SINGLE, 2.35)
    ax.grid(axis="y", zorder=0)

    series = [("two_tier", "PPO two-tier", C_BLUE, "o", "-"),
              ("oracle", "Max-total oracle", C_ORANGE, "s", "--"),
              ("inner_ppo", "Inner PPO (greedy)", C_VERMILION, "^", ":")]
    for key, label, color, marker, ls in series:
        ax.plot(x, ps[key], marker=marker, ms=2.5, lw=0.9, ls=ls, color=color,
                label=label, zorder=3)
        m = np.mean(ps[key])
        ax.axhline(m, ls=ls, color=color, lw=0.7, alpha=0.8, zorder=1)
        ax.text(len(x) + 0.45, m, f"{m:.2f}", va="center", ha="left",
                fontsize=6.3, color=color)

    ax.set_xlim(0.5, len(x) + 2.3)
    ax.set_xlabel("Held-out job set")
    ax.set_ylabel("Cumulative throughput")
    ax.legend(loc="lower left", frameon=False, fontsize=6.3)
    _save(fig, "two_tier_per_set")


def fig_training_curves(d):
    """Inner PPO training curve (episode reward) and outer PPO training curves."""
    tr = d["training"]
    win_inner, win_outer = tr["window"]["inner"], tr["window"]["outer"]

    fig, ax = _new_axes(W_SINGLE, 2.3)
    y = np.array(tr["inner_ppo"])
    m, s = _smooth_band(y, win_inner)
    x = np.arange(1, len(m) + 1)
    ax.fill_between(x, m - s, m + s, color=C_BLUE, alpha=0.18, lw=0)
    ax.plot(x, m, color=C_BLUE, lw=1.1)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode reward")
    _save(fig, "training_ppo_inner")

    fig, axs = _new_axes(W_SINGLE, 4.2, nrows=2, sharex=True)
    for ax, key, color, ylab in (
            (axs[0], "outer_ppo", C_GREEN, "Outer-agent episode reward"),
            (axs[1], "outer_eval_total", C_VERMILION, "Eval total throughput")):
        y = np.array(tr[key])
        m, s = _smooth_band(y, win_outer)
        x = np.arange(1, len(m) + 1)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.18, lw=0)
        ax.plot(x, m, color=color, lw=1.0)
        ax.set_ylabel(ylab, fontsize=7.0)
        ax.tick_params(labelbottom=(ax is axs[1]))
    axs[1].set_xlabel("Training episode")
    _save(fig, "training_outer")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot-only", action="store_true",
                    help="re-render figures from the saved figure JSON (no raw data needed)")
    args = ap.parse_args()

    if args.plot_only:
        data = load_figure_data()
        print("plotting from curves/figures/figures.json (regeneration mode)")
    else:
        data = collect()
        print("collected raw eval/training data -> curves/figures/figures.json")

    for fn in (fig_policy_comparison, fig_violin_comparison,
               fig_two_tier_per_set, fig_training_curves):
        fn(data)

    print("\nsummary:")
    for k in data["order"]:
        p = data["policies"][k]
        print(f"  {p['label']:26s} {p['mean']:8.3f} ± {p['std']:6.3f} (se {p['se']:.3f})")


if __name__ == "__main__":
    main()
