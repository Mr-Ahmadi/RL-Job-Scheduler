"""Sia (SOSP '23) baseline for the heterogeneous GPU job-scheduling environment.

Reference
---------
S. Jayaram Subramanya, D. Arfeen, S. Lin, A. Qiao, Z. Jia, G. R. Ganger.
*Sia: Heterogeneity-aware, goodput-optimized ML-cluster scheduling*, SOSP '23.

What Sia actually does
----------------------
Sia is **not** a per-job greedy rule.  In every scheduling round it

  1. enumerates the set of valid *configurations* ``C``, where a configuration
     ``c = (n, m, X)`` means "``m`` GPUs of type ``X`` spread over ``n`` nodes";
  2. builds a goodput matrix ``G`` of shape ``|J| x |C|``, where ``G[i, j]`` is
     the estimated goodput of job ``i`` under configuration ``c_j``;
  3. row-normalises ``G`` so utilities become comparable *across* jobs (Eq. 1)

         G[i, j]  <-  N_i^min * G[i, j] / min_j G[i, j]

  4. discounts configurations that would force a restart by the restart
     factor ``r_i`` (Eq. 3);
  5. solves a binary ILP over the whole queue at once (Eq. 2 / Eq. 4)

         max_A  sum_ij A[i, j] (r_i G[i, j])^p  +  lambda (1 - ||A_i||_1)

     subject to ``||A_i||_1 <= 1`` (each job gets at most one configuration)
     and per-GPU-type capacity constraints.  For ``p < 0`` the sign of the
     objective is flipped (minimise instead of maximise) to preserve its
     semantics; Sia's default fairness knob is ``p = -0.5``.

The **global capacity constraint in step 5 is what makes multi-GPU
(distributed) configurations pay off.**  Sia hands a job a second GPU only
once the cluster still has room after every other queued job has been served.
A greedy per-job rule cannot express that trade-off, which is why greedily
"duplicate whenever it looks profitable" *loses* throughput: it spends slots
that later jobs needed.

Adaptation to this environment
------------------------------
* **Cluster.**  ``S = 15`` servers, each holding one GPU of each of the
  ``A = 3`` accelerator types (k80, p100, v100), so ``S * A = 45`` GPUs.  The
  environment lets two jobs share a GPU.
* **GPU sharing as part of the configuration.**  Sia assumes exclusive GPUs,
  so a shared-GPU cluster needs the sharing decision to enter the goodput
  model.  Each GPU a configuration asks for therefore carries a *level*:

    - ``SOLO``  - the job has that GPU to itself, goodput ``T_i(X)``;
    - ``SHARE`` - the job shares that GPU with one other job, goodput
      ``T~_i(X)``, the mean of the Gavel co-located throughputs of job ``i``
      on ``X`` over the job types present in the queue.  This is exactly the
      kind of partner-agnostic profile Sia's Goodput Estimator produces: the
      scheduler knows sharing costs throughput without knowing the mate.

  The ``exclusive`` variant admits only ``SOLO`` and is Sia as published; the
  ``colocated`` variant admits both and is our shared-GPU extension.
* **Configurations.**  With ``L`` admissible levels, ``C`` contains

    - ``(1, 1, (X, l))``                one GPU;
    - ``(1, 2, (X, l1), (Y, l2))``      two GPUs on the *same* server, ``X != Y``
      (a server owns one GPU of each type);
    - ``(2, 2, (X, l1), (Y, l2))``      two GPUs on *different* servers.

  ``enable_distribution=False`` keeps only the single-GPU configurations,
  which is Sia restricted to rigid one-GPU jobs.
* **Goodput.**  Sia's goodput is throughput x statistical efficiency.  The
  Gavel tables model throughput only (no gradient statistics are available),
  so goodput collapses to throughput.  A two-GPU configuration is worth

      (1 - d) * (g(X, l1) + g(Y, l2)),   d = 0.05 same server, 0.15 across,

  where ``d`` is the environment's own distribution discount, playing the role
  of Sia's sub-linear scaling efficiency (its Figure 2).
* **Capacity.**  For each accelerator type ``X``, with ``solo_X`` GPUs handed
  out exclusively, ``share_X`` half-GPUs handed out to sharers and ``y_X`` the
  number of GPUs actually operated in shared mode,

      share_X <= 2 * y_X,        solo_X + y_X <= S.

  ``y_X`` is an auxiliary integer variable of the ILP, which makes "two
  sharers fit on one GPU" a linear constraint.
* **Restart factor.**  Every job is scheduled once and never re-allocated, so
  no configuration incurs a restart and ``r_i = 1`` (Eq. 3 with ``N_i = 0``).
* **Placement.**  The ILP picks *how many* GPUs of *which types and levels* a
  job gets; it does not pick individual GPUs, because Sia treats GPUs of one
  type as interchangeable.  A separate pass maps configurations onto concrete
  ``(server, accelerator)`` slots, most-constrained configurations first
  (same-server pairs, then cross-server pairs, then singletons), keeping
  enough GPUs in reserve for the requests not yet placed.  Among the legal
  slots it picks the one maximising the job's own throughput plus the
  externality it imposes on any incumbent, so idle GPUs win over shared ones
  and, when sharing is unavoidable, the least damaging neighbour wins.

Outputs ``results/job_scheduling/evaluation_scores_sia_*.json``.
"""
import json
import math
import os
from itertools import combinations_with_replacement, product

import numpy as np

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "scripts/sia_baseline.py needs SciPy (>= 1.9) for scipy.optimize.milp; "
        "install it with `pip install -r requirements.txt`."
    ) from exc

from nero.envs.job_scheduling.eval import Eval_JobSchedulingEnv
from nero.paths import SCORES, TEST_SETS

OUT = str(SCORES)
EPISODES = 20

P_FAIRNESS = -0.5   # Sia's default fairness knob
LAMBDA = 1.1        # queue-occupancy incentive (Eq. 2)

SOLO, SHARE = 0, 1


# --------------------------------------------------------------------------- #
# Configuration set C
# --------------------------------------------------------------------------- #
class Config:
    """A Sia configuration: ``n`` nodes and one ``(accelerator, level)`` per GPU."""

    __slots__ = ("nodes", "gpus")

    def __init__(self, nodes, gpus):
        self.nodes = nodes
        self.gpus = tuple(gpus)

    @property
    def m(self):
        return len(self.gpus)

    @property
    def rank(self):
        """Placement priority: most-constrained configurations first."""
        if self.m == 2 and self.nodes == 1:
            return 0                       # both GPUs must sit on one server
        if self.m == 2:
            return 1                       # needs two distinct servers
        return 2

    def demand(self, a, level):
        return sum(1 for (ga, gl) in self.gpus if ga == a and gl == level)

    def __repr__(self):
        names = {SOLO: "solo", SHARE: "share"}
        return f"({self.nodes}, {self.m}, {[(a, names[l]) for a, l in self.gpus]})"


def build_configs(A, enable_distribution=True, allow_share=True):
    """Enumerate the valid configuration set ``C`` (Sia Section 3.3)."""
    levels = (SOLO, SHARE) if allow_share else (SOLO,)
    units = [(a, l) for a in range(A) for l in levels]

    configs = [Config(1, (u,)) for u in units]
    if not enable_distribution:
        return configs

    # Two GPUs on one server: a server owns one GPU per type, so the two GPUs
    # must have different accelerator types.
    for a1 in range(A):
        for a2 in range(a1 + 1, A):
            for l1, l2 in product(levels, repeat=2):
                configs.append(Config(1, ((a1, l1), (a2, l2))))

    # Two GPUs on different servers: any unordered pair of units.
    for u1, u2 in combinations_with_replacement(units, 2):
        configs.append(Config(2, (u1, u2)))

    return configs


# --------------------------------------------------------------------------- #
# Throughput lookups straight from the Gavel tables
# --------------------------------------------------------------------------- #
def _solo_tp(problem, job_key, a):
    worker = problem.worker_types[a]
    return float(problem.physical_throughput_list[worker].get(job_key, {}).get("null", 0.0))


def _pair_tp(problem, job_key, other_key, a):
    """Throughput of ``job_key`` while sharing a type-``a`` GPU with ``other_key``."""
    worker = problem.worker_types[a]
    entry = problem.physical_throughput_list[worker].get(job_key, {}).get(other_key)
    return float(entry[0]) if entry else 0.0


def shared_tp_estimate(problem, job_key, a, queue_types):
    """Expected throughput on a *shared* type-``a`` GPU, averaged over the queue.

    Sia's Goodput Estimator profiles a job per GPU type, not per co-tenant, so
    the scheduler is told what sharing costs on average rather than which
    neighbour it will get.
    """
    vals = [_pair_tp(problem, job_key, other, a) for other in queue_types]
    vals = [v for v in vals if v > 0]
    return float(np.mean(vals)) if vals else 0.0


# --------------------------------------------------------------------------- #
# Steps 2-4: goodput matrix, row normalisation, fairness transform
# --------------------------------------------------------------------------- #
def goodput_matrix(env, configs):
    """Raw goodput matrix ``G`` of shape ``(J, |C|)`` (Sia Section 4.2)."""
    problem = env.problem
    queue_types = sorted(set(problem.jobs[:env.J]))

    # g[j, a, level] -- per-GPU goodput of job j on accelerator a at that level.
    g = np.zeros((env.J, env.A, 2), dtype=np.float64)
    for j in range(env.J):
        key = problem.jobs[j]
        for a in range(env.A):
            g[j, a, SOLO] = _solo_tp(problem, key, a)
            g[j, a, SHARE] = shared_tp_estimate(problem, key, a, queue_types)

    G = np.zeros((env.J, len(configs)), dtype=np.float64)
    for c, cfg in enumerate(configs):
        total = sum(g[:, a, l] for a, l in cfg.gpus)
        if cfg.m == 2:
            d = env.dist_discount_same if cfg.nodes == 1 else env.dist_discount_cross
            total = (1.0 - d) * total
        G[:, c] = total
        # A configuration is valid only if every GPU it asks for can run the job
        # (a zero Gavel entry means the job does not fit, e.g. ResNet-50 at
        # batch size 128 on a K80).
        for a, l in cfg.gpus:
            G[:, c] = np.where(g[:, a, l] > 0, G[:, c], 0.0)
    return G


def valid_mask(G):
    """Configurations each job can actually run in (Sia's *valid* set ``C``)."""
    return G > 0


def normalize_goodput(G, valid, n_min=1.0):
    """Sia Eq. 1: ``G[i, j] <- N_i^min * G[i, j] / min_j G[i, j]``.

    The row minimum runs over the job's valid configurations only; invalid ones
    are pinned to zero and never selectable.
    """
    masked = np.where(valid, G, np.inf)
    row_min = masked.min(axis=1, keepdims=True)
    row_min[~np.isfinite(row_min)] = 1.0     # job with no runnable configuration
    return np.where(valid, n_min * G / row_min, 0.0)


def utility_weights(G_norm, valid, p=P_FAIRNESS, lam=LAMBDA, r=None):
    """Per-(job, configuration) objective weight, in *maximise* orientation.

    Sia Eq. 4 maximises ``sum A_ij (r_i G_ij)^p + lambda (1 - ||A_i||_1)`` for
    ``p > 0`` and minimises the same expression for ``p < 0``.  Dropping the
    constant ``lambda |J|``, both reduce to maximising ``sum A_ij w_ij`` with
    the weights below, so a job is scheduled only when some configuration beats
    the ``lambda`` queueing penalty.
    """
    r = 1.0 if r is None else np.asarray(r, dtype=np.float64).reshape(-1, 1)
    U = np.where(valid, r * G_norm, 1.0)     # placeholder keeps the power finite
    if p > 0:
        W = np.power(U, p) - lam
    elif p < 0:
        W = lam - np.power(U, p)
    else:
        W = U - lam
    return np.where(valid, W, 0.0)


# --------------------------------------------------------------------------- #
# Step 5: the allocation ILP (Eq. 2 / Eq. 4)
# --------------------------------------------------------------------------- #
def solve_allocation(W, valid, configs, S, A, allow_share):
    """Solve Sia's binary ILP; return ``alloc[i] = config index`` or ``-1``.

    Variables are ``x[i, c] in {0, 1}`` plus, per accelerator type ``a``, an
    integer ``y[a]`` counting the GPUs of that type run in shared mode.
    """
    J, C = W.shape
    n_x = J * C
    n_var = n_x + A
    if J == 0:
        return np.zeros(0, dtype=int)

    cost = np.concatenate([-W.reshape(-1), np.zeros(A)])   # milp minimises

    n_con = J + 2 * A
    Amat = np.zeros((n_con, n_var), dtype=np.float64)
    ub = np.zeros(n_con, dtype=np.float64)

    # (1) each job takes at most one configuration:  sum_c x[i, c] <= 1
    for i in range(J):
        Amat[i, i * C:(i + 1) * C] = 1.0
        ub[i] = 1.0

    solo_demand = np.array([[cfg.demand(a, SOLO) for cfg in configs] for a in range(A)],
                           dtype=np.float64)
    share_demand = np.array([[cfg.demand(a, SHARE) for cfg in configs] for a in range(A)],
                            dtype=np.float64)

    for a in range(A):
        # (2) solo_a + y_a <= S
        row = J + a
        Amat[row, :n_x] = np.tile(solo_demand[a], J)
        Amat[row, n_x + a] = 1.0
        ub[row] = float(S)

        # (3) share_a - 2 y_a <= 0
        row = J + A + a
        Amat[row, :n_x] = np.tile(share_demand[a], J)
        Amat[row, n_x + a] = -2.0
        ub[row] = 0.0

    # Invalid (job, configuration) pairs are not part of C: pin them to zero.
    x_upper = valid.reshape(-1).astype(float)
    y_upper = np.full(A, float(S) if allow_share else 0.0)

    res = milp(
        c=cost,
        constraints=LinearConstraint(Amat, -np.inf, ub),
        integrality=np.ones(n_var),
        bounds=Bounds(np.zeros(n_var), np.concatenate([x_upper, y_upper])),
    )
    if not res.success or res.x is None:
        raise RuntimeError(f"Sia allocation ILP failed: {res.message}")

    X = np.asarray(res.x[:n_x]).round().astype(int).reshape(J, C)
    alloc = np.full(J, -1, dtype=int)
    picked = X.argmax(axis=1)
    alloc[X.max(axis=1) > 0] = picked[X.max(axis=1) > 0]
    return alloc


# --------------------------------------------------------------------------- #
# Placement: configurations -> concrete (server, accelerator) slots
# --------------------------------------------------------------------------- #
class _Cluster:
    """Grid bookkeeping that turns configurations into concrete GPU slots."""

    def __init__(self, env):
        self.S, self.A = env.S, env.A
        self.problem = env.problem
        self.mode = [[None] * self.A for _ in range(self.S)]     # None | SOLO | SHARE
        self.tenants = [[[] for _ in range(self.A)] for _ in range(self.S)]
        self.unused = [self.S] * self.A          # GPUs of each type not yet claimed
        self.half = [0] * self.A                 # shared GPUs holding a single job
        self.rem_solo = [0] * self.A             # requests still to be placed
        self.rem_share = [0] * self.A

    # -- capacity accounting ------------------------------------------------ #
    @staticmethod
    def _need(rem_solo, rem_share, half):
        """GPUs still required to serve the outstanding requests of one type."""
        return rem_solo + max(0, math.ceil((rem_share - half) / 2))

    def can_place(self, a, req_level, act_level, opening):
        """Would serving this request here still leave room for the rest?

        The ILP guarantees ``solo_a + ceil(share_a / 2) <= S`` per accelerator
        type, so honouring this check for every placement means no outstanding
        request is ever starved of a GPU.
        """
        unused = self.unused[a] - (1 if opening else 0)
        if unused < 0:
            return False
        rem_solo = self.rem_solo[a] - (1 if req_level == SOLO else 0)
        rem_share = self.rem_share[a] - (1 if req_level == SHARE else 0)
        if opening:
            half = self.half[a] + (1 if act_level == SHARE else 0)
        else:
            half = self.half[a] - 1
        return self._need(rem_solo, rem_share, half) <= unused

    # -- candidate slots ---------------------------------------------------- #
    def candidates(self, a, req_level, act_level, exclude=()):
        """Legal servers for one GPU request of type ``a`` run at ``act_level``."""
        out = []
        for s in range(self.S):
            if s in exclude:
                continue
            m = self.mode[s][a]
            if m is None:
                opening = True
            elif m == SHARE and act_level == SHARE and len(self.tenants[s][a]) < 2:
                opening = False
            else:
                continue                       # a SOLO GPU never takes a second job
            if self.can_place(a, req_level, act_level, opening):
                out.append(s)
        return out

    def score(self, job_key, s, a):
        """Own throughput plus the externality imposed on the GPU's incumbent."""
        tenants = self.tenants[s][a]
        if not tenants:
            return _solo_tp(self.problem, job_key, a)
        other = tenants[0]
        own = _pair_tp(self.problem, job_key, other, a)
        ext = _pair_tp(self.problem, other, job_key, a) - _solo_tp(self.problem, other, a)
        return own + ext

    def best(self, job_key, a, req_level, exclude=()):
        """Best server for one GPU request, relaxing its level if need be.

        Returns ``(server, actual_level)`` or ``None``.  The requested level is
        tried first; the other one is a fallback so that a job is never dropped
        just because the cluster ran out of GPUs in exactly that mode.
        """
        for act_level in (req_level, SHARE if req_level == SOLO else SOLO):
            cands = self.candidates(a, req_level, act_level, exclude=exclude)
            if cands:
                s = max(cands, key=lambda c: (self.score(job_key, c, a), -c))
                return s, act_level
        return None

    def occupy(self, job_key, s, a, req_level, act_level):
        if self.mode[s][a] is None:
            self.mode[s][a] = act_level
            self.unused[a] -= 1
            if act_level == SHARE:
                self.half[a] += 1
        else:
            self.half[a] -= 1                  # second tenant fills the GPU up
        self.tenants[s][a].append(job_key)
        self.release(a, req_level)

    def release(self, a, req_level):
        """Mark a request as resolved (placed, relaxed or abandoned)."""
        if req_level == SOLO:
            self.rem_solo[a] -= 1
        else:
            self.rem_share[a] -= 1


def place(env, alloc, configs):
    """Map the ILP's configurations onto concrete slots.

    Returns ``({job index: [(s, a), ...]}, stats)``.  Configurations are placed
    most-constrained first so that same-server pairs are not fragmented by
    singletons.

    The ILP constrains GPU *counts* per accelerator type but not server
    locality, so a same-server pair can turn out to be unseatable once earlier
    jobs have filled the cluster.  Rather than drop the job, placement degrades
    it: same server -> two servers -> a single GPU.  ``stats`` records how often
    that happened.
    """
    cluster = _Cluster(env)
    for j in range(env.J):
        if alloc[j] < 0:
            continue
        for a, level in configs[alloc[j]].gpus:
            if level == SOLO:
                cluster.rem_solo[a] += 1
            else:
                cluster.rem_share[a] += 1

    slots = {}
    stats = {"split_pairs": 0, "shrunk_to_one_gpu": 0, "unplaced": 0}
    order = sorted((j for j in range(env.J) if alloc[j] >= 0),
                   key=lambda j: (configs[alloc[j]].rank, j))

    for j in order:
        cfg = configs[alloc[j]]
        job_key = env.problem.jobs[j]

        if cfg.m == 1:
            a, level = cfg.gpus[0]
            hit = cluster.best(job_key, a, level)
            if hit is None:
                cluster.release(a, level)
                stats["unplaced"] += 1
                continue
            s, act = hit
            cluster.occupy(job_key, s, a, level, act)
            slots[j] = [(s, a)]
            continue

        (a1, l1), (a2, l2) = cfg.gpus

        # Same-server configurations first look for one server able to host both
        # GPUs; if none exists the pair is split across two servers instead.
        if cfg.nodes == 1:
            best_s, best_score, best_levels = None, -np.inf, None
            for act1 in (l1, SHARE if l1 == SOLO else SOLO):
                for act2 in (l2, SHARE if l2 == SOLO else SOLO):
                    ok1 = set(cluster.candidates(a1, l1, act1))
                    ok2 = set(cluster.candidates(a2, l2, act2))
                    for s in sorted(ok1 & ok2):
                        sc = cluster.score(job_key, s, a1) + cluster.score(job_key, s, a2)
                        if sc > best_score:
                            best_s, best_score, best_levels = s, sc, (act1, act2)
                if best_s is not None:
                    break
            if best_s is not None:
                act1, act2 = best_levels
                cluster.occupy(job_key, best_s, a1, l1, act1)
                cluster.occupy(job_key, best_s, a2, l2, act2)
                slots[j] = [(best_s, a1), (best_s, a2)]
                continue
            stats["split_pairs"] += 1

        hit1 = cluster.best(job_key, a1, l1)
        if hit1 is None:
            cluster.release(a1, l1)
            cluster.release(a2, l2)
            stats["unplaced"] += 1
            continue
        s1, act1 = hit1
        cluster.occupy(job_key, s1, a1, l1, act1)

        hit2 = cluster.best(job_key, a2, l2, exclude=(s1,))
        if hit2 is None:
            cluster.release(a2, l2)
            stats["shrunk_to_one_gpu"] += 1
            slots[j] = [(s1, a1)]
            continue
        s2, act2 = hit2
        cluster.occupy(job_key, s2, a2, l2, act2)
        slots[j] = [(s1, a1), (s2, a2)]

    return slots, stats


# --------------------------------------------------------------------------- #
# Episode execution
# --------------------------------------------------------------------------- #
def sia_schedule(env, colocated=True, enable_distribution=True,
                 p=P_FAIRNESS, lam=LAMBDA):
    """Run one Sia scheduling round over the whole queue held by ``env``.

    Returns ``(slots, stats)`` as produced by :func:`place`.
    """
    configs = build_configs(env.A, enable_distribution=enable_distribution,
                            allow_share=colocated)
    G_raw = goodput_matrix(env, configs)
    valid = valid_mask(G_raw)
    G = normalize_goodput(G_raw, valid)
    W = utility_weights(G, valid, p=p, lam=lam)
    alloc = solve_allocation(W, valid, configs, env.S, env.A, allow_share=colocated)
    return place(env, alloc, configs)


def execute(env, slots):
    """Materialise a Sia allocation in the environment and return its reward.

    Primary copies go in first (in job order), then the duplicates, so the
    environment sees each duplicate after the job it belongs to and charges the
    same-server / cross-server distribution discount correctly.
    """
    total = 0.0

    for j in sorted(slots):
        s, a = slots[j][0]
        env.current_job_idx = j
        _, reward, _, _, _ = env.step((s, a))
        total += float(sum(reward))

    for j in sorted(slots):
        if len(slots[j]) < 2:
            continue
        s, a = slots[j][1]
        dup = env.add_job(env.problem.jobs[j], original_job_idx=j)
        env.current_job_idx = dup
        _, reward, _, _, _ = env.step((s, a))
        total += float(sum(reward))

    return total


def evaluate_sia(colocated=True, enable_distribution=True, episodes=EPISODES,
                 p=P_FAIRNESS, lam=LAMBDA):
    """Evaluate one Sia variant on the held-out sets ``set_000 .. set_019``."""
    env = Eval_JobSchedulingEnv(str(TEST_SETS))
    env.set_idx = 0
    rewards, totals = [], {"split_pairs": 0, "shrunk_to_one_gpu": 0, "unplaced": 0}
    for _ in range(episodes):
        env.reset()
        slots, stats = sia_schedule(env, colocated=colocated,
                                    enable_distribution=enable_distribution,
                                    p=p, lam=lam)
        for k, v in stats.items():
            totals[k] += v
        rewards.append(execute(env, slots))
    return rewards, totals


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def report(name, rewards):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/evaluation_scores_{name}.json"
    ref = json.load(open(path)) if os.path.exists(path) else None
    with open(path, "w") as f:
        json.dump(rewards, f)
    m, s = float(np.mean(rewards)), float(np.std(rewards))
    msg = f"{name:22s} mean={m:8.4f} ± {s:6.4f} (pop)  n={len(rewards)}"
    if ref is not None and len(ref) == len(rewards):
        msg += f"  vs-committed diff={m - float(np.mean(ref)):+.4f}"
    print(msg)
    return m


if __name__ == "__main__":
    variants = [
        ("sia_original", False, False),
        ("sia_original_dist", False, True),
        ("sia_colocated", True, False),
        ("sia_colocated_dist", True, True),
    ]
    means = {}
    for name, colocated, dist in variants:
        rewards, stats = evaluate_sia(colocated=colocated, enable_distribution=dist)
        means[name] = report(name, rewards)
        if any(stats.values()):
            print(f"{'':22s} placement fallbacks over {EPISODES} episodes: {stats}")

    print("\ndistribution gain (Sia's multi-GPU configurations):")
    for base in ("sia_original", "sia_colocated"):
        print(f"  {base:16s} {means[base]:.4f} -> {means[base + '_dist']:.4f} "
              f"({means[base + '_dist'] - means[base]:+.4f})")
