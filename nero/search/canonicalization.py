"""Slot canonicalisation: collapse (server, accelerator) slots that are provably
interchangeable for the current job.

The environment exposes ``S * A = 45`` slots, but the reward of placing job ``j``
at slot ``(s, a)`` is a function of only three observable quantities:

  * the accelerator type ``a``;
  * the multiset of job types already co-located on that slot, each paired with
    the distribution discount that applies to it on server ``s``;
  * the distribution discount that applies to ``j`` itself on server ``s``
    (non-zero only when ``j`` is a duplicate of an already-placed job).

Two slots sharing that key therefore yield *exactly* the same immediate reward,
so a top-K search only ever needs to look at one representative per key.  None of
the three quantities requires the throughput table -- they come from the
assignment bookkeeping the scheduler already maintains -- so canonicalisation is
usable online.  ``verify()`` (run this module as a script) checks the exactness
claim against the environment's true reward.

Collapsing typically takes 45 feasible slots down to a handful of classes, which
is what makes an *evaluated* top-K affordable: K distinct candidates instead of K
near-duplicates of the same placement.

``slot_features`` returns the same key as a fixed-width float vector.  Because
the key is *sufficient* for the reward, a model that consumes it only has to
learn a small shared function instead of 45 independent outputs of a 632-wide
observation -- which is what makes the learned reward model in
``nero/search/heads.py`` accurate enough to stand in for the oracle.
"""

import numpy as np

from nero.envs.job_scheduling.base import MAX_JOBS_PER_GPU
from nero.paths import TEST_SETS


def slot_occupancy(env):
    """(s, a) -> list of job indices currently placed there."""
    return env.slot_occupancy()


def slot_key(env, j, s, a, occupants=None):
    """Reward-equivalence key of slot ``(s, a)`` for job ``j``.

    Returns ``None`` when the slot is infeasible (already holds two jobs).
    """
    if occupants is None:
        occupants = [oj for oj in range(env.J) if oj != j and env.assignment[oj, s, a]]
    else:
        occupants = [oj for oj in occupants if oj != j]
    if len(occupants) >= MAX_JOBS_PER_GPU:
        return None
    others = tuple(sorted(
        (env.problem.jobs[oj], round(float(env.get_distribution_discount(oj, s)), 9))
        for oj in occupants
    ))
    if env.duplicate_of is not None and env.duplicate_of[j] != -1:
        own = round(float(env.get_distribution_discount(j, s)), 9)
    else:
        own = 0.0
    return (int(a), others, own)


def canonical_classes(env, j=None):
    """Group the feasible slots for job ``j`` into reward-equivalence classes.

    Returns ``{key: [(s, a), ...]}`` preserving slot order within each class.
    """
    if j is None:
        j = env.current_job_idx
    classes = {}
    if j >= env.J or env.assignment[j].sum() > 0:
        return classes
    occ = slot_occupancy(env)
    for s in range(env.S):
        for a in range(env.A):
            key = slot_key(env, j, s, a, occupants=occ.get((s, a), ()))
            if key is None:
                continue
            classes.setdefault(key, []).append((s, a))
    return classes


def feasible_slots(env, j=None):
    """Flat list of feasible slots, in the same order canonicalisation sees them."""
    return env.feasible_slots(j)


# feature layout: current job (one-hot + batch) | accelerator one-hot |
# own distribution discount | occupant (present, one-hot, batch, discount)
def feature_dim(env):
    model_dim = len(env.model_names)
    return (model_dim + 1) + env.A + 1 + (1 + model_dim + 1 + 1)


def slot_features(env, j, slots, occ=None):
    """Numeric encoding of ``slot_key`` for each slot in ``slots``.

    A feasible slot holds at most one job (two would leave no room), so a single
    occupant block is enough.  The encoding is exact for single-copy placement.
    For a *duplicate* placement the reward also depends on the throughput of the
    original copy, which is not a function of this key; the own-discount feature
    marks those decisions but does not fully determine them.
    """
    model_dim = len(env.model_names)
    if occ is None:
        occ = slot_occupancy(env)
    out = np.zeros((len(slots), feature_dim(env)), dtype=np.float32)
    job_oh, job_b, _ = env._get_model_one_hot_and_batch_size(env.problem.jobs[j][0])
    is_dup = env.duplicate_of is not None and env.duplicate_of[j] != -1
    base = model_dim + 1
    for i, (s, a) in enumerate(slots):
        out[i, :model_dim] = job_oh
        out[i, model_dim] = job_b
        out[i, base + a] = 1.0
        out[i, base + env.A] = (float(env.get_distribution_discount(j, s))
                                if is_dup else 0.0)
        others = [oj for oj in occ.get((s, a), ()) if oj != j]
        if others:
            oj = others[0]
            o_oh, o_b, _ = env._get_model_one_hot_and_batch_size(env.problem.jobs[oj][0])
            k = base + env.A + 1
            out[i, k] = 1.0
            out[i, k + 1:k + 1 + model_dim] = o_oh
            out[i, k + 1 + model_dim] = o_b
            out[i, k + 2 + model_dim] = float(env.get_distribution_discount(oj, s))
    return out


def dense_rewards(env, j=None):
    """True reward of every feasible slot, straight from the throughput tables.

    Offline supervision only.  Equivalent to calling ``nero.evaluation.true_score``
    per slot, but it exploits the fact that a feasible slot holds at most one job,
    so each entry is two table lookups instead of a rescan of the assignment.
    """
    if j is None:
        j = env.current_job_idx
    occ = slot_occupancy(env)
    n = env.S * env.A
    r = np.zeros(n, dtype=np.float32)
    m = np.zeros(n, dtype=bool)
    idx = env.problem._comb_to_idx
    solo_j = idx[(j,)]
    for s in range(env.S):
        for a in range(env.A):
            others = [oj for oj in occ.get((s, a), ()) if oj != j]
            if len(others) >= MAX_JOBS_PER_GPU:
                continue
            if not others:
                total = env.problem.Tr[j][solo_j, a]
            else:
                oj = others[0]
                pair = idx[tuple(sorted((j, oj)))]
                new_tp = env.problem.Tr[j][pair, a]
                prev = env.problem.Tr[oj][idx[(oj,)], a]
                upd = env.problem.Tr[oj][pair, a]
                total = new_tp + (upd - prev) * (1 - env.get_distribution_discount(oj, s))
            r[s * env.A + a] = total / 100.0
            m[s * env.A + a] = True
    return r, m


def verify(episodes=5, seed=0, verbose=True):
    """Check that every slot in a class has exactly the same true reward.

    Uses the oracle reward (offline only) purely as ground truth for the check.
    """
    import random
    from nero.envs.job_scheduling.eval import Eval_JobSchedulingEnv
    from nero.evaluation import true_score

    rng = random.Random(seed)
    env = Eval_JobSchedulingEnv(str(TEST_SETS))
    n_classes, n_slots, worst, worst_dense = 0, 0, 0.0, 0.0
    for _ in range(episodes):
        env.reset()
        while env.current_job_idx < env.J:
            j = env.current_job_idx
            classes = canonical_classes(env, j)
            if not classes:
                break
            for key, members in classes.items():
                vals = []
                for (s, a) in members:
                    nt, d = true_score(env, j, s, a)
                    vals.append(nt + d)
                worst = max(worst, float(np.max(vals) - np.min(vals)))
                n_classes += 1
                n_slots += len(members)
            flat = [sa for members in classes.values() for sa in members]
            assert sorted(flat) == sorted(feasible_slots(env, j)), "class cover mismatch"
            fast_r, fast_m = dense_rewards(env, j)
            for (s, a) in flat:
                nt, d = true_score(env, j, s, a)
                assert fast_m[s * env.A + a]
                worst_dense = max(worst_dense,
                                  abs(float(fast_r[s * env.A + a]) - (nt + d) / 100.0))
            env.step(rng.choice(flat))
    if verbose:
        print(f"canonicalisation over {episodes} episodes: "
              f"{n_slots} slots -> {n_classes} classes "
              f"({n_slots / max(n_classes, 1):.2f} slots/class), "
              f"max intra-class reward spread = {worst:.3e}, "
              f"max dense-label error = {worst_dense:.3e}")
    return worst, worst_dense


if __name__ == "__main__":
    spread, dense_err = verify(episodes=5)
    assert spread < 1e-9, f"canonicalisation is not exact (spread={spread})"
    assert dense_err < 1e-5, f"dense labels disagree with the oracle ({dense_err})"
    print("OK: canonicalisation is exact and dense labels match the oracle")
