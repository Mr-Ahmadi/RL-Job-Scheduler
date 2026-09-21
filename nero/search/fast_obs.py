"""Incremental next-observation builder.

Scoring a candidate placement with a learned value function needs the
observation the cluster *would* be in after that placement.  Calling
``env._get_obs()`` for every candidate costs ~0.9 ms each, which dominates the
decision latency.  The transition of the *observation* is pure bookkeeping,
though -- it never touches the throughput table -- so it can be applied as a
small delta on top of the current observation:

  * one GPU slot gains the current job's (one-hot, batch-size) pair;
  * that slot's occupancy, its server's load, and its server's distinct-model
    count move by a fixed amount;
  * the "current job" block advances to job ``j+1`` and the future-job statistics
    become the suffix statistics from ``j+2`` (precomputed once per episode).

``NextObsBuilder.next_obs`` is therefore O(A) rather than O(J * S * A), and is
exact: ``verify()`` (run this module as a script) compares it against a real
``env.step`` for every feasible slot along random rollouts.
"""

import numpy as np

from nero.envs.job_scheduling.base import MAX_JOBS_PER_GPU
from nero.paths import TEST_SETS


class NextObsBuilder:
    """Applies a candidate placement to an observation without stepping the env."""

    def __init__(self):
        self._key = None
        self._one_hot = None
        self._batch = None
        self._midx = None
        self._suf_min = None
        self._suf_max = None

    def _ensure(self, env):
        key = (id(env.problem), env.J)
        if self._key == key:
            return
        model_dim = len(env.model_names)
        J = env.J
        one_hot = np.zeros((J, model_dim), dtype=np.float32)
        batch = np.zeros(J, dtype=np.float32)
        midx = np.full(J, -1, dtype=np.int64)
        for j in range(J):
            oh, b, name = env._get_model_one_hot_and_batch_size(env.problem.jobs[j][0])
            one_hot[j] = oh
            batch[j] = b
            if name is not None:
                midx[j] = env.model_to_index[name]
        # suffix statistics: suf_*[i] summarises jobs i .. J-1
        suf_min = np.zeros((J + 1, model_dim), dtype=np.float32)
        suf_max = np.zeros((J + 1, model_dim), dtype=np.float32)
        for j in range(J - 1, -1, -1):
            suf_min[j] = suf_min[j + 1]
            suf_max[j] = suf_max[j + 1]
            i = midx[j]
            if i >= 0:
                b = batch[j]
                if suf_min[j, i] == 0 or b < suf_min[j, i]:
                    suf_min[j, i] = b
                if b > suf_max[j, i]:
                    suf_max[j, i] = b
        self._key = key
        self._one_hot, self._batch, self._midx = one_hot, batch, midx
        self._suf_min, self._suf_max = suf_min, suf_max

    def next_obs(self, env, base_obs, s, a):
        """Observation after placing ``env.current_job_idx`` at ``(s, a)``.

        Returns ``(obs, done)``; ``done`` mirrors the environment's terminal rule.
        """
        self._ensure(env)
        j = env.current_job_idx
        model_dim = len(env.model_names)

        gpu = base_obs["gpu_state"].copy()
        occ = base_obs["occupancy"].copy()
        load = base_obs["server_load"].copy()
        uniq = base_obs["server_model_unique"].copy()

        mi = int(self._midx[j])
        seen_on_server = False
        if mi >= 0:
            per_slot = gpu[s].reshape(env.A, MAX_JOBS_PER_GPU, model_dim + 1)
            seen_on_server = bool(np.any(per_slot[:, :, mi] > 0))

        for k in range(MAX_JOBS_PER_GPU):
            start = k * (model_dim + 1)
            if np.all(gpu[s, a, start:start + model_dim] == 0):
                gpu[s, a, start:start + model_dim] = self._one_hot[j]
                gpu[s, a, start + model_dim] = self._batch[j]
                occ[s, a] += 1.0
                break

        load[s] += 1.0 / (env.A * MAX_JOBS_PER_GPU)
        if mi >= 0 and not seen_on_server:
            uniq[s] += 1.0 / model_dim

        nxt = j + 1
        if nxt >= env.J:
            current_job = {"one_hot": np.zeros(model_dim, dtype=np.float32), "batch_size": 0.0}
        else:
            current_job = {"one_hot": self._one_hot[nxt], "batch_size": float(self._batch[nxt])}

        fs_idx = min(nxt + 1, env.J)
        future_stats = np.stack([self._suf_min[fs_idx], self._suf_max[fs_idx]], axis=1)

        obs = {
            "gpu_state": gpu,
            "current_job": current_job,
            "future_job_stats": future_stats,
            "jobs_left": (env.J - nxt) / max(env.J, 1),
            "occupancy": occ,
            "server_load": load,
            "server_model_unique": uniq,
        }
        done = bool(nxt >= env.J or not np.any(occ < MAX_JOBS_PER_GPU))
        return obs, done


def verify(episodes=4, seed=0, verbose=True):
    """Compare the delta-built observation with the environment's own observation.

    The reference is ``env._get_obs()`` evaluated on the post-placement state,
    i.e. exactly what ``env.step`` returns for that slot.
    """
    import random
    from nero.envs.job_scheduling.eval import Eval_JobSchedulingEnv
    from nero.agents.ppo import flatten_obs
    from nero.search.canonicalization import feasible_slots

    rng = random.Random(seed)
    builder = NextObsBuilder()
    env = Eval_JobSchedulingEnv(str(TEST_SETS))
    worst, checks = 0.0, 0
    for _ in range(episodes):
        env.reset()
        while env.current_job_idx < env.J:
            base = env._get_obs()
            slots = feasible_slots(env)
            if not slots:
                break
            j = env.current_job_idx
            for (s, a) in slots:
                fast, fast_done = builder.next_obs(env, base, s, a)
                env.assignment[j, s, a] = 1
                env.current_job_idx = j + 1
                true_obs = env._get_obs()
                true_done = bool(env.current_job_idx >= env.J
                                 or not env._has_available_resources())
                env.assignment[j, s, a] = 0
                env.current_job_idx = j
                worst = max(worst, float(np.max(np.abs(
                    flatten_obs(fast) - flatten_obs(true_obs)))))
                assert fast_done == true_done, f"done mismatch at slot {(s, a)}"
                checks += 1
            env.step(rng.choice(slots))
    if verbose:
        print(f"next-obs builder over {episodes} episodes: {checks} candidate states, "
              f"max |fast - env| = {worst:.3e}")
    return worst


if __name__ == "__main__":
    err = verify(episodes=4)
    assert err < 1e-6, f"next-obs builder disagrees with the environment (err={err})"
    print("OK: next-obs builder matches env.step exactly")
