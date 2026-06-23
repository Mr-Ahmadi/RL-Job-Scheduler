import gymnasium as gym
from gymnasium import spaces
import numpy as np
import re
from .._problem.problem import ProblemDescription, model_names, batch_sizes_range


class Base_JobSchedulingEnv(gym.Env):
    def __init__(self, dist_discount_same=0.05, dist_discount_cross=0.15):
        self.model_names = model_names
        self.model_to_index = {name: i for i, name in enumerate(model_names)}
        self.batch_sizes_range = batch_sizes_range

        self.problem = None
        self.S = None
        self.A = None
        self.J = None
        self.assignment = None
        self.current_job_idx = 0

        self.duplicate_of = None
        self.distributed_flag = None

        self.dist_discount_same = dist_discount_same
        self.dist_discount_cross = dist_discount_cross

        self.action_space = None

    def add_job(self, job, original_job_idx=-1):
        if self.problem is None:
            raise ValueError("Problem not initialized. Call reset() first.")

        new_job_idx = self.problem.add_job(job)

        old_J = self.J
        self.J = self.problem.J
        old_assignment = self.assignment
        self.assignment = np.zeros((self.J, self.S, self.A), dtype=np.int8)
        self.assignment[:old_J, :, :] = old_assignment

        if self.duplicate_of is None:
            self.duplicate_of = np.full(self.J, -1, dtype=int)
        else:
            new_dup = np.full(self.J, -1, dtype=int)
            new_dup[:old_J] = self.duplicate_of
            self.duplicate_of = new_dup

        if self.distributed_flag is None:
            self.distributed_flag = np.zeros(self.J, dtype=bool)
        else:
            new_flag = np.zeros(self.J, dtype=bool)
            new_flag[:old_J] = self.distributed_flag
            self.distributed_flag = new_flag

        if original_job_idx is not None and original_job_idx >= 0:
            self.duplicate_of[new_job_idx] = original_job_idx
        else:
            self.duplicate_of[new_job_idx] = -1

        return new_job_idx

    def _init_problem(self, job_list):
        self.problem = ProblemDescription(jobs=job_list)

        self.S = self.problem.S
        self.A = self.problem.A
        self.J = self.problem.J
        self.current_job_idx = 0
        self.assignment = np.zeros((self.J, self.S, self.A), dtype=np.int8)

        self.duplicate_of = np.full(self.J, -1, dtype=int)
        self.distributed_flag = np.zeros(self.J, dtype=bool)

        self.action_space = spaces.MultiDiscrete([self.S, self.A])

    def _get_model_one_hot_and_batch_size(self, model_str):
        model_name = None
        for name in self.model_names:
            if model_str.startswith(name):
                model_name = name
                break

        one_hot = np.zeros(len(self.model_names), dtype=np.float32)
        if model_name is not None:
            one_hot[self.model_to_index[model_name]] = 1.0

        match = re.search(r"batch size (\d+)", model_str)
        batch_size = int(match.group(1)) if match else 0

        if model_name in self.batch_sizes_range:
            min_b, max_b = self.batch_sizes_range[model_name]
            batch_size = (batch_size - min_b) / (max_b - min_b)
            batch_size = batch_size * 0.9 + 0.1
            batch_size = np.clip(batch_size, 0.0, 1.0)

        return one_hot, float(batch_size), model_name

    def _has_available_resources(self):
        for s in range(self.S):
            for a in range(self.A):
                colocated_jobs = sum(self.assignment[j, s, a] for j in range(self.J))
                if colocated_jobs < 2:
                    return True
        return False

    def get_distribution_discount(self, j, s):
        if self.duplicate_of[j] == -1:
            if not self.distributed_flag[j]:
                return 0.0
            else:
                duplicates = [dj for dj in range(self.J) if self.duplicate_of[dj] == j]
                assigned_servers = set()
                for dj in duplicates:
                    for so in range(self.S):
                        for ao in range(self.A):
                            if self.assignment[dj, so, ao]:
                                assigned_servers.add(so)
                if (s in assigned_servers) and len(assigned_servers) == 1:
                    discount = self.dist_discount_same
                else:
                    discount = self.dist_discount_cross
                return discount
        else:
            orig = self.duplicate_of[j]
            assigned_servers = set()
            for so in range(self.S):
                for ao in range(self.A):
                    if self.assignment[orig, so, ao]:
                        assigned_servers.add(so)
            if not assigned_servers:
                return 0.0

            if (s in assigned_servers) and len(assigned_servers) == 1:
                discount = self.dist_discount_same
            else:
                discount = self.dist_discount_cross

            return discount

    def _get_obs(self, job_limit=None):
        if job_limit is None:
            job_limit = self.J

        model_dim = len(self.model_names)
        max_jobs_per_gpu = 2
        gpu_obs = np.zeros((self.S, self.A, max_jobs_per_gpu * (model_dim + 1)), dtype=np.float32)
        occupancy = np.zeros((self.S, self.A), dtype=np.float32)

        for j in range(min(self.J, job_limit)):
            for s in range(self.S):
                for a in range(self.A):
                    if self.assignment[j, s, a]:
                        model_str, _ = self.problem.jobs[j]
                        one_hot, batch_size, _ = self._get_model_one_hot_and_batch_size(model_str)
                        for k in range(max_jobs_per_gpu):
                            start = k * (model_dim + 1)
                            if np.all(gpu_obs[s, a, start:start+model_dim] == 0):
                                gpu_obs[s, a, start:start+model_dim] = one_hot
                                gpu_obs[s, a, start+model_dim] = batch_size
                                occupancy[s, a] += 1.0
                                break

        j = self.current_job_idx
        if j >= job_limit:
            current_job = {"one_hot": [0, 0, 0, 0, 0], "batch_size": 0}
        else:
            model_str, _ = self.problem.jobs[j]
            one_hot, batch_size, _ = self._get_model_one_hot_and_batch_size(model_str)
            current_job = {"one_hot": one_hot, "batch_size": batch_size}

        future_stats = np.zeros((model_dim, 2), dtype=np.float32)
        for jj in range(j + 1, job_limit):
            m_str, _ = self.problem.jobs[jj]
            _, b, m_name = self._get_model_one_hot_and_batch_size(m_str)
            if m_name is not None:
                idx = self.model_to_index[m_name]
                if future_stats[idx, 0] == 0 or b < future_stats[idx, 0]:
                    future_stats[idx, 0] = b
                if b > future_stats[idx, 1]:
                    future_stats[idx, 1] = b

        server_load = np.zeros(self.S, dtype=np.float32)
        server_model_set = [set() for _ in range(self.S)]
        for s in range(self.S):
            for a in range(self.A):
                server_load[s] += occupancy[s, a]
                for j in range(min(self.J, job_limit)):
                    if self.assignment[j, s, a]:
                        m_str, _ = self.problem.jobs[j]
                        _, _, m_name = self._get_model_one_hot_and_batch_size(m_str)
                        if m_name is not None:
                            server_model_set[s].add(m_name)
        server_load /= (self.A * max_jobs_per_gpu)
        server_model_unique = np.array([len(s) for s in server_model_set], dtype=np.float32) / model_dim

        return {
            "gpu_state": gpu_obs,
            "current_job": current_job,
            "future_job_stats": future_stats,
            "jobs_left": (job_limit - self.current_job_idx) / max(job_limit, 1),
            "occupancy": occupancy,
            "server_load": server_load,
            "server_model_unique": server_model_unique,
        }

    def step(self, action):
        s, a = action
        j = self.current_job_idx

        if j >= self.J:
            return self._get_obs(), (0.0, 0.0), True, False, {}

        if self.assignment[j].sum() > 0:
            return self._get_obs(), (-1000.0, 0.0), False, False, {}

        colocated = [oj for oj in range(self.J) if oj != j and self.assignment[oj, s, a]]
        if len(colocated) >= 2:
            return self._get_obs(), (-1000.0, 0.0), False, False, {}

        prev_tp = {oj: self._estimate_job_throughput_given_combination(oj, s, a)
                   for oj in colocated}

        self.assignment[j, s, a] = 1
        new_tp = self._estimate_job_throughput_given_combination(j, s, a)

        upd_tp = {oj: self._estimate_job_throughput_given_combination(oj, s, a)
                  for oj in colocated}

        delta = sum((upd_tp[oj] - prev_tp[oj]) * (1 - self.get_distribution_discount(oj, s))
                    for oj in colocated) if colocated else 0.0

        if self.duplicate_of is not None and self.duplicate_of[j] != -1:
            orig = int(self.duplicate_of[j])
            tr_orig = 0.0
            orig_positions = [(so, ao)
                            for so in range(self.S)
                            for ao in range(self.A)
                            if self.assignment[orig, so, ao]]
            if orig_positions:
                so, ao = orig_positions[0]
                tr_orig = self._estimate_job_throughput_given_combination(orig, so, ao)

            tr_dup = new_tp

            same_server = False
            if orig_positions:
                servers_of_orig = {so for (so, ao) in orig_positions}
                same_server = (s in servers_of_orig)

            if orig < len(self.distributed_flag) and not self.distributed_flag[orig]:
                if same_server:
                    discount = self.dist_discount_same
                else:
                    discount = self.dist_discount_cross

                delta -= discount * (tr_orig + tr_dup)

                self.distributed_flag[orig] = True

        reward = (float(new_tp) / 100, float(delta) / 100)

        self.current_job_idx += 1
        done = (self.current_job_idx >= self.J) or not self._has_available_resources()

        return self._get_obs(), reward, done, False, {}

    def _estimate_job_throughput_given_combination(self, j, s, a):
        colocated = [oj for oj in range(self.J) if oj != j and self.assignment[oj, s, a]]
        comb = tuple(sorted(colocated + [j]))
        idx = self.problem._comb_to_idx.get(comb)
        if idx is not None:
            return self.problem.Tr[j][idx, a]
        return 0.0

    def render(self):
        print("\nFinal Job Assignments:")
        for j in range(self.J):
            for s in range(self.S):
                for a in range(self.A):
                    if self.assignment[j, s, a]:
                        print(f"  Job {j} \u2192 Server {s}, Accelerator {a}")
