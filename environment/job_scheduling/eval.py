import os
import json
import random
import numpy as np
from .base import Base_JobSchedulingEnv


class Eval_JobSchedulingEnv(Base_JobSchedulingEnv):
    def __init__(self, set_dir, dist_discount_same=0.05, dist_discount_cross=0.15):
        super().__init__(dist_discount_same=dist_discount_same, dist_discount_cross=dist_discount_cross)
        self.set_dir = set_dir
        self.set_files = sorted([
            os.path.join(self.set_dir, f) for f in os.listdir(self.set_dir) if f.endswith(".json")
        ])
        self.set_idx = 0

    def _load_next_job_set(self):
        if self.set_idx >= len(self.set_files):
            random.shuffle(self.set_files)
            self.set_idx = 0

        set_file = self.set_files[self.set_idx]
        with open(set_file, "r") as f:
            _job_list = json.load(f)
        self.set_idx += 1

        job_list = [(job, 1) for job in _job_list]
        return job_list

    def reset(self, _job_list=None, seed=None):
        super().reset(seed=seed)

        if _job_list is None:
            job_list = self._load_next_job_set()
        else:
            job_list = _job_list

        self._init_problem(job_list)
        return self._get_obs(), {}
