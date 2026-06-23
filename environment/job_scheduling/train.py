import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .base import Base_JobSchedulingEnv


class Train_JobSchedulingEnv(Base_JobSchedulingEnv):
    def __init__(self, dist_discount_same=0.05, dist_discount_cross=0.15):
        super().__init__(dist_discount_same=dist_discount_same, dist_discount_cross=dist_discount_cross)

    def reset(self, job_list=None, seed=None):
        super().reset(seed=seed)
        self._init_problem(job_list)
        return self._get_obs(), {}
