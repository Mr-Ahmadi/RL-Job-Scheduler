import gymnasium as gym
import numpy as np
import torch

from .base import Base_SubsetSelectorEnv
from ._requirements import flatten_obs
from ..job_scheduling.eval import Eval_JobSchedulingEnv


class Eval_SubsetSelectorEnv(Base_SubsetSelectorEnv):
    def __init__(self, set_dir):
        super().__init__(set_dir=set_dir, learn_after_each_phase=False)

        self.env = Eval_JobSchedulingEnv(set_dir=set_dir)

        state_dim = flatten_obs(self.env.reset()[0]).shape[0]
        action_dim = self.env.S * self.env.A

        self._create_agents(state_dim, action_dim)

    def _schedule_with_primary_agent(self, dup_info=None):
        return self._schedule_with_agent(self.primary_agent, dup_info=dup_info, training=False)

    def _schedule_with_secondary_agent(self, dup_info=None):
        return self._schedule_with_agent(self.secondary_agent, dup_info=dup_info, training=False)

    def reset(self, seed=None):
        super().reset(seed=seed)

        self.primary_agent.actor.load_state_dict(
            torch.load("models/job_scheduling/ppo/actor.pth", weights_only=False)
        )
        self.primary_agent.actor.eval()

        self.secondary_agent.actor.load_state_dict(
            torch.load("models/job_scheduling/ppo/actor.pth", weights_only=False)
        )
        self.secondary_agent.actor.eval()

        self.last_obs, _ = self.env.reset()
        self.reward1 = 0.0
        self.reward2 = 0.0

        self.reward1 += self._schedule_with_primary_agent()

        return self.reset_common()
