import gymnasium as gym
import numpy as np
import torch

from .base import Base_SubsetSelectorEnv
from ._requirements import flatten_obs, PPOAgent
from ..job_scheduling.train import Train_JobSchedulingEnv


class Train_SubsetSelectorEnv(Base_SubsetSelectorEnv):
    def __init__(self, learn_after_each_phase: bool = False):
        super().__init__(set_dir=None, learn_after_each_phase=learn_after_each_phase)

        self.env = Train_JobSchedulingEnv()

        obs, _ = self.env.reset()
        state_dim = flatten_obs(obs).shape[0]
        action_dim = self.env.S * self.env.A

        self.primary_agent = PPOAgent(
            state_dim=state_dim, action_dim=action_dim,
            lr_actor=1e-4, lr_critic=1e-3,
            gamma=0.99, lamda=0.95, clip=0.2, epochs=4, batch_size=64,
            checkpoint="models/job_scheduling/ppo/primary"
        )
        self.secondary_agent = PPOAgent(
            state_dim=state_dim, action_dim=action_dim,
            lr_actor=3e-5, lr_critic=3e-4,
            gamma=0.99, lamda=0.95, clip=0.2, epochs=10, batch_size=128,
            checkpoint="models/job_scheduling/ppo/secondary",
            ent_start=0.01, ent_end=0.001, ent_decay=0.9995,
            max_grad_norm=0.5,
            lr_actor_decay=0.9999, lr_critic_decay=0.9999,
        )

        self.primary_agent.actor.load_state_dict(
            torch.load("models/job_scheduling/ppo/actor.pth", weights_only=False)
        )
        self.primary_agent.critic.load_state_dict(
            torch.load("models/job_scheduling/ppo/critic.pth", weights_only=False)
        )

        self.secondary_agent.actor.load_state_dict(
            torch.load("models/job_scheduling/ppo/actor.pth", weights_only=False)
        )
        self.secondary_agent.critic.load_state_dict(
            torch.load("models/job_scheduling/ppo/critic.pth", weights_only=False)
        )

    def _schedule_with_primary_agent(self, dup_info=None):
        return self._schedule_with_agent(self.primary_agent, dup_info=dup_info, training=False)

    def _schedule_with_secondary_agent(self, dup_info=None):
        return self._schedule_with_agent(self.secondary_agent, dup_info=dup_info, training=True)

    def learn_secondary(self):
        if len(self.secondary_agent.memory.states) == 0:
            return
        self.secondary_agent.learn()
        self.secondary_agent.save()

    def reset(self, seed=None):
        super().reset(seed=seed)

        self.last_obs, _ = self.env.reset()
        self.reward1 = 0.0
        self.reward2 = 0.0

        self.reward1 += self._schedule_with_primary_agent()

        return self.reset_common()
