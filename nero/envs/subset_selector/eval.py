import torch

from .base import Base_SubsetSelectorEnv
from nero.envs.subset_selector.common import flatten_obs, PPOAgent
from ..job_scheduling.eval import Eval_JobSchedulingEnv
from nero.paths import INNER, PRIMARY, SECONDARY


class Eval_SubsetSelectorEnv(Base_SubsetSelectorEnv):
    def __init__(self, set_dir):
        super().__init__(set_dir=set_dir, learn_after_each_phase=False)

        self.env = Eval_JobSchedulingEnv(set_dir=set_dir)

        state_dim = flatten_obs(self.env.reset()[0]).shape[0]
        action_dim = self.env.S * self.env.A

        self.primary_agent = PPOAgent(
            state_dim=state_dim, action_dim=action_dim,
            lr_actor=1e-4, lr_critic=1e-3,
            gamma=0.99, lamda=0.95, clip=0.2, epochs=4, batch_size=64,
            checkpoint=str(PRIMARY),  # never written: frozen
        )
        self.secondary_agent = PPOAgent(
            state_dim=state_dim, action_dim=action_dim,
            lr_actor=1e-4, lr_critic=1e-3,
            gamma=0.99, lamda=0.95, clip=0.2, epochs=4, batch_size=64,
            checkpoint=str(SECONDARY)
        )

    def _schedule_with_primary_agent(self, dup_info=None):
        return self._schedule_with_agent(self.primary_agent, dup_info=dup_info, training=False)

    def _schedule_with_secondary_agent(self, dup_info=None):
        return self._schedule_with_agent(self.secondary_agent, dup_info=dup_info, training=False)

    def reset(self, seed=None):
        super().reset(seed=seed)

        self.primary_actor_override = getattr(self, "primary_actor_override", None)
        self.secondary_actor_override = getattr(self, "secondary_actor_override", None)

        if self.primary_actor_override is not None:
            self.primary_agent.actor = self.primary_actor_override
        else:
            self.primary_agent.actor.load_state_dict(
                torch.load(str(INNER / "actor.pth"), weights_only=False)
            )
        self.primary_agent.actor.eval()

        try:
            if self.secondary_actor_override is not None:
                self.secondary_agent.actor = self.secondary_actor_override
            else:
                self.secondary_agent.actor.load_state_dict(
                    torch.load(str(SECONDARY / "actor.pth"), weights_only=False)
                )
        except (FileNotFoundError, OSError):
            self.secondary_agent.actor.load_state_dict(
                torch.load(str(INNER / "actor.pth"), weights_only=False)
            )
        self.secondary_agent.actor.eval()

        self.last_obs, _ = self.env.reset()
        self.reward1 = 0.0
        self.reward2 = 0.0

        self.reward1 += self._schedule_with_primary_agent()

        return self.reset_common()
