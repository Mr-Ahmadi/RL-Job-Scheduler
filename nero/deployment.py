"""Deployable online inference for the trained scheduling agents.

All inference runs in eager fp32 on CPU. For this problem size the compute is
tiny; eager CPU dispatch is the lowest-overhead backend available here (the
MPS/GPU path pays a large per-call dispatch cost and torch.compile/INT8 are
not available in this torch build). Buffers are reused across decisions to
avoid per-call allocations.

Agents exposed:
  * OnlineActor      -- inner agent: job -> (server, accelerator) placement.
  * SubsetSelector   -- outer agent: job state -> duplicate? (0/1).
  * LearnedTopKActor -- re-exported from ``nero.search.topk``: the same inner
    placement decision, but searching over the policy's top-K candidates and
    picking among them with learned heads instead of the throughput table.  It
    costs more per decision than ``OnlineActor`` and scores materially higher
    (17.48 vs 15.15 on the held-out sets).
"""

import numpy as np
import torch

from nero.envs.job_scheduling.eval import Eval_JobSchedulingEnv
from nero.agents.ppo import PPOAgent, flatten_obs
from nero.envs.subset_selector.common import flatten_obs_subset
from nero.envs.subset_selector.eval import Eval_SubsetSelectorEnv

from nero.search.topk import LearnedTopKActor
from nero.paths import INNER, OUTER, TEST_SETS

__all__ = ["OnlineActor", "SubsetSelector", "LearnedTopKActor"]


def _valid_slots(env, j):
    """Feasible slots for job ``j`` -- the environment owns this rule."""
    return env.feasible_slots(j)


class OnlineActor:
    def __init__(self, checkpoint_dir=str(INNER), threads=1):
        torch.set_num_threads(threads)
        probe = Eval_JobSchedulingEnv(str(TEST_SETS))
        probe.reset()
        state_dim = len(flatten_obs(probe._get_obs()))
        action_dim = probe.S * probe.A
        self._agent = PPOAgent(state_dim, action_dim, lr_actor=5e-4, lr_critic=1e-3,
                               gamma=0.99, lamda=0.95, clip=0.2, epochs=10,
                               batch_size=128, checkpoint_dir=checkpoint_dir)
        self._agent.load()
        self.actor = self._agent.actor.to("cpu").float().eval()
        self._obs_buf = None
        self._mask_buf = None

    def decide(self, env):
        if env.current_job_idx >= env.J:
            return None
        obs = flatten_obs(env._get_obs())
        if self._obs_buf is None or self._obs_buf.size != obs.size:
            self._obs_buf = np.empty(obs.size, dtype=np.float32)
        self._obs_buf[:] = obs
        st = torch.from_numpy(self._obs_buf).unsqueeze(0)
        with torch.no_grad():
            _, probs = self.actor(st)
        if self._mask_buf is None or self._mask_buf.shape != probs.shape:
            self._mask_buf = torch.zeros_like(probs)
        self._mask_buf.zero_()
        for (s, a) in _valid_slots(env, env.current_job_idx):
            self._mask_buf[0, s * env.A + a] = probs[0, s * env.A + a]
        return divmod(int(torch.argmax(self._mask_buf, dim=-1).item()), env.A)


class SubsetSelector:
    def __init__(self, checkpoint_dir=str(OUTER), threads=1):
        torch.set_num_threads(threads)
        probe = Eval_SubsetSelectorEnv(str(TEST_SETS))
        sample_obs, _ = probe.reset()
        state_dim = len(flatten_obs_subset(sample_obs))
        self._agent = PPOAgent(state_dim, 2, lr_actor=3e-4, lr_critic=1e-3,
                               gamma=0.99, lamda=0.95, clip=0.2, epochs=8,
                               batch_size=64, checkpoint_dir=checkpoint_dir)
        self._agent.load()
        self.actor = self._agent.actor.to("cpu").float().eval()
        self._obs_buf = None

    def decide(self, obs):
        if self._obs_buf is None or self._obs_buf.size != obs.size:
            self._obs_buf = np.empty(obs.size, dtype=np.float32)
        self._obs_buf[:] = obs
        st = torch.from_numpy(self._obs_buf).unsqueeze(0)
        with torch.no_grad():
            _, probs = self.actor(st)
        if obs[-1] <= 0.5:
            return 0
        return int(torch.argmax(probs, dim=-1).item())
