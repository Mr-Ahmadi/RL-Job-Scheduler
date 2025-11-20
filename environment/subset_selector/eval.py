import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from ..job_scheduling.eval import Eval_JobSchedulingEnv
from ._requirments import flatten_obs, PPOAgent


class Eval_SubsetSelectorEnv(gym.Env):
    def __init__(self, set_dir):
        super().__init__()
        self.env = Eval_JobSchedulingEnv(set_dir=set_dir)

        state_dim = flatten_obs(self.env.reset()[0]).shape[0]
        action_dim = self.env.S * self.env.A

        self.primary_agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=1e-4,
            lr_critic=1e-3,
            gamma=0.99,
            lamda=0.95,
            clip=0.2,
            epochs=4,
            batch_size=64,
            checkpoint="models/job_scheduling/ppo/primary"
        )

        self.secondary_agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=1e-4,
            lr_critic=1e-3,
            gamma=0.99,
            lamda=0.95,
            clip=0.2,
            epochs=4,
            batch_size=64,
            checkpoint="models/job_scheduling/ppo/secondary"
        )

        self.action_space = spaces.Discrete(2)  # {0: skip, 1: duplicate}
        # self.observation_space = self.env.observation_space

    # ----------------------
    # Duplication rules
    # ----------------------
    def _can_duplicate(self, job_idx):
        """
        A job can be duplicated if:
        - It has already been assigned somewhere.
        - There is at least one (server, accelerator) pair with a free slot
        (colocation < 2), either on the same server or another server.
        """
        # Must already be assigned somewhere
        assigned = np.any(self.env.assignment[job_idx] > 0)
        if not assigned:
            return False

        # Check for any available resource (not just same server)
        for s in range(self.env.S):
            for a in range(self.env.A):
                if self.env.assignment[:, s, a].sum() < 2:  # free slot
                    return True
        return False

    def _schedule_with_primary_agent(self, dup_info=None):
        done = False
        state = flatten_obs(self.last_obs)
        total_reward = 0.0

        dup_initial_idx = dup_info['dup_idx'] if dup_info is not None else None
        original_job_idx = dup_info['original_job_idx'] if dup_info is not None else None
        for_time = dup_info.get('for_dup', False) if dup_info is not None else False

        while not done:
            apply_mask = (for_time and self.env.current_job_idx == dup_initial_idx)

            with torch.no_grad():
                a_idx, _, _ = self.primary_agent.choose_action(
                    state,
                    self.env,
                    for_dup=apply_mask,
                    original_job_idx=original_job_idx,
                )

            if a_idx is None:
                # no valid action -> stop scheduling this phase
                break

            s, a = divmod(a_idx, self.env.A)

            next_obs, rew, term, trunc, _ = self.env.step((s, a))
            r = float(np.sum(rew))
            done = bool(term or trunc)

            total_reward += r
            state = flatten_obs(next_obs) if not done else state
            self.last_obs = next_obs

        return total_reward


    def _schedule_with_secondary_agent(self, dup_info=None):
        done = False
        state = flatten_obs(self.last_obs)
        total_reward = 0.0

        dup_initial_idx = dup_info['dup_idx'] if dup_info is not None else None
        original_job_idx = dup_info['original_job_idx'] if dup_info is not None else None
        for_time = dup_info.get('for_dup', False) if dup_info is not None else False

        while not done:
            apply_mask = (for_time and self.env.current_job_idx == dup_initial_idx)

            with torch.no_grad():
                a_idx, _, _ = self.secondary_agent.choose_action(
                    state,
                    self.env,
                    for_dup=apply_mask,
                    original_job_idx=original_job_idx,
                )

            if a_idx is None:
                # no valid action -> stop scheduling this phase
                break

            s, a = divmod(a_idx, self.env.A)

            next_obs, rew, term, trunc, _ = self.env.step((s, a))
            r = float(np.sum(rew))
            done = bool(term or trunc)

            total_reward += r
            state = flatten_obs(next_obs) if not done else state
            self.last_obs = next_obs

        return total_reward

    # ----------------------
    # Outer loop API
    # ----------------------
    def reset(self, seed=None):
        super().reset(seed=seed)

        # Load pre-trained scheduler actor
        self.primary_agent.actor.load_state_dict(
            torch.load("models/job_scheduling/ppo/primary/actor.pth", weights_only=False)
        )
        self.primary_agent.actor.eval()
        
        # Load pre-trained scheduler actor
        self.secondary_agent.actor.load_state_dict(
            torch.load("models/job_scheduling/ppo/secondary/actor.pth", weights_only=False)
        )
        self.secondary_agent.actor.eval()

        # Reset environment
        self.last_obs, _ = self.env.reset()
        self.reward1, self.reward2 = 0.0, 0.0

        # Primary scheduling: assign all jobs once
        self.reward1 += self._schedule_with_primary_agent()

        # Reset to start outer loop
        self.env.current_job_idx = 0
        self.primary_J = self.env.J

        return self.env._get_obs(), {"total_reward": self.reward1 + self.reward2}

    def step(self, action):
        if self.env.current_job_idx >= self.primary_J:
            return self.env._get_obs(self.primary_J), 0.0, True, False, {
                "total_reward": self.reward1 + self.reward2
            }

        if action == 1 and self._can_duplicate(self.env.current_job_idx):
            # ✅ duplicate job
            current_job_idx = self.env.current_job_idx
            current_job = self.env.problem.jobs[current_job_idx]

            new_job_idx = self.env.add_job(current_job, original_job_idx=current_job_idx)

            # Schedule duplicate with masking
            self.env.current_job_idx = new_job_idx
            self.last_obs = self.env._get_obs()

            dup_info = {"for_dup": True, "original_job_idx": current_job_idx, "dup_idx": new_job_idx}
            reward_from_dup = self._schedule_with_secondary_agent(dup_info=dup_info)

            # Resume outer loop
            self.env.current_job_idx = current_job_idx + 1
            done_outer = (self.env.current_job_idx >= self.primary_J) or not self.env._has_available_resources()

            self.reward2 += reward_from_dup
            return self.env._get_obs(self.primary_J), reward_from_dup, done_outer, False, {
                "total_reward": self.reward1 + self.reward2
            }

        else:
            # ❌ fallback → skip
            self.env.current_job_idx += 1
            done_outer = self.env.current_job_idx >= self.primary_J
            return self.env._get_obs(self.primary_J), 0.0, done_outer, False, {
                "total_reward": self.reward1 + self.reward2
            }

    def render(self):
        self.env.render()
