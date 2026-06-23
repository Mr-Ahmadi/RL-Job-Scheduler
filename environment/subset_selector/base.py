import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from ._requirements import flatten_obs, flatten_obs_subset, PPOAgent


class Base_SubsetSelectorEnv(gym.Env):
    def __init__(self, set_dir=None, learn_after_each_phase=True):
        super().__init__()
        self.learn_after_each_phase = learn_after_each_phase
        self.set_dir = set_dir

        self.env = None
        self.primary_agent = None
        self.secondary_agent = None
        self.last_obs = None
        self.reward1 = 0.0
        self.reward2 = 0.0
        self.primary_J = 0

        self.action_space = spaces.Discrete(2)
        self.dup_count = 0

    def _create_agents(self, state_dim, action_dim):
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

    def _can_duplicate(self, job_idx):
        assigned = np.any(self.env.assignment[job_idx] > 0)
        if not assigned:
            return False

        for s in range(self.env.S):
            for a in range(self.env.A):
                if self.env.assignment[:, s, a].sum() < 2:
                    return True
        return False

    def _schedule_with_agent(self, agent, dup_info=None, training=True):
        dup_initial_idx = dup_info['dup_idx'] if dup_info is not None and 'dup_idx' in dup_info else None
        original_job_idx = dup_info.get('original_job_idx') if dup_info is not None else None
        for_time = dup_info.get('for_dup', False) if dup_info is not None else False

        if not training:
            agent.actor.eval()
            agent.critic.eval()

        done = False
        state = flatten_obs(self.last_obs)
        total_reward = 0.0

        while not done:
            apply_dup_mask = (for_time and (self.env.current_job_idx == dup_initial_idx))

            if training:
                a_idx, logp, val = agent.choose_action(
                    state, self.env,
                    for_dup=apply_dup_mask,
                    original_job_idx=original_job_idx,
                )
            else:
                with torch.no_grad():
                    a_idx, _, _ = agent.choose_action(
                        state, self.env,
                        for_dup=apply_dup_mask,
                        original_job_idx=original_job_idx,
                        greedy=True,
                    )

            if a_idx is None:
                break

            s, a = divmod(a_idx, self.env.A)

            next_obs, rew, term, trunc, _ = self.env.step((s, a))
            r = float(np.sum(rew))
            done = bool(term or trunc)

            if training:
                agent.store(state, a_idx, logp, val, r, float(done))

            total_reward += r
            state = flatten_obs(next_obs) if not done else state
            self.last_obs = next_obs

            if for_time:
                break

        if not training:
            agent.actor.train()
            agent.critic.train()

        if training and self.learn_after_each_phase:
            agent.learn()
            agent.save()

        return total_reward

    def _schedule_with_primary_agent(self, dup_info=None):
        return self._schedule_with_agent(self.primary_agent, dup_info=dup_info, training=False)

    def _schedule_with_secondary_agent(self, dup_info=None):
        return self._schedule_with_agent(self.secondary_agent, dup_info=dup_info, training=False)

    def reset_common(self):
        self.env.current_job_idx = 0
        self.primary_J = self.env.J
        self.dup_count = 0
        return self._augment_obs(self.env._get_obs()), {"total_reward": self.reward1 + self.reward2}

    def _augment_obs(self, obs):
        total_slots = self.env.S * self.env.A
        occupied = int(np.sum(self.env.assignment[:self.primary_J].sum(axis=0) > 0))
        frac_occupied = occupied / total_slots
        can_dup = float(self._can_duplicate(self.env.current_job_idx)) if self.env.current_job_idx < self.primary_J else 0.0
        obs['meta'] = np.array([
            self.dup_count / max(self.primary_J, 1),
            frac_occupied,
            can_dup,
        ], dtype=np.float32)
        return obs

    def step(self, action):
        if self.env.current_job_idx >= self.primary_J:
            return self._augment_obs(self.env._get_obs(self.primary_J)), 0.0, True, False, {
                "total_reward": self.reward1 + self.reward2
            }

        if action == 1 and self._can_duplicate(self.env.current_job_idx):
            current_job_idx = self.env.current_job_idx
            current_job = self.env.problem.jobs[current_job_idx]

            new_job_idx = self.env.add_job(current_job, original_job_idx=current_job_idx)

            original_job_idx = current_job_idx
            self.env.current_job_idx = new_job_idx
            self.last_obs = self.env._get_obs()

            dup_info = {"for_dup": True, "original_job_idx": original_job_idx, "dup_idx": new_job_idx}
            reward_from_dup = self._schedule_with_secondary_agent(dup_info=dup_info)

            self.env.current_job_idx = original_job_idx + 1
            done_outer = (self.env.current_job_idx >= self.primary_J) or not self.env._has_available_resources()

            self.reward2 += reward_from_dup
            self.dup_count += 1

            return self._augment_obs(self.env._get_obs(self.primary_J)), reward_from_dup, done_outer, False, {
                "total_reward": self.reward1 + self.reward2
            }
        else:
            self.env.current_job_idx += 1
            done = self.env.current_job_idx >= self.primary_J
            return self._augment_obs(self.env._get_obs(self.primary_J)), 0.0, done, False, {
                "total_reward": self.reward1 + self.reward2
            }

    def render(self):
        self.env.render()
