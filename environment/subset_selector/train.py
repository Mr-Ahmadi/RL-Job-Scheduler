import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from ._requirments import flatten_obs, PPOAgent
from ..job_scheduling.train import Train_JobSchedulingEnv


class Train_SubsetSelectorEnv(gym.Env):
    def __init__(self, learn_after_each_phase: bool = True):
        super().__init__()
        self.env = Train_JobSchedulingEnv()

        # initialize environment once to shape networks
        obs, _ = self.env.reset()
        state_dim = flatten_obs(obs).shape[0]
        action_dim = self.env.S * self.env.A  # internal scheduler's flattened actions

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

        self.learn_after_each_phase = learn_after_each_phase
        # outer action space: 2 choices -> duplicate (1) or skip (0)
        self.action_space = spaces.Discrete(2)
        # self.observation_space = self.env.observation_space

        # Load (or keep) pre-trained actor weights and SWITCH TO TRAIN MODE so it can keep learning
        self.primary_agent.actor.load_state_dict(torch.load("models/job_scheduling/ppo/pretrained/actor.pth", weights_only=False))
        self.primary_agent.critic.load_state_dict(torch.load("models/job_scheduling/ppo/pretrained/critic.pth", weights_only=False))
        
        # Load (or keep) pre-trained actor weights and SWITCH TO TRAIN MODE so it can keep learning
        self.secondary_agent.actor.load_state_dict(torch.load("models/job_scheduling/ppo/pretrained/actor.pth", weights_only=False))
        self.secondary_agent.critic.load_state_dict(torch.load("models/job_scheduling/ppo/pretrained/critic.pth", weights_only=False))
        

    # ----------------------
    # Utility for duplication check
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

    # ----------------------
    # Internal scheduling helper
    # ----------------------
    def _schedule_with_primary_agent(self, dup_info=None):
        done = False
        state = flatten_obs(self.last_obs)
        total_reward = 0.0

        dup_initial_idx = dup_info['dup_idx'] if dup_info is not None and 'dup_idx' in dup_info else None
        original_job_idx = dup_info.get('original_job_idx') if dup_info is not None else None
        for_time = dup_info.get('for_dup', False) if dup_info is not None else False

        while not done:
            apply_dup_mask = (for_time and (self.env.current_job_idx == dup_initial_idx))

            a_idx, logp, val = self.primary_agent.choose_action(
                state,
                self.env,
                for_dup=apply_dup_mask,
                original_job_idx=original_job_idx
            )

            if a_idx is None:
                # No valid action available. This can happen if masking removes all actions.
                # Safe behavior: break inner scheduling loop (stop scheduling this phase).
                # For duplicate scheduling you could optionally remove the added duplicate job here
                # if you have a remove_job() API. We just break and return accumulated reward.
                break

            s, a = divmod(a_idx, self.env.A)

            next_obs, rew, term, trunc, _ = self.env.step((s, a))
            r = float(np.sum(rew))
            done = bool(term or trunc)

            # store experience for internal agent
            self.primary_agent.store(state, a_idx, logp, val, r, float(done))

            total_reward += r
            state = flatten_obs(next_obs) if not done else state
            self.last_obs = next_obs

        if self.learn_after_each_phase:
            self.primary_agent.learn()
            self.primary_agent.save()

        return total_reward
    
    
        # ----------------------
    # Internal scheduling helper
    # ----------------------
    def _schedule_with_secondary_agent(self, dup_info=None):
        done = False
        state = flatten_obs(self.last_obs)
        total_reward = 0.0

        dup_initial_idx = dup_info['dup_idx'] if dup_info is not None and 'dup_idx' in dup_info else None
        original_job_idx = dup_info.get('original_job_idx') if dup_info is not None else None
        for_time = dup_info.get('for_dup', False) if dup_info is not None else False

        while not done:
            apply_dup_mask = (for_time and (self.env.current_job_idx == dup_initial_idx))

            a_idx, logp, val = self.secondary_agent.choose_action(
                state,
                self.env,
                for_dup=apply_dup_mask,
                original_job_idx=original_job_idx
            )

            if a_idx is None:
                # No valid action available. This can happen if masking removes all actions.
                # Safe behavior: break inner scheduling loop (stop scheduling this phase).
                # For duplicate scheduling you could optionally remove the added duplicate job here
                # if you have a remove_job() API. We just break and return accumulated reward.
                break

            s, a = divmod(a_idx, self.env.A)

            next_obs, rew, term, trunc, _ = self.env.step((s, a))
            r = float(np.sum(rew))
            done = bool(term or trunc)

            # store experience for internal agent
            self.secondary_agent.store(state, a_idx, logp, val, r, float(done))

            total_reward += r
            state = flatten_obs(next_obs) if not done else state
            self.last_obs = next_obs

        if self.learn_after_each_phase:
            self.secondary_agent.learn()
            self.secondary_agent.save()

        return total_reward

    # ----------------------
    # Outer loop API
    # ----------------------
    def reset(self, seed=None):
        super().reset(seed=seed)
        # perform first scheduling pass: schedule all jobs once
        self.last_obs, _ = self.env.reset()
        self.reward1 = 0.0
        self.reward2 = 0.0

        # schedule primary phase (all jobs once)
        self.reward1 += self._schedule_with_primary_agent()

        # mark primary J so outer loop knows where to stop when duplicating/skipping
        self.env.current_job_idx = 0
        self.primary_J = self.env.J

        return self.env._get_obs(), {"total_reward": self.reward1 + self.reward2}

    def step(self, action):
        """
        Outer action:
         - 1 -> duplicate current job (if possible, else fallback to skip)
         - 0 -> skip current job
        After action, the outer loop moves on to the next primary job.
        """
        if self.env.current_job_idx >= self.primary_J:
            # no jobs left in primary phase
            return self.env._get_obs(self.primary_J), 0.0, True, False, {
                "total_reward": self.reward1 + self.reward2
            }

        if action == 1 and self._can_duplicate(self.env.current_job_idx):
            # ✅ proceed with duplication
            current_job_idx = self.env.current_job_idx
            current_job = self.env.problem.jobs[current_job_idx]

            # in Train_SubsetSelectorEnv.step, in the duplication branch:
            new_job_idx = self.env.add_job(current_job, original_job_idx=current_job_idx)


            # Temporarily schedule starting from the duplicate
            original_job_idx = current_job_idx
            self.env.current_job_idx = new_job_idx
            self.last_obs = self.env._get_obs()

            # Enforce same-server constraint on duplicate
            dup_info = {"for_dup": True, "original_job_idx": original_job_idx, "dup_idx": new_job_idx}
            reward_from_dup = self._schedule_with_secondary_agent(dup_info=dup_info)

            # Resume outer loop → next primary job
            self.env.current_job_idx = original_job_idx + 1
            done_outer = (self.env.current_job_idx >= self.primary_J) or not self.env._has_available_resources()

            self.reward2 += reward_from_dup

            return self.env._get_obs(self.primary_J), reward_from_dup, done_outer, False, {
                "total_reward": self.reward1 + self.reward2
            }

        else:
            # ❌ fallback → skip duplication
            self.env.current_job_idx += 1
            done = self.env.current_job_idx >= self.primary_J
            return self.env._get_obs(self.primary_J), 0.0, done, False, {
                "total_reward": self.reward1 + self.reward2
            }

    def render(self):
        # delegate to inner env
        self.env.render()
