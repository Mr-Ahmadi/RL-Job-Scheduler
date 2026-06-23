import os
import numpy as np
import torch
from environment.ppo.core import (
    device, PPOMemory, ActorNet, CriticNet, flatten_obs, PPOAgent as BasePPOAgent
)


def flatten_obs_subset(obs):
    base = flatten_obs(obs)
    meta = obs.get('meta', np.zeros(3, dtype=np.float32))
    return np.concatenate([base, meta])


def get_valid_action_indices(env, for_dup=False, original_job_idx=None):
    valid = []
    j = env.current_job_idx

    if not for_dup and (j >= env.J or env.assignment[j].sum() > 0):
        return valid

    if not for_dup:
        for s in range(env.S):
            for a in range(env.A):
                if env.assignment[:, s, a].sum() < 2:
                    valid.append(s * env.A + a)
        return valid

    if original_job_idx is None or original_job_idx >= env.J:
        return valid

    orig_positions = [(s, a) for s in range(env.S) for a in range(env.A) if env.assignment[original_job_idx, s, a]]
    if not orig_positions:
        return valid
    orig_slots_set = set(orig_positions)
    for s in range(env.S):
        for a in range(env.A):
            if (s, a) in orig_slots_set:
                continue
            if env.assignment[:, s, a].sum() < 2:
                valid.append(s * env.A + a)
    return valid


class PPOAgent(BasePPOAgent):
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, lamda,
                 clip, epochs, batch_size, checkpoint,
                 ent_start=0.1, ent_end=0.01, ent_decay=0.99):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            lamda=lamda,
            clip=clip,
            epochs=epochs,
            batch_size=batch_size,
            checkpoint_dir=checkpoint,
            ent_start=ent_start,
            ent_end=ent_end,
            ent_decay=ent_decay
        )

    def choose_action(self, state, env, for_dup=False, original_job_idx=None, greedy=False):
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        dist, probs = self.actor(state_t)
        val = float(self.critic(state_t).squeeze().item())

        valid_idxs = get_valid_action_indices(env, for_dup=for_dup, original_job_idx=original_job_idx)
        if not valid_idxs:
            return None, None, val

        mask = torch.zeros_like(probs)
        mask[0, valid_idxs] = 1.0
        masked = probs * mask
        sum_masked = masked.sum(dim=-1, keepdim=True)
        if (sum_masked == 0).any():
            masked = torch.zeros_like(probs)
            masked[0, valid_idxs] = 1.0 / float(len(valid_idxs))
        else:
            masked = masked / sum_masked

        if greedy:
            action = torch.argmax(masked, dim=-1)
            masked_dist = torch.distributions.Categorical(masked)
            logp = masked_dist.log_prob(action).item()
        else:
            masked_dist = torch.distributions.Categorical(masked)
            action = masked_dist.sample()
            logp = masked_dist.log_prob(action).item()
        return int(action.item()), float(logp), val
