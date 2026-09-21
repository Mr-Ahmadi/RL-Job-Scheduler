import numpy as np
import torch

from nero.agents.ppo import device, flatten_obs, PPOAgent as BasePPOAgent


def flatten_obs_subset(obs):
    base = flatten_obs(obs)
    meta = obs.get('meta', np.zeros(3, dtype=np.float32))
    return np.concatenate([base, meta])


def get_valid_action_indices(env, for_dup=False, original_job_idx=None):
    """Feasible action indices, optionally for placing a *duplicate*.

    Feasibility itself comes from ``Base_JobSchedulingEnv.feasible_slots``; the
    only extra rule here is that a duplicate may not land on a slot its original
    already occupies.
    """
    j = env.current_job_idx
    if not for_dup:
        return [s * env.A + a for (s, a) in env.feasible_slots(j)]

    if original_job_idx is None or original_job_idx >= env.J:
        return []
    orig_slots = {(s, a) for s in range(env.S) for a in range(env.A)
                  if env.assignment[original_job_idx, s, a]}
    if not orig_slots:
        return []
    return [s * env.A + a for (s, a) in env.feasible_slots(j)
            if (s, a) not in orig_slots]


class PPOAgent(BasePPOAgent):
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, lamda,
                 clip, epochs, batch_size, checkpoint,
                 ent_start=0.1, ent_end=0.01, ent_decay=0.99,
                 **kwargs):
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
            ent_decay=ent_decay,
            **kwargs
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



