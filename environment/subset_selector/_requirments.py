# ===========================
# Requirements for PPO Agent & Utilities
# ===========================

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# ---------------------------
# Device & Reproducibility
SEED = 42
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if hasattr(torch, "mps") and torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)


# ===========================
# PPO Memory
# ===========================
class PPOMemory:
    def __init__(self, batch_size):
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.values, self.dones = [], [], []
        self.batch_size = batch_size

    def store(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def generate_batches(self):
        n = len(self.states)
        indices = np.arange(n)
        np.random.shuffle(indices)
        batch_starts = np.arange(0, n, self.batch_size)
        batches = [indices[i:i+self.batch_size] for i in batch_starts]
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.log_probs),
            np.array(self.values),
            np.array(self.rewards),
            np.array(self.dones),
            batches
        )

    def clear(self):
        self.__init__(self.batch_size)


# ===========================
# Actor & Critic
# ===========================
class ActorNet(nn.Module):
    def __init__(self, input_dim, output_dim, lr, checkpoint_file,
                 hidden_size=256, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size, output_dim)
        self.checkpoint_file = checkpoint_file
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        logits = self.fc3(x)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist, probs

    def save(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class CriticNet(nn.Module):
    def __init__(self, input_dim, lr, checkpoint_file,
                 hidden_size=256, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.checkpoint_file = checkpoint_file
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        return self.fc3(x)

    def save(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


# ===========================
# Utilities
# ===========================
def flatten_obs(obs):
    """Convert nested dict observation to flat vector"""
    gpu_state = obs['gpu_state'].flatten()
    current_job = np.concatenate([obs['current_job']['one_hot'],
                                  [obs['current_job']['batch_size']]])
    future_stats = obs['future_job_stats'].flatten()
    jobs_left = np.array([obs['jobs_left']], dtype=np.float32)
    return np.concatenate([gpu_state, current_job, future_stats, jobs_left])


def get_valid_action_indices(env, for_dup=False, original_job_idx=None):
    valid = []
    j = env.current_job_idx

    # Skip if current job already assigned (normal scheduling)
    if not for_dup and (j >= env.J or env.assignment[j].sum() > 0):
        return valid

    if not for_dup:
        # Normal job: any slot with colocated < 2
        for s in range(env.S):
            for a in range(env.A):
                if env.assignment[:, s, a].sum() < 2:
                    valid.append(s * env.A + a)
        return valid

    # Duplicate scheduling: allow any server/accelerator except exact same slot as original
    if original_job_idx is None or original_job_idx >= env.J:
        return valid

    # find (s,a) positions occupied by original; there could be multiple places
    orig_positions = [(s, a) for s in range(env.S) for a in range(env.A) if env.assignment[original_job_idx, s, a]]

    # If original is not assigned, no valid duplicate placement
    if not orig_positions:
        return valid

    # Build valid slots: any slot with colocated < 2 and not in the set of exact (s,a) where original sits
    orig_slots_set = set(orig_positions)
    for s in range(env.S):
        for a in range(env.A):
            if (s, a) in orig_slots_set:
                continue  # don't allow exact same slot
            if env.assignment[:, s, a].sum() < 2:
                valid.append(s * env.A + a)

    return valid


def select_action_eval(state, env, actor_model, for_dup=False, original_job_idx=None):
    """Select action greedily with masking for evaluation. Returns None if no valid action."""
    st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _, probs = actor_model(st)

    valid_idxs = get_valid_action_indices(env, for_dup=for_dup, original_job_idx=original_job_idx)
    if not valid_idxs:
        # No valid placement possible (do not return an arbitrary 0)
        return None

    mask = torch.zeros_like(probs)
    mask[0, valid_idxs] = 1.0
    masked = probs * mask
    sum_masked = masked.sum(dim=-1, keepdim=True)

    if (sum_masked == 0).any():
        # uniform over valid indexes
        uniform = torch.zeros_like(probs)
        uniform[0, valid_idxs] = 1.0 / float(len(valid_idxs))
        return int(torch.argmax(uniform, dim=-1).item())
    else:
        masked = masked / sum_masked
        return int(torch.argmax(masked, dim=-1).item())


# ===========================
# PPO Agent
# ===========================
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, lamda,
                 clip, epochs, batch_size,
                 checkpoint,
                 ent_start=0.1, ent_end=0.01, ent_decay=0.99):
        self.gamma, self.lamda, self.clip = gamma, lamda, clip
        self.epochs = epochs
        self.ent_coef = ent_start
        self.ent_end, self.ent_decay = ent_end, ent_decay

        self.actor = ActorNet(state_dim, action_dim, lr_actor,
                              f'{checkpoint}/actor.pth')
        self.critic = CriticNet(state_dim, lr_critic,
                                f'{checkpoint}/critic.pth')
        self.memory = PPOMemory(batch_size)

    def choose_action(self, state, env, for_dup=False, original_job_idx=None):
        """
        Returns (action_index | None, log_prob | None, value).
        If no valid action indices -> returns (None, None, value).
        """
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        dist, probs = self.actor(state_t)
        val = float(self.critic(state_t).squeeze().item())

        # build mask
        valid_idxs = get_valid_action_indices(env, for_dup=for_dup, original_job_idx=original_job_idx)
        if not valid_idxs:
            # No valid action — return sentinel None (caller must handle)
            return None, None, val

        mask = torch.zeros_like(probs)
        mask[0, valid_idxs] = 1.0
        masked = probs * mask
        sum_masked = masked.sum(dim=-1, keepdim=True)
        if (sum_masked == 0).any():
            # fallback uniform across valid indices
            masked = torch.zeros_like(probs)
            masked[0, valid_idxs] = 1.0 / float(len(valid_idxs))
        else:
            masked = masked / sum_masked

        masked_dist = Categorical(masked)
        action = masked_dist.sample()
        logp = masked_dist.log_prob(action).item()
        return int(action.item()), float(logp), val

    def store(self, *args):
        self.memory.store(*args)

    def learn(self):
        states, actions, old_logp, vals, rewards, dones, batches = self.memory.generate_batches()
        if len(states) == 0:
            return

        # GAE advantage estimation
        advantages = np.zeros_like(rewards, dtype=float)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = 0.0 if t == len(rewards) - 1 else vals[t + 1]
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_val * mask - vals[t]
            gae = delta + self.gamma * self.lamda * mask * gae
            advantages[t] = gae

        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        vals_t = torch.tensor(vals, dtype=torch.float32, device=device)
        old_logp_t = torch.tensor(old_logp, dtype=torch.float32, device=device)

        for _ in range(self.epochs):
            for batch in batches:
                st = torch.tensor(states[batch], dtype=torch.float32, device=device)
                ac = torch.tensor(actions[batch], device=device)
                new_dist, _ = self.actor(st)
                new_logp = new_dist.log_prob(ac)
                ratio = (new_logp - old_logp_t[batch]).exp()

                surr1 = ratio * advantages[batch]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages[batch]
                entropy = new_dist.entropy().mean()
                actor_loss = -torch.min(surr1, surr2).mean() - self.ent_coef * entropy

                ret = advantages[batch] + vals_t[batch]
                critic_val = self.critic(st).squeeze(-1)  # only squeeze the last dim
                # critic_loss = F.mse_loss(critic_val, ret)
                # critic_loss = F.mse_loss(self.critic(st).squeeze(), ret)
                critic_loss = F.mse_loss(critic_val.squeeze(), ret.squeeze())

                loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear()
        self.ent_coef = max(self.ent_end, self.ent_coef * self.ent_decay)

    def save(self):
        self.actor.save()
        self.critic.save()

    def load(self):
        self.actor.load()
        self.critic.load()


# ===========================
# Plotting Helper
# ===========================
def plot_learning_curve(scores, file='curves/ppo.png'):
    plt.plot(scores)
    plt.xlabel("Evaluation Iteration")
    plt.ylabel("Avg Reward")
    plt.title("PPO Evaluation Curve")
    plt.grid()
    plt.savefig(file)
    plt.show()
