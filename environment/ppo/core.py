import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if hasattr(torch, "mps") and torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
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
        batches = [indices[i:i + self.batch_size] for i in batch_starts]
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


class ActorNet(nn.Module):
    def __init__(self, input_dim, output_dim, lr, checkpoint_file,
                 hidden_size=1024, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.drop3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(hidden_size, output_dim)
        self.checkpoint_file = checkpoint_file
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.drop2(x)
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.drop3(x)
        logits = self.fc4(x)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist, probs

    def save(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=device))


class CriticNet(nn.Module):
    def __init__(self, input_dim, lr, checkpoint_file,
                 hidden_size=1024, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.drop3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.checkpoint_file = checkpoint_file
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.drop2(x)
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.drop3(x)
        value = self.fc4(x)
        return value

    def save(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=device))


_flatten_buf = None

def flatten_obs(obs):
    global _flatten_buf
    gpu = obs['gpu_state']
    cj = obs['current_job']
    fs = obs['future_job_stats']
    jl = obs['jobs_left']
    occ = obs.get('occupancy')
    sl = obs.get('server_load')
    sm = obs.get('server_model_unique')
    occ_size = occ.size if occ is not None else 0
    sl_size = sl.size if sl is not None else 0
    sm_size = sm.size if sm is not None else 0
    total = gpu.size + occ_size + sl_size + sm_size + len(cj['one_hot']) + 1 + fs.size + 1
    if _flatten_buf is None or _flatten_buf.size != total:
        _flatten_buf = np.empty(total, dtype=np.float32)
    offset = 0
    n = gpu.size
    _flatten_buf[offset:offset+n] = gpu.ravel()
    offset += n
    if occ is not None:
        n = occ.size
        _flatten_buf[offset:offset+n] = occ.ravel()
        offset += n
    if sl is not None:
        n = sl.size
        _flatten_buf[offset:offset+n] = sl.ravel()
        offset += n
    if sm is not None:
        n = sm.size
        _flatten_buf[offset:offset+n] = sm.ravel()
        offset += n
    n = len(cj['one_hot'])
    _flatten_buf[offset:offset+n] = cj['one_hot']
    offset += n
    _flatten_buf[offset] = cj['batch_size']
    offset += 1
    n = fs.size
    _flatten_buf[offset:offset+n] = fs.ravel()
    offset += n
    _flatten_buf[offset] = jl
    return _flatten_buf.copy()


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, lamda,
                 clip, epochs, batch_size, checkpoint_dir,
                 ent_start=0.1, ent_end=0.01, ent_decay=0.99,
                 max_grad_norm=0.5, lr_actor_decay=0.9999, lr_critic_decay=0.9999,
                 clip_end=None):
        self.gamma = gamma
        self.lamda = lamda
        self.clip = clip
        self.clip_start = clip
        self.clip_end = clip_end
        self.epochs = epochs
        self.ent_coef = ent_start
        self.ent_end = ent_end
        self.ent_decay = ent_decay
        self.max_grad_norm = max_grad_norm
        self.actor = ActorNet(state_dim, action_dim, lr_actor, f'{checkpoint_dir}/actor.pth')
        self.critic = CriticNet(state_dim, lr_critic, f'{checkpoint_dir}/critic.pth')
        self.memory = PPOMemory(batch_size)
        self._lr_actor_decay = lr_actor_decay
        self._lr_critic_decay = lr_critic_decay

    def learn(self):
        states, actions, old_logp, vals, rewards, dones, batches = self.memory.generate_batches()
        if len(states) == 0:
            return
        advantages = np.zeros_like(rewards, dtype=float)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = 0.0 if t == len(rewards) - 1 else vals[t + 1]
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_val * mask - vals[t]
            gae = delta + self.gamma * self.lamda * mask * gae
            advantages[t] = gae
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
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
                critic_val = self.critic(st).squeeze()
                critic_loss = F.mse_loss(critic_val.squeeze(), ret.squeeze())
                loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear()
        self.ent_coef = max(self.ent_end, self.ent_coef * self.ent_decay)
        if self.clip_end is not None:
            self.clip = max(self.clip_end, self.clip * 0.9995)
        for pg in self.actor.optimizer.param_groups:
            pg['lr'] *= self._lr_actor_decay
        for pg in self.critic.optimizer.param_groups:
            pg['lr'] *= self._lr_critic_decay

    def save(self):
        self.actor.save()
        self.critic.save()

    def load(self):
        self.actor.load()
        self.critic.load()

    def store(self, *args):
        self.memory.store(*args)

    def choose_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        dist, _ = self.actor(state_t)
        action = dist.sample()
        logp = dist.log_prob(action).item()
        val = self.critic(state_t).squeeze().item()
        return action.item(), logp, val

    def eval_mode(self):
        class _EvalCtx:
            def __init__(self, agent):
                self.agent = agent
            def __enter__(self):
                self.agent.actor.eval()
                self.agent.critic.eval()
                return self.agent
            def __exit__(self, *args):
                self.agent.actor.train()
                self.agent.critic.train()
        return _EvalCtx(self)


def plot_learning_curve(scores, file_path, title="PPO Evaluation Curve"):
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(scores, color='#1f77b4', linewidth=1.5)
    plt.xlabel("Evaluation Iteration", fontsize=12)
    plt.ylabel("Avg Reward", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(file_path, dpi=150, bbox_inches='tight')
    plt.close()
