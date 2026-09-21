"""Learned replacements for the oracle used by the top-K search.

Three heads, none of which ever reads the throughput table at decision time:

``SlotRewardModel``  r_hat(features)
    The immediate reward of a placement -- the ``new_tp + delta`` term the oracle
    top-K looks up.  Its input is not the 632-wide observation but the
    18-dimensional *sufficient statistic* of the reward that
    ``nero.search.canonicalization.slot_features`` extracts (current job, accelerator,
    co-located job, applicable discounts).  One shared function is fitted across
    all slots instead of 45 independent outputs, which is what makes it accurate
    enough to replace exact table lookups.  The table supplies the labels
    offline, the way logged cluster measurements would.

``SlotValueHead``  V(state)
    The discounted return from a state under the deployed policy, refitted on
    the collected rollouts with Monte-Carlo targets.  Used to price the *future*
    cost of a placement via ``r_hat + gamma * V(s')``, where ``s'`` comes from the
    exact delta builder in ``nero/search/fast_obs.py``.

``SlotQHead``  q(state, .)
    Double-DQN action values over all 45 slots, trained with a Polyak-averaged
    target network.  This is the long-horizon term the myopic oracle cannot
    provide at all.

``LearnedScorer`` combines them.  Modes:

  ``reward``        r_hat                              -- learned myopic oracle
  ``reward_value``  r_hat + gamma * V(s')              -- one-step lookahead
  ``q``             q(state, a)                        -- no next state needed
  ``blend``         (1 - beta) * reward_value + beta * q
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nero.agents.ppo import flatten_obs

from nero.search.canonicalization import slot_features
from nero.search.fast_obs import NextObsBuilder
from nero.paths import HEADS

DEFAULT_DIR = str(HEADS)
SCORER_MODES = ("reward", "reward_value", "q", "blend")


class MLPHead(nn.Module):
    """Small LayerNorm MLP with checkpointing, shared by all three heads."""

    def __init__(self, input_dim, output_dim, checkpoint_file,
                 hidden_size=256, dropout=0.0, lr=3e-4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size, output_dim)
        self.checkpoint_file = checkpoint_file
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.drop2(x)
        return self.fc3(x)

    def save(self):
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load(self, map_location="cpu"):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=map_location))


class SlotRewardModel(MLPHead):
    """r_hat over the canonical slot features (shared across all slots)."""

    def __init__(self, feature_dim, checkpoint_dir=DEFAULT_DIR, hidden_size=256, **kw):
        super().__init__(feature_dim, 1, os.path.join(checkpoint_dir, "reward_model.pth"),
                         hidden_size=hidden_size, **kw)

    def predict(self, features):
        """features: (n, feature_dim) tensor -> (n,) rewards."""
        return self.forward(features).squeeze(-1)


class SlotValueHead(MLPHead):
    """V(state) refitted on rollout returns."""

    def __init__(self, state_dim, checkpoint_dir=DEFAULT_DIR, hidden_size=512, **kw):
        super().__init__(state_dim, 1, os.path.join(checkpoint_dir, "value_head.pth"),
                         hidden_size=hidden_size, **kw)


class SlotQHead(MLPHead):
    """q(state, .) over every slot."""

    def __init__(self, state_dim, action_dim, checkpoint_dir=DEFAULT_DIR,
                 hidden_size=512, **kw):
        super().__init__(state_dim, action_dim, os.path.join(checkpoint_dir, "q_head.pth"),
                         hidden_size=hidden_size, **kw)


class LearnedScorer:
    """Scores candidate slots without ever reading the throughput table."""

    def __init__(self, state_dim, action_dim, feature_dim, checkpoint_dir=DEFAULT_DIR,
                 critic=None, gamma=0.99, mode="blend", beta=0.5, device="cpu",
                 reward_model=None, value_head=None, q_head=None):
        if mode not in SCORER_MODES:
            raise ValueError(f"mode must be one of {SCORER_MODES}, got {mode!r}")
        self.mode = mode
        self.beta = beta
        self.gamma = gamma
        self.device = device
        self.action_dim = action_dim
        self.builder = NextObsBuilder()

        self.reward_model = reward_model
        self.value_head = value_head
        self.q_head = q_head
        if self._needs_reward and self.reward_model is None:
            self.reward_model = SlotRewardModel(feature_dim, checkpoint_dir)
            self.reward_model.load(map_location=device)
        if self._needs_q and self.q_head is None:
            self.q_head = SlotQHead(state_dim, action_dim, checkpoint_dir)
            self.q_head.load(map_location=device)
        if self._needs_value and self.value_head is None:
            head = SlotValueHead(state_dim, checkpoint_dir)
            if os.path.exists(head.checkpoint_file):
                head.load(map_location=device)
                self.value_head = head
            elif critic is not None:
                self.value_head = critic        # fall back to the PPO critic
            else:
                raise ValueError(f"mode {mode!r} needs a value head or a critic")
        for net in (self.reward_model, self.value_head, self.q_head):
            if net is not None:
                net.to(device).eval()

    @property
    def _needs_reward(self):
        return self.mode in ("reward", "reward_value", "blend")

    @property
    def _needs_q(self):
        return self.mode in ("q", "blend")

    @property
    def _needs_value(self):
        return self.mode in ("reward_value", "blend")

    def score(self, env, base_obs, state, candidates):
        """Score ``candidates`` (a list of ``(s, a)`` slots) for the current job."""
        j = env.current_job_idx
        with torch.no_grad():
            r_hat = q_hat = rv = None
            if self._needs_reward:
                feats = torch.from_numpy(slot_features(env, j, candidates)).to(self.device)
                r_hat = self.reward_model.predict(feats).cpu().numpy()
            if self._needs_q:
                st = torch.from_numpy(np.asarray(state, dtype=np.float32))
                st = st.unsqueeze(0).to(self.device)
                idx = [s * env.A + a for (s, a) in candidates]
                q_hat = self.q_head(st)[0, idx].cpu().numpy()
            if self._needs_value:
                nxt = np.empty((len(candidates), state.shape[0]), dtype=np.float32)
                cont = np.ones(len(candidates), dtype=np.float32)
                for i, (s, a) in enumerate(candidates):
                    obs_p, done_p = self.builder.next_obs(env, base_obs, s, a)
                    nxt[i] = flatten_obs(obs_p)
                    cont[i] = 0.0 if done_p else 1.0
                v = self.value_head(torch.from_numpy(nxt).to(self.device))
                v = v.squeeze(-1).cpu().numpy()
                rv = r_hat + self.gamma * cont * v

        if self.mode == "reward":
            return r_hat
        if self.mode == "reward_value":
            return rv
        if self.mode == "q":
            return q_hat
        return (1.0 - self.beta) * rv + self.beta * q_hat
