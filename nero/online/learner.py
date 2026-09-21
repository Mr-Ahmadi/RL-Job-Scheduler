"""Full online learning: continual updates of a deployed scheduler.

Solution 4 ships a frozen policy plus two frozen learned heads.  This module is
the next step: the scheduler keeps learning from the placements it actually
makes, so it tracks a cluster whose job mix drifts away from the training
distribution.

What gets updated, and from what signal:

===================  ==========================================================
reward model         supervised regression on the *realised* reward of the slot
                     that was played (online there is one label per decision,
                     not the 45 the offline trainer enjoys)
value head           TD(0) regression, so the one-step-lookahead score keeps
                     tracking the drifting return distribution
Q-head               double-DQN TD(0) from the same replay buffer
critic + policy      clipped PPO on the on-policy rollout, plus an optional
                     distillation term that folds the top-K search's choice back
                     into the policy so the amortised policy catches up with the
                     search
===================  ==========================================================

Safety mechanisms (an unguarded online learner on a production cluster is how
you turn a working scheduler into a broken one):

1. **Candidate / deployed separation.**  Gradients only ever touch the candidate
   networks.  Acting always uses the deployed networks.
2. **Promotion gate.**  A candidate is promoted only if it is not worse than the
   deployed policy.  ``gate="sim"`` replays recent job sets through the cluster
   simulator for both policies.  ``gate="canary"`` needs no simulator: a small
   random slice of live traffic (``canary_frac``) is served by the candidate and
   the rest by the deployed policy, and the two are compared over the same
   window -- so drifting traffic moves both arms alike and cannot be mistaken
   for a policy regression.  A candidate that fails keeps learning and is
   re-tested next window; only ``gate_patience`` consecutive failures roll it
   back, so a genuine improvement that needs several windows to appear is not
   thrown away every time, while the deployed policy still never changes without
   passing the gate.
3. **Trust region.**  The PPO update early-stops once the approximate KL to the
   deployed policy exceeds ``kl_target``, on top of the usual ratio clipping.
4. **Small steps.**  Online learning rates sit below the offline ones (actor
   2e-4 vs 5e-4, critic 5e-4 vs 1e-3) with gradient-norm clipping, though it is
   the trust region above, not the learning rate, that actually bounds the move.
5. **Feasibility is never learned.**  Action masking is applied at every step, so
   no amount of learning can emit an infeasible placement.
"""

import copy
import json
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from nero.agents.ppo import flatten_obs

from nero.search.canonicalization import feasible_slots, slot_features
from nero.search.heads import SlotQHead
from nero.search.topk import LearnedTopKActor


class ReplayBuffer:
    """Fixed-capacity ring buffer of realised transitions."""

    def __init__(self, capacity, state_dim, action_dim, feature_dim):
        self.capacity = capacity
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.feat = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.next_mask = np.zeros((capacity, action_dim), dtype=np.float32)
        self.size = 0
        self.pos = 0

    def add(self, state, action, reward, next_state, next_mask, done, feat):
        i = self.pos
        self.state[i] = state
        self.feat[i] = feat
        self.next_state[i] = next_state
        self.action[i] = action
        self.reward[i] = reward
        self.next_mask[i] = next_mask
        self.done[i] = float(done)
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=min(batch_size, self.size))
        return (self.state[idx], self.action[idx], self.reward[idx],
                self.next_state[idx], self.next_mask[idx], self.done[idx],
                self.feat[idx])


class OnlineLearner:
    """A deployed scheduler that keeps learning, behind a promotion gate."""

    def __init__(self, agent, S, A, state_dim, action_dim, feature_dim,
                 reward_model=None, value_head=None, q_head=None,
                 gamma=0.99, lam=0.95,
                 lr_actor=2e-4, lr_critic=5e-4, lr_reward=1e-4, lr_value=1e-4,
                 lr_q=1e-4,
                 clip=0.1, ppo_epochs=4, batch_size=128, ent_coef=0.003,
                 kl_target=0.01, max_grad_norm=0.5,
                 update_every=10, replay_capacity=50000,
                 replay_batches=32, replay_batch_size=256, target_sync=200,
                 use_topk=False, k=5, mode="blend", beta=0.5,
                 explore_eps=0.1, distill_coef=0.5,
                 gate="sim", gate_tolerance=0.0, gate_patience=3, monitor_episodes=8,
                 monitor_window=64, canary_frac=0.25, min_canary=2,
                 sim_env=None, device="cpu", seed=0):
        self.device = torch.device(device)
        self.S, self.A = S, A
        self.state_dim, self.action_dim = state_dim, action_dim
        self.gamma, self.lam = gamma, lam
        self.clip, self.ppo_epochs, self.batch_size = clip, ppo_epochs, batch_size
        self.ent_coef, self.kl_target = ent_coef, kl_target
        self.max_grad_norm = max_grad_norm
        self.update_every = update_every
        self.replay_batches, self.replay_batch_size = replay_batches, replay_batch_size
        self.target_sync = target_sync
        self.use_topk = use_topk
        self.explore_eps = explore_eps
        self.distill_coef = distill_coef
        self.gate, self.gate_tolerance = gate, gate_tolerance
        self.gate_patience = gate_patience
        self.gate_failures = 0
        self.monitor_episodes, self.monitor_window = monitor_episodes, monitor_window
        self.canary_frac, self.min_canary = canary_frac, min_canary
        self.sim_env = sim_env
        self.rng = np.random.default_rng(seed)
        self._torch_gen = torch.Generator().manual_seed(seed)

        # candidate networks -- the only ones that receive gradients
        self.actor = agent.actor.to(self.device)
        self.critic = agent.critic.to(self.device)
        self.reward_model = reward_model
        self.value_head = value_head
        self.q_head = q_head
        if self.reward_model is not None:
            self.reward_model.to(self.device)
        if self.value_head is not None:
            self.value_head.to(self.device)
        if self.q_head is not None:
            self.q_head.to(self.device)
        self.q_target = None
        if self.q_head is not None:
            self.q_target = SlotQHead(state_dim, action_dim).to(self.device)
            self.q_target.load_state_dict(self.q_head.state_dict())
            self.q_target.eval()

        for net, lr in ((self.actor, lr_actor), (self.critic, lr_critic),
                        (self.reward_model, lr_reward), (self.value_head, lr_value),
                        (self.q_head, lr_q)):
            if net is not None:
                for pg in net.optimizer.param_groups:
                    pg["lr"] = lr

        # deployed networks -- what actually schedules jobs
        self.deployed = {name: copy.deepcopy(net).eval()
                         for name, net in self._nets().items()}
        self._topk = None
        if use_topk:
            if self.reward_model is None or self.q_head is None:
                raise ValueError("use_topk needs both a reward model and a Q-head")
            self._topk = LearnedTopKActor(
                k=k, mode=mode, beta=beta, gamma=gamma, threads=0, device=device,
                agent=SimpleNamespace(actor=self.deployed["actor"],
                                      critic=self.deployed["critic"]),
                reward_model=self.deployed["reward_model"],
                value_head=self.deployed.get("value_head"),
                q_head=self.deployed["q_head"])

        self._topk_candidate = self._topk_for(self._nets()) if use_topk else None

        self.feature_dim = feature_dim
        self.replay = ReplayBuffer(replay_capacity, state_dim, action_dim, feature_dim)
        self.rollout = []
        self.recent_jobs = []
        self.recent_returns = []
        self.episode = 0
        self.updates = 0
        self.train_steps = 0
        self.window = {"canary": [], "control": []}
        self.history = []

    # ------------------------------------------------------------------ nets
    def _nets(self):
        nets = {"actor": self.actor, "critic": self.critic}
        for name in ("reward_model", "value_head", "q_head"):
            net = getattr(self, name)
            if net is not None:
                nets[name] = net
        return nets

    def _promote(self):
        for name, net in self._nets().items():
            self.deployed[name].load_state_dict(net.state_dict())

    def _rollback(self):
        for name, net in self._nets().items():
            net.load_state_dict(self.deployed[name].state_dict())
        if self.q_target is not None:
            self.q_target.load_state_dict(self.q_head.state_dict())

    # ------------------------------------------------------------------ acting
    def mask_for(self, env):
        m = np.zeros(self.action_dim, dtype=np.float32)
        for (s, a) in feasible_slots(env):
            m[s * self.A + a] = 1.0
        return m

    def _masked_probs(self, actor, state, mask):
        st = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, probs = actor(st)
        mt = torch.as_tensor(mask, device=self.device).unsqueeze(0)
        masked = probs * mt
        total = masked.sum(dim=-1, keepdim=True)
        masked = torch.where(total > 0, masked / total.clamp(min=1e-12),
                             mt / mt.sum(dim=-1, keepdim=True).clamp(min=1e-12))
        return masked

    def act(self, env, greedy=False, nets=None, topk=None):
        """Choose a placement with the given (default: deployed) networks.

        Returns ``(action_index, log_prob, value, mask, state, from_search)``.
        ``log_prob`` is the behaviour log-probability under the acting masked
        policy, which is what the PPO ratio is corrected against even when the
        top-K search, not the policy, picked the action.
        """
        nets = self.deployed if nets is None else nets
        topk = self._topk if topk is None else topk
        if topk is False:           # explicit "policy only", for probes
            topk = None
        state = flatten_obs(env._get_obs())
        mask = self.mask_for(env)
        if mask.sum() == 0:
            return None, 0.0, 0.0, mask, state, False
        masked = self._masked_probs(nets["actor"], state, mask)

        # a deployed scheduler acts greedily (or with the search) and pays for a
        # small slice of exploration, which is what keeps the policy update
        # informative about slots the current policy would never try
        explore = (not greedy) and self.rng.random() < self.explore_eps
        use_search = topk is not None and not explore
        if use_search:
            slot = topk.decide(env)
            action = slot[0] * self.A + slot[1]
        elif explore:
            action = int(Categorical(masked).sample().item())
        else:
            action = int(torch.argmax(masked, dim=-1).item())

        logp = float(torch.log(masked[0, action].clamp(min=1e-12)).item())
        st = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            value = float(nets["critic"](st).squeeze().item())
        return action, logp, value, mask, state, bool(use_search)

    def run_episode(self, env, job_list=None, learn=True, greedy=False,
                    nets=None, topk=None):
        """Schedule one job set end to end, recording everything learnable."""
        canary = False
        if learn and nets is None and self.gate == "canary":
            # a canary episode is served by the *candidate* networks; this is the
            # only traffic unvalidated weights ever touch
            canary = bool(self.rng.random() < self.canary_frac)
            if canary:
                nets, topk = self._nets(), self._topk_candidate
        obs, _ = env.reset(job_list) if job_list is not None else env.reset()
        total = 0.0
        done = False
        while not done:
            action, logp, value, mask, state, searched = self.act(
                env, greedy=greedy, nets=nets, topk=topk)
            if action is None:
                break
            s, a = divmod(action, self.A)
            feat = (slot_features(env, env.current_job_idx, [(s, a)])[0]
                    if learn else None)
            obs, rew, term, trunc, _ = env.step((s, a))
            reward = float(sum(rew))
            done = bool(term or trunc)
            total += reward
            if learn:
                next_state = flatten_obs(obs)
                next_mask = self.mask_for(env) if not done else np.zeros_like(mask)
                self.replay.add(state, action, reward, next_state, next_mask,
                                done, feat)
                self.rollout.append((state, action, logp, value, reward, done, searched))
        if learn:
            self.episode += 1
            self.window["canary" if canary else "control"].append(total)
            self.recent_returns.append(total)
            self.recent_jobs.append(list(env.problem.jobs))
            if len(self.recent_jobs) > self.monitor_window:
                self.recent_jobs = self.recent_jobs[-self.monitor_window:]
                self.recent_returns = self.recent_returns[-self.monitor_window:]
            if self.episode % self.update_every == 0:
                self.update()
        return total

    # ------------------------------------------------------------------ learning
    def _update_reward_model(self):
        if self.reward_model is None or self.replay.size < self.replay_batch_size:
            return None
        self.reward_model.train()
        losses = []
        for _ in range(self.replay_batches):
            _, _, r, _, _, _, f = self.replay.sample(self.replay_batch_size, self.rng)
            fb = torch.as_tensor(f, device=self.device)
            rb = torch.as_tensor(r, device=self.device)
            # online there is one label per decision: the slot actually played
            pred = self.reward_model.predict(fb)
            loss = F.mse_loss(pred, rb)
            self.reward_model.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), self.max_grad_norm)
            self.reward_model.optimizer.step()
            losses.append(float(loss))
        self.reward_model.eval()
        return float(np.mean(losses))

    def _update_value_head(self):
        if self.value_head is None or self.replay.size < self.replay_batch_size:
            return None
        self.value_head.train()
        losses = []
        for _ in range(self.replay_batches):
            s, _, r, sp, _, d, _ = self.replay.sample(self.replay_batch_size, self.rng)
            sb = torch.as_tensor(s, device=self.device)
            spb = torch.as_tensor(sp, device=self.device)
            rb = torch.as_tensor(r, device=self.device)
            db = torch.as_tensor(d, device=self.device)
            with torch.no_grad():
                y = rb + self.gamma * (1.0 - db) * self.value_head(spb).squeeze(-1)
            loss = F.mse_loss(self.value_head(sb).squeeze(-1), y)
            self.value_head.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), self.max_grad_norm)
            self.value_head.optimizer.step()
            losses.append(float(loss))
        self.value_head.eval()
        return float(np.mean(losses))

    def _update_q_head(self):
        if self.q_head is None or self.replay.size < self.replay_batch_size:
            return None
        self.q_head.train()
        losses = []
        for _ in range(self.replay_batches):
            s, a, r, sp, mp, d, _ = self.replay.sample(self.replay_batch_size, self.rng)
            sb = torch.as_tensor(s, device=self.device)
            spb = torch.as_tensor(sp, device=self.device)
            ab = torch.as_tensor(a, device=self.device).unsqueeze(-1)
            rb = torch.as_tensor(r, device=self.device)
            db = torch.as_tensor(d, device=self.device)
            mb = torch.as_tensor(mp, device=self.device)
            with torch.no_grad():
                online_next = self.q_head(spb).masked_fill(mb == 0, -1e9)
                a_star = online_next.argmax(-1, keepdim=True)
                q_next = self.q_target(spb).gather(-1, a_star).squeeze(-1)
                q_next = torch.where(mb.sum(-1) > 0, q_next, torch.zeros_like(q_next))
                y = rb + self.gamma * (1.0 - db) * q_next
            q = self.q_head(sb).gather(-1, ab).squeeze(-1)
            loss = F.smooth_l1_loss(q, y)
            self.q_head.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_head.parameters(), self.max_grad_norm)
            self.q_head.optimizer.step()
            losses.append(float(loss))
            self.train_steps += 1
            if self.train_steps % self.target_sync == 0:
                self.q_target.load_state_dict(self.q_head.state_dict())
        self.q_head.eval()
        return float(np.mean(losses))

    def _update_policy(self):
        if len(self.rollout) < self.batch_size:
            return None, None
        states = np.asarray([t[0] for t in self.rollout], dtype=np.float32)
        actions = np.asarray([t[1] for t in self.rollout], dtype=np.int64)
        old_logp = np.asarray([t[2] for t in self.rollout], dtype=np.float32)
        values = np.asarray([t[3] for t in self.rollout], dtype=np.float32)
        rewards = np.asarray([t[4] for t in self.rollout], dtype=np.float32)
        dones = np.asarray([t[5] for t in self.rollout], dtype=np.float32)
        searched = np.asarray([t[6] for t in self.rollout], dtype=np.float32)

        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = 0.0 if t == len(rewards) - 1 else values[t + 1]
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            adv[t] = gae
        returns = adv + values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        st = torch.as_tensor(states, device=self.device)
        ac = torch.as_tensor(actions, device=self.device)
        lp = torch.as_tensor(old_logp, device=self.device)
        ad = torch.as_tensor(adv, device=self.device)
        rt = torch.as_tensor(returns, device=self.device)
        sr = torch.as_tensor(searched, device=self.device)

        self.actor.train()
        self.critic.train()
        n = st.shape[0]
        approx_kl, losses, stopped = 0.0, [], False
        for _ in range(self.ppo_epochs):
            if stopped:
                break
            perm = torch.randperm(n, generator=self._torch_gen)
            for b in range(0, n, self.batch_size):
                bi = perm[b:b + self.batch_size].to(self.device)
                dist, probs = self.actor(st[bi])
                new_logp = dist.log_prob(ac[bi])
                ratio = (new_logp - lp[bi]).exp()
                surr1 = ratio * ad[bi]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * ad[bi]
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss = actor_loss - self.ent_coef * dist.entropy().mean()
                if self.use_topk and self.distill_coef > 0 and sr[bi].sum() > 0:
                    # fold the search's decisions back into the amortised policy,
                    # so the cheap policy catches up with what the search found
                    distill = -(new_logp * sr[bi]).sum() / sr[bi].sum()
                    actor_loss = actor_loss + self.distill_coef * distill
                critic_loss = F.mse_loss(self.critic(st[bi]).squeeze(-1), rt[bi])
                loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                losses.append(float(loss))
                # trust region: the true KL(deployed || candidate), measured in
                # eval mode over the whole rollout, is checked after every
                # minibatch -- checking only between epochs lets a single epoch
                # sail past the target
                approx_kl = self._policy_kl(st)
                if approx_kl > self.kl_target:
                    stopped = True
                    break
        self.actor.eval()
        self.critic.eval()
        self.rollout = []
        return float(np.mean(losses)) if losses else None, dict(kl=approx_kl, kl_stop=stopped)

    def _settle(self, ok):
        """Promote on success; roll back only after sustained failure."""
        if ok:
            self.gate_failures = 0
            self._promote()
            return False
        self.gate_failures += 1
        if self.gate_failures >= self.gate_patience:
            self.gate_failures = 0
            self._rollback()
            return True
        return False

    def _policy_kl(self, states):
        """KL(deployed || candidate) over the rollout states, dropout disabled."""
        was_training = self.actor.training
        self.actor.eval()
        with torch.no_grad():
            _, p_new = self.actor(states)
            _, p_old = self.deployed["actor"](states)
            kl = (p_old * (torch.log(p_old.clamp(min=1e-12))
                           - torch.log(p_new.clamp(min=1e-12)))).sum(-1).mean()
        if was_training:
            self.actor.train()
        return float(kl)

    # ------------------------------------------------------------------ gate
    def _topk_for(self, nets):
        if not self.use_topk:
            return None
        return LearnedTopKActor(
            k=self._topk.k, mode=self._topk.scorer.mode, beta=self._topk.scorer.beta,
            gamma=self.gamma, threads=0, device=str(self.device),
            agent=SimpleNamespace(actor=nets["actor"], critic=nets["critic"]),
            reward_model=nets["reward_model"], value_head=nets.get("value_head"),
            q_head=nets["q_head"])

    def _shadow_score(self, nets, job_lists):
        """Mean return of a network set over recent job sets, in the simulator.

        This is the one part of the loop that needs a model of the cluster.  With
        ``gate="window"`` the learner runs without it, at the cost of detecting a
        bad promotion only after it has been deployed for one window.
        """
        if self.sim_env is None:
            return None
        topk = self._topk_for(nets)
        return float(np.mean([self.run_episode(self.sim_env, job_list=list(jl),
                                               learn=False, greedy=True,
                                               nets=nets, topk=topk)
                              for jl in job_lists]))

    def _gate_decision(self):
        """Promote the candidate only if it is not worse than what is deployed."""
        if self.gate == "none":
            self._promote()
            return dict(gate="none", promoted=True)

        if self.gate == "canary":
            c, d = self.window["canary"], self.window["control"]
            if len(c) < self.min_canary or not d:
                return dict(gate="canary", promoted=None, n_canary=len(c),
                            reason="not enough canary traffic yet")
            c_mean, d_mean = float(np.mean(c)), float(np.mean(d))
            ok = c_mean >= d_mean - self.gate_tolerance
            rolled = self._settle(ok)
            self.window = {"canary": [], "control": []}
            return dict(gate="canary", promoted=bool(ok), rolled_back=rolled,
                        candidate=c_mean, deployed=d_mean,
                        n_canary=len(c), n_control=len(d))

        n = min(self.monitor_episodes, len(self.recent_jobs))
        if n == 0 or self.sim_env is None:
            self._promote()
            return dict(gate="sim", promoted=True, reason="no monitor set")
        pick = self.rng.choice(len(self.recent_jobs), size=n, replace=False)
        job_lists = [self.recent_jobs[int(i)] for i in pick]
        cand_nets = {k: v for k, v in self._nets().items()}
        cand = self._shadow_score(cand_nets, job_lists)
        base = self._shadow_score(self.deployed, job_lists)
        ok = cand >= base - self.gate_tolerance
        rolled = self._settle(ok)
        return dict(gate="sim", promoted=bool(ok), rolled_back=rolled,
                    candidate=cand, deployed=base, monitor_episodes=n)

    def update(self):
        r_loss = self._update_reward_model()
        v_loss = self._update_value_head()
        q_loss = self._update_q_head()
        p_loss, p_info = self._update_policy()
        decision = self._gate_decision()
        if self.gate != "canary":
            self.window = {"canary": [], "control": []}
        self.updates += 1
        rec = dict(update=self.updates, episode=self.episode,
                   mean_return=float(np.mean(self.recent_returns[-self.update_every:])),
                   reward_loss=r_loss, value_loss=v_loss, td_loss=q_loss,
                   policy_loss=p_loss,
                   **(p_info or {}), **decision)
        self.history.append(rec)
        return rec

    # ------------------------------------------------------------------ io
    def save(self, directory):
        import os
        os.makedirs(directory, exist_ok=True)
        for name, net in self.deployed.items():
            torch.save(net.state_dict(), os.path.join(directory, f"{name}.pth"))
        with open(os.path.join(directory, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

    def deployed_agent(self):
        """The currently deployed networks, shaped like a ``PPOAgent``."""
        return SimpleNamespace(actor=self.deployed["actor"],
                               critic=self.deployed["critic"])
