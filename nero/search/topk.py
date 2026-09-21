"""Solution 4: learned Q-head + reward model + canonicalisation + top-K.

The committed ``ppo_topk_k3/k5`` policies rank candidate slots with the PPO
policy and then pick among them with the environment's *true* reward -- exact
throughputs including co-location interference.  That makes them an offline
upper bound, not a scheduler: a real cluster does not know the interference
term of a placement it has not made yet.

This actor keeps the structure and removes the oracle:

  1. **Propose.**  The frozen PPO policy scores all feasible slots.
  2. **Canonicalise.**  Slots that are provably reward-equivalent for this job
     (``nero.search.canonicalization``) collapse into one class, so the K candidates
     are K genuinely different placements rather than K spellings of one.
     ``canonical="dedupe"`` (the default) folds duplicates *inside* the top-K
     list.  The surviving member of each class is the one the policy ranked
     highest, and the discarded ones score identically, so the decision is
     provably unchanged while typically only ~2 of the 5 candidates are actually
     scored.  ``canonical="expand"`` instead spends the whole budget on
     *distinct* classes, reaching further down the policy's ranking; that covers
     more genuinely different placements but, with a myopic score, makes the
     scheduler greedier (see the ablation in the README).  ``canonical="none"``
     scores the raw top-K list.
  3. **Evaluate.**  ``nero.search.heads.LearnedScorer`` scores one
     representative per class with the learned reward model, the PPO critic on
     the delta-built next state, and/or the learned Q-head.
  4. **Commit.**  The best-scoring candidate is played.  Scores within
     ``tie_tol`` of the best count as tied and are resolved by the policy's own
     ranking -- reward-equivalent placements are not future-equivalent, and the
     policy's prior over server identity is a better tie-break than float32
     noise in the last digit of the score.

Every input is observable at decision time, so the whole path runs online.
"""

import numpy as np
import torch

from nero.envs.job_scheduling.eval import Eval_JobSchedulingEnv
from nero.agents.ppo import flatten_obs

from nero.agents.loading import load_inner_agent
from nero.search.canonicalization import (canonical_classes, feasible_slots, feature_dim,
                               slot_key, slot_occupancy)
from nero.search.heads import DEFAULT_DIR, LearnedScorer
from nero.paths import INNER, TEST_SETS


class LearnedTopKActor:
    """Oracle-free top-K placement search."""

    CANONICAL_MODES = ("dedupe", "expand", "none")

    def __init__(self, k=5, mode="reward", beta=0.5, canonical="dedupe",
                 ppo_dir=str(INNER), learned_dir=DEFAULT_DIR,
                 gamma=0.99, tie_tol=1e-6, threads=1, device="cpu",
                 set_dir=str(TEST_SETS), agent=None,
                 reward_model=None, value_head=None, q_head=None):
        if threads:
            torch.set_num_threads(threads)
        if agent is None:
            agent, state_dim, action_dim = load_inner_agent(ppo_dir, device=device)
        else:
            state_dim = agent.actor.fc1.in_features
            action_dim = agent.actor.fc4.out_features
        self.agent = agent
        self.actor = agent.actor
        self.critic = agent.critic
        self.k = k
        self.tie_tol = tie_tol
        if canonical is True:
            canonical = "expand"
        elif canonical is False:
            canonical = "none"
        if canonical not in self.CANONICAL_MODES:
            raise ValueError(f"canonical must be one of {self.CANONICAL_MODES}")
        self.canonical = canonical
        self.device = device
        probe = Eval_JobSchedulingEnv(set_dir)
        probe.reset()
        self.feature_dim = feature_dim(probe)
        self.scorer = LearnedScorer(state_dim, action_dim, self.feature_dim,
                                    learned_dir, critic=self.critic, gamma=gamma,
                                    mode=mode, beta=beta, device=device,
                                    reward_model=reward_model,
                                    value_head=value_head, q_head=q_head)

    def _candidates(self, env, probs, allowed=None):
        """The K candidate slots to score, best first.

        ``allowed`` restricts the search to a caller-supplied set of slots, which
        the two-tier system needs: a duplicate may not be placed on a slot its
        original already occupies.
        """
        j = env.current_job_idx
        if self.canonical == "expand":
            classes = canonical_classes(env, j)
            if not classes:
                return []
            ranked = []
            for members in classes.values():
                if allowed is not None:
                    members = [sa for sa in members if sa in allowed]
                    if not members:
                        continue
                best = max(members, key=lambda sa: probs[sa[0] * env.A + sa[1]])
                ranked.append((float(probs[best[0] * env.A + best[1]]), best))
            ranked.sort(key=lambda t: -t[0])
            return [sa for _, sa in ranked[:self.k]]

        slots = feasible_slots(env, j)
        if allowed is not None:
            slots = [sa for sa in slots if sa in allowed]
        if not slots:
            return []
        slots.sort(key=lambda sa: -probs[sa[0] * env.A + sa[1]])
        slots = slots[:self.k]
        if self.canonical == "none":
            return slots
        # fold reward-identical duplicates inside the top-K: the survivor is the
        # one the policy ranked highest, so the argmax is unchanged
        occ = slot_occupancy(env)
        out, seen = [], set()
        for (s, a) in slots:
            key = slot_key(env, j, s, a, occupants=occ.get((s, a), ()))
            if key in seen:
                continue
            seen.add(key)
            out.append((s, a))
        return out

    def decide(self, env, allowed=None):
        """Pick a slot for the current job, optionally restricted to ``allowed``."""
        j = env.current_job_idx
        if j >= env.J:
            return None
        if allowed is not None:
            allowed = set(allowed)
            if not allowed:
                return None
        obs = env._get_obs()
        state = flatten_obs(obs)
        st = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, probs = self.actor(st)
        probs = probs[0].cpu().numpy()

        cands = self._candidates(env, probs, allowed=allowed)
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        scores = np.asarray(self.scorer.score(env, obs, state, cands))
        best = float(scores.max())
        return cands[int(np.flatnonzero(scores >= best - self.tie_tol)[0])]


def make_learned_topk_policy(**kwargs):
    """Policy function compatible with ``nero.evaluation.run_episode``."""
    actor = LearnedTopKActor(**kwargs)
    return actor.decide, actor
