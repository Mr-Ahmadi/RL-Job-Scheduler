"""Two-tier scheduling with the oracle-free top-K search doing the placements.

The two-tier system (``nero/envs/subset_selector``) runs in two phases:

1. a **primary** phase that places every queued job once, and
2. a **secondary** phase where the outer agent decides, job by job, whether to
   duplicate it, and a fine-tuned inner agent places the duplicate.

Both phases currently place with a masked ``argmax`` over the inner policy.
``LearnedTopK_SubsetSelectorEnv`` swaps either phase for the Solution 4 search.
The outer duplication policy is untouched.

**The primary phase is exact.** No duplicates exist yet when it runs, so the
18-d canonical features fully determine the reward of every candidate slot, and
the search is the same one measured in ``scripts/evaluate.py``.

**The secondary phase is not, and that is why it is off by default.** Placing a
duplicate pays ``- discount(s) * (tr_orig + tr_dup)``, where ``tr_orig`` is the
throughput of the copy already running. ``tr_orig`` does not vary with the
candidate slot, but it *scales* the discount term, and it is not in the feature
vector — so the learned score ranks duplicate slots by an expression missing one
term. Deployed for real you would feed in the measured throughput of the running
copy, which is observable; inside this simulator reading it would mean querying
the throughput table, which is the thing the search exists to avoid. Enable it
with ``secondary=True`` to measure the cost of the approximation.
"""

import numpy as np

from nero.envs.subset_selector.common import get_valid_action_indices
from nero.envs.subset_selector.eval import Eval_SubsetSelectorEnv

from nero.agents.loading import load_inner_agent
from nero.search.topk import LearnedTopKActor


class LearnedTopK_SubsetSelectorEnv(Eval_SubsetSelectorEnv):
    """Two-tier evaluation env whose placements come from the learned top-K search."""

    def __init__(self, set_dir, k=5, mode="reward", beta=0.5, canonical="dedupe",
                 primary=True, secondary=False, agent=None, secondary_agent=None,
                 learned_dir=None, device="cpu"):
        """``secondary_agent`` selects which policy proposes the duplicate slots.

        By default both phases search over the *primary* inner policy, which means
        the fine-tuned secondary network is bypassed rather than re-ranked. Pass
        the secondary weights here to search over them instead.
        """
        super().__init__(set_dir)
        self.use_primary = primary
        self.use_secondary = secondary
        if agent is None:
            agent, _, _ = load_inner_agent(device=device)
        kw = dict(k=k, mode=mode, beta=beta, canonical=canonical, device=device,
                  threads=0, agent=agent)
        if learned_dir:
            kw["learned_dir"] = learned_dir
        self.topk = LearnedTopKActor(**kw)
        if secondary_agent is None:
            self.topk_secondary = self.topk
        else:
            self.topk_secondary = LearnedTopKActor(
                **{**kw, "agent": secondary_agent,
                   "reward_model": self.topk.scorer.reward_model,
                   "value_head": self.topk.scorer.value_head,
                   "q_head": self.topk.scorer.q_head})

    def _search_place(self, dup_info=None):
        """Place jobs with the search until the phase ends, returning the reward."""
        for_dup = bool(dup_info and dup_info.get("for_dup"))
        actor = self.topk_secondary if for_dup else self.topk
        original_job_idx = dup_info.get("original_job_idx") if dup_info else None

        total, done = 0.0, False
        while not done:
            allowed = None
            if for_dup and self.env.current_job_idx == dup_info.get("dup_idx"):
                # a duplicate may not land on a slot its original occupies
                allowed = [divmod(i, self.env.A) for i in get_valid_action_indices(
                    self.env, for_dup=True, original_job_idx=original_job_idx)]
            slot = actor.decide(self.env, allowed=allowed)
            if slot is None:
                break
            next_obs, rew, term, trunc, _ = self.env.step(slot)
            total += float(np.sum(rew))
            done = bool(term or trunc)
            self.last_obs = next_obs
            if for_dup:
                break
        return total

    def _schedule_with_primary_agent(self, dup_info=None):
        if not self.use_primary:
            return super()._schedule_with_primary_agent(dup_info=dup_info)
        return self._search_place(dup_info)

    def _schedule_with_secondary_agent(self, dup_info=None):
        if not self.use_secondary:
            return super()._schedule_with_secondary_agent(dup_info=dup_info)
        return self._search_place(dup_info)
