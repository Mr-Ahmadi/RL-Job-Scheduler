"""Honest wall-clock benchmark of the deployment inference paths.

Reports medians and means over many decisions for:
  * max-total oracle decision (its cheap part: exact table lookups)
  * inner PPO forward pass on MPS vs CPU
  * inner full decision path (obs build + forward + masked argmax) on CPU
  * outer PPO forward pass on MPS vs CPU
  * outer full duplication decision on CPU

  * the oracle-free learned top-K decision (Solution 4) in its two useful
    scoring modes, against the oracle top-K it replaces

Also verifies that the CPU deploy paths produce exactly the same actions as
the reference greedy policies used to produce the reported eval scores.
"""

import time

import numpy as np
import torch

from nero.agents.loading import load_inner_agent
from nero.deployment import OnlineActor, SubsetSelector
from nero.envs.job_scheduling.eval import Eval_JobSchedulingEnv
from nero.agents.ppo import PPOAgent, device, flatten_obs
from nero.envs.subset_selector.common import flatten_obs_subset
from nero.envs.subset_selector.eval import Eval_SubsetSelectorEnv
from nero.evaluation import greedy_max_total, make_ppo_greedy, make_topk, valid_actions
from nero.paths import INNER, OUTER, TEST_SETS

torch.set_num_threads(1)

WARMUP, N = 30, 500


def bench(fn, n=N):
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return np.median(ts) * 1e3, ts.mean() * 1e3


def fmt(name, med, mean):
    print(f"{name:44s} median {med:6.3f} ms   mean {mean:6.3f} ms")


def bench_learned_topk(env, ref_agent, k=5):
    """Latency of the oracle-free top-K decision path (Solution 4)."""
    import os

    from nero.search.canonicalization import canonical_classes, feasible_slots
    from nero.search.heads import DEFAULT_DIR
    from nero.search.topk import LearnedTopKActor

    print("== learned top-K (oracle-free, Solution 4) ==")
    if not os.path.exists(f"{DEFAULT_DIR}/reward_model.pth"):
        print("  (skipped: run `python -m scripts.train_reward_model` first)")
        return

    actor = LearnedTopKActor(k=k, mode="reward", device="cpu", threads=1,
                             agent=ref_agent if str(next(ref_agent.actor.parameters()).device)
                             == "cpu" else None)

    # statistics and the timing state both come from a trajectory this actor
    # actually plays, not from an artificial fill pattern
    slots, classes, scored = [], [], []
    probe_env = Eval_JobSchedulingEnv(str(TEST_SETS))
    probe_env.reset()
    while probe_env.current_job_idx < probe_env.J:
        v = feasible_slots(probe_env)
        if not v:
            break
        slots.append(len(v))
        classes.append(len(canonical_classes(probe_env)))
        st = flatten_obs(probe_env._get_obs())
        with torch.no_grad():
            _, p = actor.actor(torch.from_numpy(st).unsqueeze(0))
        scored.append(len(actor._candidates(probe_env, p[0].numpy())))
        probe_env.step(actor.decide(probe_env))
    print(f"  {np.mean(slots):.1f} feasible slots -> {np.mean(classes):.1f} canonical "
          f"classes per decision ({np.mean(slots) / max(np.mean(classes), 1e-9):.2f}x)")
    print(f"  top-{k} list -> {np.mean(scored):.2f} candidates actually scored "
          f"({100 * (1 - np.mean(scored) / k):.0f}% of the scorings skipped)")

    env.reset()
    for _ in range(env.J // 2):        # half-full cluster: the interesting regime
        env.step(actor.decide(env))

    fmt("feasible-slot scan (valid_actions)", *bench(lambda: valid_actions(env), n=200))
    fmt("canonicalisation of all slots", *bench(lambda: canonical_classes(env), n=200))
    for mode in ("reward", "blend"):
        a = LearnedTopKActor(k=k, mode=mode, device="cpu", threads=1, agent=actor.agent)
        fmt(f"full learned top-K decision (K={k}, {mode})",
            *bench(lambda: a.decide(env), n=200))
    oracle_topk = make_topk(ref_agent, k)
    fmt(f"oracle top-K decision (K={k}, needs true reward)",
        *bench(lambda: oracle_topk(env), n=200))


class _ArmedTable:
    """Stands in for ``problem.Tr`` and raises on any read."""

    def __init__(self, real):
        self.real = real

    def __getitem__(self, key):
        raise AssertionError(f"oracle leak: read problem.Tr[{key!r}] at decision time")

    def __len__(self):
        return len(self.real)


def check_no_oracle_access(agent, episodes=2, k=5):
    """Prove the learned search never reads the throughput table while deciding.

    The table and the interference estimator are swapped for objects that raise,
    for the duration of each decision only -- the environment still needs them to
    pay out the reward once the action is taken.  The oracle policies are run
    through the same guard as a control: a guard that nothing trips proves nothing.
    """
    from nero.search.heads import SCORER_MODES
    from nero.search.topk import LearnedTopKActor

    def guarded(fn, env):
        real_tr = env.problem.Tr
        real_est = env._estimate_job_throughput_given_combination

        def boom(*a, **kw):
            raise AssertionError("oracle leak: called "
                                 "_estimate_job_throughput_given_combination")

        env.problem.Tr = _ArmedTable(real_tr)
        env._estimate_job_throughput_given_combination = boom
        try:
            return fn(env)
        finally:
            env.problem.Tr = real_tr
            env._estimate_job_throughput_given_combination = real_est

    env = Eval_JobSchedulingEnv(str(TEST_SETS))
    for mode in SCORER_MODES:
        actor = LearnedTopKActor(k=k, mode=mode, device="cpu", threads=1, agent=agent)
        n = 0
        for _ in range(episodes):
            env.reset()
            while env.current_job_idx < env.J:
                action = guarded(actor.decide, env)
                if action is None:
                    break
                env.step(action)
                n += 1
        print(f"  learned top-K (mode={mode:12s}) made {n:4d} decisions with the "
              f"throughput table armed")

    for name, fn in (("oracle top-K", make_topk(agent, k)),
                     ("gavel_max_total", greedy_max_total)):
        env.reset()
        try:
            guarded(fn, env)
            print(f"  CONTROL FAILED: {name} did not trip the guard")
        except AssertionError:
            print(f"  control: {name} trips the guard on its first decision")


def check_canonicalisation_equivalence(agent, episodes=3, k=5):
    """Deduplicating the top-K list must not change any decision."""
    from nero.search.topk import LearnedTopKActor

    plain = LearnedTopKActor(k=k, mode="reward", canonical="none", device="cpu",
                             threads=1, agent=agent)
    dedup = LearnedTopKActor(k=k, mode="reward", canonical="dedupe", device="cpu",
                             threads=1, agent=agent)
    env = Eval_JobSchedulingEnv(str(TEST_SETS))
    n = 0
    for _ in range(episodes):
        env.reset()
        while env.current_job_idx < env.J:
            a_plain, a_dedup = plain.decide(env), dedup.decide(env)
            if a_plain != a_dedup:
                print(f"  MISMATCH canonicalisation: none={a_plain} dedupe={a_dedup}")
                return
            n += 1
            env.step(a_dedup)
    print(f"  canonicalised top-K identical to plain top-K over {n} decisions")


def main():
    env = Eval_JobSchedulingEnv(str(TEST_SETS))
    env.reset()

    state_dim = len(flatten_obs(env._get_obs()))
    action_dim = env.S * env.A

    ref_agent = PPOAgent(state_dim, action_dim, lr_actor=5e-4, lr_critic=1e-3,
                         gamma=0.99, lamda=0.95, clip=0.2, epochs=10,
                         batch_size=128, checkpoint_dir=str(INNER))
    ref_agent.load()
    ref_agent.actor.eval()

    deploy = OnlineActor(threads=1)

    x = torch.tensor(flatten_obs(env._get_obs()), dtype=torch.float32, device=device).unsqueeze(0)
    x_cpu = x.cpu()

    def oracle():
        env.current_job_idx = 0
        greedy_max_total(env)

    def fwd_mps():
        with torch.no_grad():
            ref_agent.actor(x)

    def fwd_cpu():
        with torch.no_grad():
            deploy.actor(x_cpu)

    def full_cpu():
        deploy.decide(env)

    print("== inner agent ==")
    fmt("forward pass, MPS (original path)", *bench(fwd_mps))
    fmt("forward pass, CPU", *bench(fwd_cpu))
    fmt("full decision (obs+fwd+mask), CPU", *bench(full_cpu))
    fmt("oracle full decision (exact lookups)", *bench(oracle))

    bench_learned_topk(env, ref_agent)

    ss_env = Eval_SubsetSelectorEnv(str(TEST_SETS))
    obs_ss, _ = ss_env.reset()
    ss_state_dim = len(flatten_obs_subset(obs_ss))
    ref_ss = PPOAgent(ss_state_dim, 2, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99,
                      lamda=0.95, clip=0.2, epochs=8, batch_size=64,
                      checkpoint_dir=str(OUTER))
    ref_ss.load()
    ref_ss.actor.eval()

    deploy_ss = SubsetSelector(threads=1)
    st = flatten_obs_subset(obs_ss)
    st_t = torch.tensor(st, dtype=torch.float32, device=device).unsqueeze(0)
    st_t_cpu = st_t.cpu()

    def ss_fwd_mps():
        with torch.no_grad():
            ref_ss.actor(st_t)

    def ss_fwd_cpu():
        with torch.no_grad():
            deploy_ss.actor(st_t_cpu)

    def ss_full():
        deploy_ss.decide(st)

    print("== outer agent (duplication) ==")
    fmt("forward pass, MPS", *bench(ss_fwd_mps))
    fmt("forward pass, CPU", *bench(ss_fwd_cpu))
    fmt("full duplication decision, CPU", *bench(ss_full))

    print("== correctness (CPU deploy == reference greedy) ==")
    ref_policy = make_ppo_greedy(ref_agent)
    agree, steps = 0, 0
    for _ in range(5):
        env.reset()
        while env.current_job_idx < env.J:
            a_ref = ref_policy(env)
            a_fast = deploy.decide(env)
            if a_ref != a_fast:
                print(f"  MISMATCH inner: ref={a_ref} fast={a_fast} at step {env.current_job_idx}")
                return
            agree += 1
            steps += 1
            env.step(a_fast)
    print(f"  inner actions identical over {steps} decisions")

    agree = 0
    for _ in range(20):
        obs_ss, _ = ss_env.reset()
        st = flatten_obs_subset(obs_ss)
        done = False
        while not done:
            can_dup = st[-1] > 0.5
            if can_dup:
                st_t = torch.tensor(st, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    _, p = ref_ss.actor(st_t)
                ref_a = int(torch.argmax(p * torch.tensor([[1.0, 1.0]], device=device), -1).item())
            else:
                ref_a = 0
            fast_a = deploy_ss.decide(st)
            if ref_a != fast_a:
                print(f"  MISMATCH outer: ref={ref_a} fast={fast_a}")
                return
            agree += 1
            obs_ss, r, term, trunc, info = ss_env.step(fast_a)
            done = term or trunc
            st = flatten_obs_subset(obs_ss) if not done else None
    print(f"  outer actions identical over {agree} decisions")

    import os

    from nero.search.heads import DEFAULT_DIR
    if os.path.exists(f"{DEFAULT_DIR}/reward_model.pth"):
        cpu_agent, _, _ = load_inner_agent(device="cpu")
        check_canonicalisation_equivalence(cpu_agent)
        print("== no-oracle-access guard ==")
        check_no_oracle_access(cpu_agent)


if __name__ == "__main__":
    main()
