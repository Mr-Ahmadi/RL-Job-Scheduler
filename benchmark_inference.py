"""Honest wall-clock benchmark of the deployment inference paths.

Reports medians and means over many decisions for:
  * max-total oracle decision (its cheap part: exact table lookups)
  * inner PPO forward pass on MPS vs CPU
  * inner full decision path (obs build + forward + masked argmax) on CPU
  * outer PPO forward pass on MPS vs CPU
  * outer full duplication decision on CPU

Also verifies that the CPU deploy paths produce exactly the same actions as
the reference greedy policies used to produce the reported eval scores.
"""

import time

import numpy as np
import torch

from deploy.online_actor import OnlineActor, SubsetSelector
from environment.job_scheduling.eval import Eval_JobSchedulingEnv
from environment.ppo.core import PPOAgent, device, flatten_obs
from environment.subset_selector._requirements import flatten_obs_subset
from environment.subset_selector.eval import Eval_SubsetSelectorEnv
from eval_all_policies import greedy_max_total, make_ppo_greedy, run_episode, valid_actions

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


def main():
    env = Eval_JobSchedulingEnv("saved_job_sets")
    env.reset()

    state_dim = len(flatten_obs(env._get_obs()))
    action_dim = env.S * env.A

    ref_agent = PPOAgent(state_dim, action_dim, lr_actor=5e-4, lr_critic=1e-3,
                         gamma=0.99, lamda=0.95, clip=0.2, epochs=10,
                         batch_size=128, checkpoint_dir="models/job_scheduling/ppo")
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

    ss_env = Eval_SubsetSelectorEnv("saved_job_sets")
    obs_ss, _ = ss_env.reset()
    ss_state_dim = len(flatten_obs_subset(obs_ss))
    ref_ss = PPOAgent(ss_state_dim, 2, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99,
                      lamda=0.95, clip=0.2, epochs=8, batch_size=64,
                      checkpoint_dir="models/subset_selector/ppo")
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


if __name__ == "__main__":
    main()
