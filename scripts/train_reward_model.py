"""Train the learned heads used by the oracle-free top-K search (Solution 4).

The throughput table is used here, offline, purely as a *supervision signal* --
the same way a real deployment would use logged cluster measurements.  Nothing
it produces is consulted at decision time.

Three heads are fitted from rollouts on the training distribution:

Training data is always randomly generated job sets; the 20 sets in
``data/saved_job_sets/`` are the test set and are used here only to *report* a final
score, never to fit or select anything.

* **reward model** ``r_hat(features)`` -- dense regression on the canonical slot
  features (``nero.search.canonicalization.slot_features``).  At every visited state
  the true reward of *every* feasible slot is computed and regressed at once, so
  one env step supplies ~45 labels instead of 1.  Because the features are a
  sufficient statistic for the reward, this is a small, learnable function.
* **value head** ``V(state)`` -- Monte-Carlo regression on discounted
  returns-to-go, collected on the *greedy* episodes only, so it evaluates the
  policy that is actually deployed rather than the exploratory behaviour.
* **Q-head** ``q(state, .)`` -- double DQN with a Polyak-averaged target network.
  Each env step contributes the played transition plus ``--counterfactuals``
  off-policy transitions whose next state comes from the exact delta builder in
  ``nero/search/fast_obs.py``, which widens action coverage without extra env steps.

Usage:  python -m scripts.train_reward_model [--episodes 800] [--epochs 30]
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from nero.agents.loading import load_inner_agent
from nero.search.canonicalization import dense_rewards, feature_dim, slot_features
from nero.envs.job_scheduling.base import MAX_JOBS_PER_GPU
from nero.search.fast_obs import NextObsBuilder
from nero.search.heads import (DEFAULT_DIR, SlotQHead, SlotRewardModel,
                                   SlotValueHead)
from nero.envs.job_scheduling.eval import Eval_JobSchedulingEnv
from nero.envs.job_scheduling.train import Train_JobSchedulingEnv
from nero.agents.ppo import device as default_device
from nero.agents.ppo import flatten_obs
from nero.paths import SCORES, TEST_SETS

CURVE_DIR = str(SCORES)


def all_slots(S, A):
    return [(s, a) for s in range(S) for a in range(A)]


def restricted_job_sampler(exclude, rng):
    """Draw job sets from the training distribution minus the excluded families.

    Used to fit a reward model that has never seen some model families, so the
    online bootstrap in ``scripts/online_reward_model.py`` has something real to learn.
    """
    from nero.envs.problem import JobTable

    types = [(t.model, 1) for t in JobTable
             if not any(t.model.startswith(x) for x in exclude)]
    if not types:
        raise SystemExit(f"--exclude-models removed every job type: {exclude}")

    def sample():
        return rng.choices(types, k=rng.randint(20, 90))
    return sample, types


def collect(agent, episodes, eps, greedy_frac, n_cf, gamma, seed, device,
            exclude=(), verbose=True):
    """Roll out a mixture behaviour policy and log dense labels and returns."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    env = Train_JobSchedulingEnv()
    # the labels that train the reward model may only come from randomly
    # generated job sets; the 20 sets in data/saved_job_sets/ are the test set
    assert not isinstance(env, Eval_JobSchedulingEnv), \
        "reward-model supervision must not come from the held-out test sets"
    assert getattr(env, "set_dir", None) is None, \
        "collection environment must not be backed by a job-set directory"
    sample_jobs = None
    if exclude:
        sample_jobs, kept = restricted_job_sampler(exclude, rng)
        print(f"  restricted to {len(kept)} of 24 job types "
              f"(excluded: {', '.join(exclude)})")
    builder = NextObsBuilder()
    probe = Eval_JobSchedulingEnv(str(TEST_SETS))
    probe.reset()
    S, A = probe.S, probe.A
    slots = all_slots(S, A)
    fdim = feature_dim(probe)

    states, dec_state, feats, dense_r, dense_m = [], [], [], [], []
    mc_return, mc_valid = [], []
    tr_s, tr_a, tr_r, tr_sp, tr_done = [], [], [], [], []
    checked = 0
    t0 = time.time()

    for ep in range(episodes):
        obs, _ = env.reset(sample_jobs()) if sample_jobs else env.reset()
        state = flatten_obs(obs)
        greedy_ep = rng.random() < greedy_frac
        done = False
        ep_rewards = []
        while not done:
            j = env.current_job_idx
            r_all, m_all = dense_rewards(env, j)
            valid = np.flatnonzero(m_all)
            if valid.size == 0:
                break

            s_idx = len(states)
            states.append(state)
            dec_state.append(s_idx)
            feats.append(slot_features(env, j, slots))
            dense_r.append(r_all)
            dense_m.append(m_all)

            st = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, probs = agent.actor(st)
            probs = probs[0].cpu().numpy()
            masked = np.where(m_all, probs, 0.0)
            if greedy_ep:
                action = int(np.argmax(masked))
            elif rng.random() < eps:
                action = int(rng.choice(valid.tolist()))
            else:
                p = masked[valid]
                p = p / p.sum() if p.sum() > 0 else np.full(valid.size, 1.0 / valid.size)
                action = int(np_rng.choice(valid, p=p))

            # off-policy transitions from the exact delta builder (no env step)
            others = [int(v) for v in valid if int(v) != action]
            rng.shuffle(others)
            for cf in others[:n_cf]:
                cs, ca = divmod(cf, A)
                obs_p, done_p = builder.next_obs(env, obs, cs, ca)
                tr_s.append(s_idx)
                tr_a.append(cf)
                tr_r.append(float(r_all[cf]))
                tr_sp.append(len(states))
                states.append(flatten_obs(obs_p))
                tr_done.append(bool(done_p))

            s, a = divmod(action, A)
            obs, rew, term, trunc, _ = env.step((s, a))
            reward = float(sum(rew))
            done = term or trunc
            if checked < 200:  # dense labels must match what the env actually pays
                assert abs(reward - float(r_all[action])) <= 1e-5 * max(1.0, abs(reward)), \
                    f"dense label {r_all[action]} != env reward {reward}"
                checked += 1
            ep_rewards.append(reward)
            next_state = flatten_obs(obs)
            tr_s.append(s_idx)
            tr_a.append(action)
            tr_r.append(reward)
            tr_sp.append(len(states))
            states.append(next_state)
            tr_done.append(bool(done))
            state = next_state

        g = 0.0
        returns = []
        for r in reversed(ep_rewards):
            g = r + gamma * g
            returns.append(g)
        returns.reverse()
        mc_return.extend(returns)
        mc_valid.extend([greedy_ep] * len(returns))
        # a truncated episode can log a decision it never got to play
        while len(mc_return) < len(dec_state):
            mc_return.append(0.0)
            mc_valid.append(False)

        if verbose and (ep + 1) % 50 == 0:
            print(f"  collected {ep + 1}/{episodes} episodes "
                  f"({len(dec_state)} decisions, {len(tr_s)} transitions, "
                  f"{time.time() - t0:.0f}s)", flush=True)

    return dict(
        states=np.asarray(states, dtype=np.float32),
        dec_state=np.asarray(dec_state, dtype=np.int64),
        feats=np.asarray(feats, dtype=np.float32),
        dense_r=np.asarray(dense_r, dtype=np.float32),
        dense_m=np.asarray(dense_m, dtype=bool),
        mc_return=np.asarray(mc_return, dtype=np.float32),
        mc_valid=np.asarray(mc_valid, dtype=bool),
        tr_s=np.asarray(tr_s, dtype=np.int64),
        tr_a=np.asarray(tr_a, dtype=np.int64),
        tr_r=np.asarray(tr_r, dtype=np.float32),
        tr_sp=np.asarray(tr_sp, dtype=np.int64),
        tr_done=np.asarray(tr_done, dtype=np.float32),
        fdim=fdim, S=S, A=A,
    )


def train_reward_model(net, data, epochs, batch_size, device, verbose=True):
    """Dense regression of r_hat on the canonical slot features."""
    X = torch.from_numpy(data["feats"])              # (n, S*A, fdim)
    Y = torch.from_numpy(data["dense_r"])            # (n, S*A)
    M = torch.from_numpy(data["dense_m"])            # (n, S*A)
    n = X.shape[0]
    perm = torch.randperm(n)
    split = int(0.9 * n)
    tr, va = perm[:split], perm[split:]
    history = []
    net.to(device).train()
    for ep in range(epochs):
        idx = tr[torch.randperm(tr.numel())]
        tot, seen = 0.0, 0
        for b in range(0, idx.numel(), batch_size):
            bi = idx[b:b + batch_size]
            xb = X[bi].to(device)
            yb = Y[bi].to(device)
            mb = M[bi].to(device).float()
            pred = net.predict(xb.reshape(-1, xb.shape[-1])).reshape(xb.shape[:2])
            loss = (((pred - yb) ** 2) * mb).sum() / mb.sum().clamp(min=1.0)
            net.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            net.optimizer.step()
            tot += float(loss) * bi.numel()
            seen += bi.numel()
        net.eval()
        with torch.no_grad():
            xb = X[va].to(device)
            yb = Y[va].to(device)
            mb = M[va].to(device).float()
            pred = net.predict(xb.reshape(-1, xb.shape[-1])).reshape(xb.shape[:2])
            mae = float((((pred - yb).abs()) * mb).sum() / mb.sum().clamp(min=1.0))
            big = torch.where(mb > 0, pred, torch.full_like(pred, -1e9))
            big_y = torch.where(mb > 0, yb, torch.full_like(yb, -1e9))
            top1 = float((big.argmax(-1) == big_y.argmax(-1)).float().mean())
            regret = float((big_y.max(-1).values
                            - big_y.gather(-1, big.argmax(-1, keepdim=True)).squeeze(-1)).mean())
        net.train()
        history.append(dict(epoch=ep + 1, train_mse=tot / max(seen, 1),
                            val_mae=mae, val_top1=top1, val_regret=regret))
        if verbose:
            print(f"  reward-model epoch {ep + 1:2d}/{epochs}  "
                  f"train_mse={tot / max(seen, 1):.3e}  val_mae={mae:.4e}  "
                  f"argmax-agreement={top1:.3f}  mean-regret={regret:.4e}", flush=True)
    net.eval()
    return history


def train_value_head(net, data, epochs, batch_size, device, verbose=True):
    """Monte-Carlo regression of V on the greedy-episode decision states."""
    keep = np.flatnonzero(data["mc_valid"])
    if keep.size < batch_size:
        print("  (skipped: not enough greedy episodes)")
        return []
    X = torch.from_numpy(data["states"][data["dec_state"][keep]])
    Y = torch.from_numpy(data["mc_return"][keep])
    n = X.shape[0]
    perm = torch.randperm(n)
    split = int(0.9 * n)
    tr, va = perm[:split], perm[split:]
    history = []
    net.to(device).train()
    for ep in range(epochs):
        idx = tr[torch.randperm(tr.numel())]
        tot, seen = 0.0, 0
        for b in range(0, idx.numel(), batch_size):
            bi = idx[b:b + batch_size]
            pred = net(X[bi].to(device)).squeeze(-1)
            loss = F.mse_loss(pred, Y[bi].to(device))
            net.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            net.optimizer.step()
            tot += float(loss) * bi.numel()
            seen += bi.numel()
        net.eval()
        with torch.no_grad():
            pred = net(X[va].to(device)).squeeze(-1)
            mae = float((pred - Y[va].to(device)).abs().mean())
        net.train()
        history.append(dict(epoch=ep + 1, train_mse=tot / max(seen, 1), val_mae=mae))
        if verbose:
            print(f"  value-head epoch {ep + 1:2d}/{epochs}  "
                  f"train_mse={tot / max(seen, 1):.3e}  val_mae={mae:.4e}", flush=True)
    net.eval()
    return history


def train_q_head(net, data, gamma, iters, batch_size, tau, device, verbose=True):
    """Double DQN with a Polyak-averaged target network."""
    S, A = data["S"], data["A"]
    states = torch.from_numpy(data["states"])
    tr_s = torch.from_numpy(data["tr_s"])
    tr_a = torch.from_numpy(data["tr_a"])
    tr_r = torch.from_numpy(data["tr_r"])
    tr_sp = torch.from_numpy(data["tr_sp"])
    tr_done = torch.from_numpy(data["tr_done"])
    occ_lo = S * A * 2 * 6
    occ = data["states"][:, occ_lo:occ_lo + S * A]
    # replayed observations, so feasibility is read from the stored occupancy
    # block rather than from a live environment -- same rule, same constant
    valid = torch.from_numpy((occ < MAX_JOBS_PER_GPU).astype(np.float32))

    n = tr_s.numel()
    target = SlotQHead(states.shape[1], S * A).to(device)
    target.load_state_dict(net.state_dict())
    target.eval()
    net.to(device).train()

    history = []
    g = torch.Generator().manual_seed(0)
    running, report = 0.0, max(iters // 20, 1)
    for it in range(iters):
        bi = torch.randint(0, n, (batch_size,), generator=g)
        sb = states[tr_s[bi]].to(device)
        spb = states[tr_sp[bi]].to(device)
        ab = tr_a[bi].to(device).unsqueeze(-1)
        rb = tr_r[bi].to(device)
        db = tr_done[bi].to(device)
        vb = valid[tr_sp[bi]].to(device)

        with torch.no_grad():
            a_star = net(spb).masked_fill(vb == 0, -1e9).argmax(-1, keepdim=True)
            q_next = target(spb).gather(-1, a_star).squeeze(-1)
            q_next = torch.where(vb.sum(-1) > 0, q_next, torch.zeros_like(q_next))
            y = rb + gamma * (1.0 - db) * q_next

        q = net(sb).gather(-1, ab).squeeze(-1)
        loss = F.smooth_l1_loss(q, y)
        net.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        net.optimizer.step()
        running += float(loss)

        with torch.no_grad():   # Polyak: a slowly moving target keeps TD stable
            for tp, p in zip(target.parameters(), net.parameters()):
                tp.mul_(1 - tau).add_(tau * p)

        if (it + 1) % report == 0:
            history.append(dict(iter=it + 1, td_loss=running / report))
            if verbose:
                print(f"  q-head iter {it + 1:6d}/{iters}  "
                      f"td_loss={running / report:.4e}", flush=True)
            running = 0.0
    net.eval()
    return history


def quick_eval(mode, k, beta, learned_dir, device, episodes=20, agent=None):
    """Evaluate the oracle-free top-K actor on the held-out sets."""
    from nero.search.topk import LearnedTopKActor
    from nero.evaluation import run_episode

    actor = LearnedTopKActor(k=k, mode=mode, beta=beta, learned_dir=learned_dir,
                             device=device, threads=0, agent=agent)
    env = Eval_JobSchedulingEnv(str(TEST_SETS))
    return [run_episode(env, actor.decide) for _ in range(episodes)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=800, help="rollout episodes to collect")
    ap.add_argument("--eps", type=float, default=0.35,
                    help="uniform-random action rate on exploratory episodes")
    ap.add_argument("--greedy-frac", type=float, default=0.4,
                    help="share of episodes played greedily (these carry the MC returns)")
    ap.add_argument("--counterfactuals", type=int, default=2,
                    help="off-policy transitions logged per env step")
    ap.add_argument("--epochs", type=int, default=30, help="reward-model epochs")
    ap.add_argument("--value-epochs", type=int, default=15)
    ap.add_argument("--q-iters", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--tau", type=float, default=0.005, help="Polyak rate for the Q target")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--q-lr", type=float, default=1e-4)
    ap.add_argument("--out-dir", default=DEFAULT_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=str(default_device))
    ap.add_argument("--exclude-models", default="",
                    help="comma-separated model families to keep OUT of the "
                         "offline supervision (e.g. 'ResNet-50,Transformer')")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    agent, state_dim, action_dim = load_inner_agent(device=device)
    print(f"device={device}  state_dim={state_dim}  action_dim={action_dim}")
    print(f"collecting {args.episodes} episodes (eps={args.eps}, "
          f"greedy_frac={args.greedy_frac}, "
          f"counterfactuals/step={args.counterfactuals}) ...", flush=True)
    t0 = time.time()
    exclude = tuple(x.strip() for x in args.exclude_models.split(",") if x.strip())
    data = collect(agent, args.episodes, args.eps, args.greedy_frac,
                   args.counterfactuals, args.gamma, args.seed, device,
                   exclude=exclude)
    print(f"  {data['dec_state'].size} labelled decisions "
          f"({int(data['mc_valid'].sum())} from greedy episodes), "
          f"{data['tr_s'].size} transitions, {data['states'].shape[0]} states, "
          f"{time.time() - t0:.0f}s", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    reward_model = SlotRewardModel(data["fdim"], args.out_dir, lr=args.lr)
    value_head = SlotValueHead(state_dim, args.out_dir, lr=args.lr)
    q_head = SlotQHead(state_dim, action_dim, args.out_dir, lr=args.q_lr)

    print("training reward model (dense supervision over all feasible slots) ...")
    r_hist = train_reward_model(reward_model, data, args.epochs, args.batch_size, device)
    reward_model.save()

    print("training value head (Monte-Carlo returns, greedy episodes) ...")
    v_hist = train_value_head(value_head, data, args.value_epochs, args.batch_size, device)
    if v_hist:
        value_head.save()

    print("training Q-head (double DQN, Polyak target) ...")
    q_hist = train_q_head(q_head, data, args.gamma, args.q_iters, args.batch_size,
                          args.tau, device)
    q_head.save()
    print(f"saved -> {args.out_dir}/", flush=True)

    # the diagnostics file follows --out-dir, so a scratch run cannot overwrite
    # the committed one
    os.makedirs(CURVE_DIR, exist_ok=True)
    suffix = "" if os.path.abspath(args.out_dir) == os.path.abspath(DEFAULT_DIR) \
        else "_" + os.path.basename(os.path.normpath(args.out_dir))
    diag = dict(reward_model=r_hist, value_head=v_hist, q_head=q_hist,
                config=vars(args), n_decisions=int(data["dec_state"].size),
                n_transitions=int(data["tr_s"].size))
    if not args.skip_eval:
        cpu_agent, _, _ = load_inner_agent(device="cpu")
        scores = {}
        for mode in ("reward", "reward_value", "q", "blend"):
            rewards = quick_eval(mode, 5, 0.5, args.out_dir, "cpu", agent=cpu_agent)
            scores[mode] = rewards
            print(f"held-out K=5 mode={mode:12s} mean={np.mean(rewards):7.4f} "
                  f"± {np.std(rewards, ddof=1):6.4f}", flush=True)
        diag["heldout_k5"] = scores
    with open(f"{CURVE_DIR}/learned_topk_training{suffix}.json", "w") as f:
        json.dump(diag, f, indent=2)


if __name__ == "__main__":
    main()
