import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from environment.job_scheduling.eval import Eval_JobSchedulingEnv
from environment.ppo.core import device, flatten_obs


def gavel_score(env, s, a, use_delta=True):
    j = env.current_job_idx
    colocated = [oj for oj in range(env.J) if oj != j and env.assignment[oj, s, a] > 0]
    if len(colocated) >= 2:
        return -float('inf')
    prev_tp = {oj: env._estimate_job_throughput_given_combination(oj, s, a) for oj in colocated}
    env.assignment[j, s, a] = 1
    new_tp = env._estimate_job_throughput_given_combination(j, s, a)
    upd_tp = {oj: env._estimate_job_throughput_given_combination(oj, s, a) for oj in colocated}
    env.assignment[j, s, a] = 0
    if use_delta:
        delta = sum(upd_tp[oj] - prev_tp[oj] for oj in colocated) if colocated else 0.0
        return new_tp + delta
    return new_tp


def gavel_max_throughput(env):
    best_score, best_action = -float('inf'), None
    for s in range(env.S):
        for a in range(env.A):
            score = gavel_score(env, s, a, use_delta=False)
            if score > best_score:
                best_score, best_action = score, (s, a)
    return best_action


def gavel_max_total(env):
    best_score, best_action = -float('inf'), None
    for s in range(env.S):
        for a in range(env.A):
            score = gavel_score(env, s, a, use_delta=True)
            if score > best_score:
                best_score, best_action = score, (s, a)
    return best_action


def random_policy(env):
    valid = []
    j = env.current_job_idx
    for s in range(env.S):
        for a in range(env.A):
            colocated = [oj for oj in range(env.J) if oj != j and env.assignment[oj, s, a] > 0]
            if len(colocated) < 2:
                valid.append((s, a))
    return valid[np.random.randint(len(valid))] if valid else None


def get_valid_action_indices(env):
    valid = []
    j = env.current_job_idx
    if env.assignment[j].sum() > 0:
        return valid
    for s in range(env.S):
        for a in range(env.A):
            colocated = [oj for oj in range(env.J) if oj != j and env.assignment[oj, s, a] > 0]
            if len(colocated) < 2:
                valid.append(s * env.A + a)
    return valid


def evaluate_policy(policy_fn, env, episodes=20):
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action = policy_fn(env)
            if action is None:
                break
            obs, reward, term, trunc, _ = env.step(action)
            total += sum(reward)
            done = term or trunc
        rewards.append(total)
    return np.mean(rewards), rewards


if __name__ == "__main__":
    os.makedirs('curves/job_scheduling', exist_ok=True)

    eval_env = Eval_JobSchedulingEnv("saved_job_sets")

    color_map = {
        'ppo': '#2ecc71', 'gavel_max_total': '#3498db',
        'gavel_max_throughput': '#9b59b6', 'random': '#e74c3c',
    }

    policies = {
        "gavel_max_total": gavel_max_total,
        "gavel_max_throughput": gavel_max_throughput,
        "random": random_policy,
    }

    # --- PPO Top-K ---
    try:
        from environment.ppo.core import ActorNet
        import job_scheduling_ppo as jsp

        sample_obs, _ = eval_env.reset()
        sample_state = flatten_obs(sample_obs)
        state_dim = len(sample_state)
        action_dim = eval_env.S * eval_env.A

        agent = jsp.JobSchedulingPPOAgent(state_dim, action_dim)
        agent.actor.load()
        agent.actor.eval()
        print(f"PPO model loaded: state_dim={state_dim}, action_dim={action_dim}")

        def make_ppo_topk(k=3):
            def ppo_topk_policy(env):
                obs = flatten_obs(env._get_obs())
                st = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    _, probs = agent.actor(st)
                mask = torch.zeros_like(probs)
                idxs = get_valid_action_indices(env)
                if not idxs:
                    return None
                mask[0, idxs] = 1
                masked = probs * mask
                masked = masked / masked.sum(dim=-1, keepdim=True)
                top_k = torch.topk(masked[0], min(k, len(idxs))).indices.tolist()
                best_action, best_score = None, -float('inf')
                for a_idx in top_k:
                    s, a = divmod(a_idx, env.A)
                    score = gavel_score(env, s, a, use_delta=True)
                    if score > best_score:
                        best_score, best_score_action = score, (s, a)
                return best_score_action
            return ppo_topk_policy

        ppo_topk_k3 = make_ppo_topk(3)
        ppo_topk_k5 = make_ppo_topk(5)
        policies["ppo_topk_k3"] = ppo_topk_k3
        policies["ppo_topk_k5"] = ppo_topk_k5
        color_map["ppo_topk_k3"] = '#f39c12'
        color_map["ppo_topk_k5"] = '#e67e22'
        print("PPO Top-K policies registered")
    except Exception as e:
        print(f"PPO Top-K not available: {e}")

    results = {}
    for name, policy_fn in policies.items():
        mean_r, rewards = evaluate_policy(policy_fn, eval_env, episodes=20)
        results[name] = {"mean": mean_r, "rewards": rewards, "std": np.std(rewards)}
        print(f"{name}: mean = {mean_r:.4f} ± {np.std(rewards):.4f}, all = {[f'{r:.2f}' for r in rewards]}")

        with open(f'curves/job_scheduling/evaluation_scores_{name}.json', 'w') as f:
            json.dump(rewards, f)

    ppo_path = 'curves/job_scheduling/evaluation_scores_ppo.json'
    if os.path.exists(ppo_path):
        with open(ppo_path) as f:
            ppo_scores = json.load(f)
        results["ppo"] = {"mean": np.mean(ppo_scores), "rewards": ppo_scores, "std": np.std(ppo_scores)}
        print(f"ppo: mean = {np.mean(ppo_scores):.4f} ± {np.std(ppo_scores):.4f}")

    labels = list(results.keys())
    means = [results[k]["mean"] for k in labels]
    errs = [results[k]["std"] for k in labels]

    plt.figure(figsize=(10, 6), dpi=150)
    colors = [color_map.get(l, '#95a5a6') for l in labels]
    bars = plt.bar(labels, means, yerr=errs, capsize=5, color=colors, alpha=0.85, edgecolor='gray', linewidth=0.5)
    plt.ylabel("Mean Cumulative Reward", fontsize=13)
    plt.title("Scheduling Policy Comparison on 20 Held-Out Job Sets", fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    for bar, mean, err in zip(bars, means, errs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.3,
                 f'{mean:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('curves/job_scheduling/policy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nComparison plot saved to curves/job_scheduling/policy_comparison.png")
