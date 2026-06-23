import json
import numpy as np
import torch
from environment.ppo.core import PPOAgent, plot_learning_curve
from environment.subset_selector._requirements import flatten_obs_subset
from environment.subset_selector.train import Train_SubsetSelectorEnv
from environment.subset_selector.eval import Eval_SubsetSelectorEnv


def evaluate_model(agent, env, episodes=20):
    agent.actor.eval()
    rewards = []
    total_sum = []
    for _ in range(episodes):
        obs, _ = env.reset()
        st = flatten_obs_subset(obs)
        done = False
        total = 0.0
        while not done:
            with torch.no_grad():
                a, _, _ = agent.choose_action(st)
            obs, reward, term, trunc, info = env.step(int(a))
            total += reward
            done = term or trunc
            st = flatten_obs_subset(obs) if not done else None
        total_sum.append(info["total_reward"])
        rewards.append(total)
    agent.actor.train()
    return np.mean(rewards), rewards, np.mean(total_sum)


def train_ppo(
    n_episodes=5000, eval_interval=20, eval_episodes=20,
    lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, lamda=0.95,
    clip=0.2, epochs=8, batch_size=64
):
    train_env = Train_SubsetSelectorEnv()
    eval_env = Eval_SubsetSelectorEnv("saved_job_sets")
    obs, _ = train_env.reset()
    state_dim = len(flatten_obs_subset(obs))
    action_dim = train_env.action_space.n

    agent = PPOAgent(
        state_dim, action_dim,
        lr_actor=lr_actor, lr_critic=lr_critic,
        gamma=gamma, lamda=lamda,
        clip=clip, epochs=epochs, batch_size=batch_size,
        checkpoint_dir='models/subset_selector/ppo',
        ent_start=0.05, ent_end=0.001, ent_decay=0.998,
        max_grad_norm=0.5,
    )

    eval_scores = []
    eval_total_sum = []
    best_total = -float('inf')

    for ep in range(1, n_episodes + 1):
        obs, _ = train_env.reset()
        st = flatten_obs_subset(obs)
        done = False
        ep_reward = 0.0

        while not done:
            a_idx, logp, val = agent.choose_action(st)
            next_obs, reward, term, trunc, info = train_env.step(int(a_idx))
            done = term or trunc
            reward_float = float(reward)
            agent.store(st, a_idx, logp, val, reward_float, done)
            ep_reward += reward_float
            st = flatten_obs_subset(next_obs) if not done else None

        agent.learn()
        agent.save()

        if ep % eval_interval == 0:
            mean_reward, rewards, total_sum = evaluate_model(agent, eval_env, episodes=eval_episodes)
            eval_scores.append(mean_reward)
            eval_total_sum.append(total_sum)
            print(f"[EP {ep}] || eval_mean={mean_reward:.3f} || total_sum={total_sum:.3f}")

            if total_sum > best_total:
                best_total = total_sum
                agent.save()

    return agent, eval_scores, eval_total_sum


if __name__ == "__main__":
    agent, eval_scores, eval_total_sum = train_ppo(
        n_episodes=5000,
        eval_interval=20,
        eval_episodes=20,
        lr_actor=2e-4,
        lr_critic=1e-3,
        clip=0.2,
        epochs=8,
        batch_size=64
    )

    plot_learning_curve(eval_scores, file_path='curves/subset_selector/training_ppo.png',
                        title="Subset Selector PPO Evaluation (Mean Reward)")
    plot_learning_curve(eval_total_sum, file_path='curves/subset_selector/total_sum.png',
                        title="Subset Selector PPO Evaluation (Total Sum)")

    with open('curves/subset_selector/training_ppo.json', 'w') as f:
        json.dump(eval_scores, f)

    with open('curves/subset_selector/eval_total_sum.json', 'w') as f:
        json.dump(eval_total_sum, f)
