import json
import numpy as np
import torch
from nero.agents.ppo import PPOAgent
from nero.envs.subset_selector.common import flatten_obs_subset
from nero.envs.subset_selector.train import Train_SubsetSelectorEnv
from nero.envs.subset_selector.eval import Eval_SubsetSelectorEnv
from nero.paths import OUTER, OUTER_CURVES, TEST_SETS


def build_mask(state):
    can_dup = state[-1]
    if can_dup > 0.5:
        return [1.0, 1.0]
    return [1.0, 0.0]


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
            mask = build_mask(st)
            with torch.no_grad():
                a, _, _ = agent.choose_action(st, action_mask=mask)
            obs, reward, term, trunc, info = env.step(int(a))
            total += reward
            done = term or trunc
            st = flatten_obs_subset(obs) if not done else None
        total_sum.append(info["total_reward"])
        rewards.append(total)
    agent.actor.train()
    return np.mean(rewards), rewards, np.mean(total_sum)


def train_ppo(
    n_episodes=10000, eval_interval=20, eval_episodes=20,
    lr_actor=5e-4, lr_critic=1e-3, gamma=0.99, lamda=0.95,
    clip=0.2, epochs=10, batch_size=128, n_steps=512
):
    train_env = Train_SubsetSelectorEnv()
    eval_env = Eval_SubsetSelectorEnv(str(TEST_SETS))
    obs, _ = train_env.reset()
    state_dim = len(flatten_obs_subset(obs))
    action_dim = train_env.action_space.n

    n_updates = (n_episodes * 20) // n_steps
    actor_decay = (1e-4 / lr_actor) ** (1.0 / n_updates) if n_updates > 0 else 1.0
    critic_decay = (2e-4 / lr_critic) ** (1.0 / n_updates) if n_updates > 0 else 1.0

    agent = PPOAgent(
        state_dim, action_dim,
        lr_actor=lr_actor, lr_critic=lr_critic,
        gamma=gamma, lamda=lamda,
        clip=clip, epochs=epochs, batch_size=batch_size,
        checkpoint_dir=str(OUTER),
        ent_start=0.05, ent_end=0.005, ent_decay=0.9995,
        max_grad_norm=0.5,
        lr_actor_decay=actor_decay, lr_critic_decay=critic_decay,
        clip_end=0.1,
    )

    eval_scores = []
    eval_total_sum = []
    best_total = -float('inf')
    steps_since_learn = 0

    for ep in range(1, n_episodes + 1):
        obs, _ = train_env.reset()
        st = flatten_obs_subset(obs)
        done = False
        ep_reward = 0.0

        while not done:
            mask = build_mask(st)
            a_idx, logp, val = agent.choose_action(st, action_mask=mask)
            next_obs, reward, term, trunc, info = train_env.step(int(a_idx))
            done = term or trunc
            reward_float = float(reward)
            agent.store(st, a_idx, logp, val, reward_float, done)
            steps_since_learn += 1
            ep_reward += reward_float
            st = flatten_obs_subset(next_obs) if not done else None

        if steps_since_learn >= n_steps:
            agent.learn()
            train_env.learn_secondary()
            steps_since_learn = 0

        if ep % eval_interval == 0:
            mean_reward, rewards, total_sum = evaluate_model(agent, eval_env, episodes=eval_episodes)
            eval_scores.append(mean_reward)
            eval_total_sum.append(total_sum)
            print(f"[EP {ep}] || eval_mean={mean_reward:.3f} || total_sum={total_sum:.3f}")

            if total_sum > best_total:
                best_total = total_sum
                agent.save()

    if steps_since_learn > 0:
        agent.learn()
        train_env.learn_secondary()

    return agent, eval_scores, eval_total_sum


if __name__ == "__main__":
    agent, eval_scores, eval_total_sum = train_ppo(
        n_episodes=10000,
        eval_interval=20,
        eval_episodes=20,
    )

    # the curve data is written here; scripts/figures.py renders it
    with open(str(OUTER_CURVES / "training_ppo.json"), 'w') as f:
        json.dump(eval_scores, f)

    with open(str(OUTER_CURVES / "eval_total_sum.json"), 'w') as f:
        json.dump(eval_total_sum, f)
