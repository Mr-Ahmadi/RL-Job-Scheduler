import json
import numpy as np
import torch
from environment.ppo.core import (
    device, PPOAgent, flatten_obs, plot_learning_curve
)
from environment.job_scheduling.eval import Eval_JobSchedulingEnv
from environment.job_scheduling.train import Train_JobSchedulingEnv


def get_valid_action_indices(env):
    valid = []
    j = env.current_job_idx
    if env.assignment[j].sum() > 0:
        return valid
    for s in range(env.S):
        for a in range(env.A):
            colocated = [jj for jj in range(env.J) if jj != j and env.assignment[jj, s, a] > 0]
            if len(colocated) < 2:
                valid.append(s * env.A + a)
    return valid


def select_action_eval(state, env, actor_model):
    st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _, probs = actor_model(st)
    mask = torch.zeros_like(probs)
    mask[0, get_valid_action_indices(env)] = 1
    masked = probs * mask
    masked = masked / masked.sum(dim=-1, keepdim=True)
    return int(torch.argmax(masked, dim=-1).item())


def evaluate_model(actor_model, env, episodes=20):
    set_files = env.set_files
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        state = flatten_obs(obs)
        done = False
        total = 0
        while not done:
            a_idx = select_action_eval(state, env, actor_model)
            s, a = divmod(a_idx, env.A)
            obs, rew, term, trunc, _ = env.step((s, a))
            total += sum(rew)
            done = term or trunc
            state = flatten_obs(obs) if not done else None
        rewards.append(total)
    mean = np.mean(rewards)
    return mean, rewards, set_files


class JobSchedulingPPOAgent(PPOAgent):
    def __init__(self, state_dim, action_dim):
        n_updates = (15000 * 55) // 2048
        actor_decay = (1e-4 / 5e-4) ** (1.0 / n_updates) if n_updates > 0 else 1.0
        critic_decay = (2e-4 / 1e-3) ** (1.0 / n_updates) if n_updates > 0 else 1.0
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=5e-4,
            lr_critic=1e-3,
            gamma=0.99,
            lamda=0.95,
            clip=0.2,
            epochs=10,
            batch_size=128,
            checkpoint_dir='models/job_scheduling/ppo',
            ent_start=0.05,
            ent_end=0.005,
            ent_decay=0.9995,
            max_grad_norm=0.5,
            lr_actor_decay=actor_decay,
            lr_critic_decay=critic_decay,
            clip_end=0.1
        )

    def choose_action(self, state, env):
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        dist, probs = self.actor(state_t)
        mask = torch.zeros_like(probs)
        idxs = get_valid_action_indices(env)
        mask[0, idxs] = 1
        masked = probs * mask
        masked = masked / masked.sum(dim=-1, keepdim=True)
        masked_dist = torch.distributions.Categorical(masked)
        action = masked_dist.sample()
        logp = masked_dist.log_prob(action).item()
        val = self.critic(state_t).squeeze().item()
        return action.item(), logp, val


if __name__ == "__main__":
    train_env = Train_JobSchedulingEnv()
    eval_env = Eval_JobSchedulingEnv("saved_job_sets")
    obs, _ = train_env.reset()
    state_dim = len(flatten_obs(obs))
    action_dim = train_env.S * train_env.A

    agent = JobSchedulingPPOAgent(state_dim, action_dim)

    n_steps = 2048
    total_episodes = 15000
    eval_freq_episodes = 300
    best_average_reward = -float('inf')
    evaluation_scores = []
    global_step = 0
    episode = 0
    steps_since_learn = 0

    print(f"Training for {total_episodes} episodes, update every {n_steps} steps, eval every {eval_freq_episodes} episodes")
    print(f"Device: {device}")

    obs, _ = train_env.reset()
    state = flatten_obs(obs)
    done = False
    ep_reward = 0

    while episode < total_episodes:
        if done:
            obs, _ = train_env.reset()
            state = flatten_obs(obs)
            done = False
            ep_reward = 0
            episode += 1

        action, logp, val = agent.choose_action(state, train_env)
        s, a = divmod(action, train_env.A)
        obs, rew, term, trunc, _ = train_env.step((s, a))
        done = term or trunc
        reward = sum(rew)
        agent.store(state, action, logp, val, reward, done)
        global_step += 1
        steps_since_learn += 1
        ep_reward += reward
        state = flatten_obs(obs) if not done else None

        if steps_since_learn >= n_steps:
            agent.learn()
            steps_since_learn = 0

        if done and episode % eval_freq_episodes == 0:
            agent.actor.eval()
            mean_reward, rewards_list, _ = evaluate_model(agent.actor, eval_env, 20)
            agent.actor.train()
            evaluation_scores.append(mean_reward)
            pct = 100.0 * episode / total_episodes
            print(f"Ep {episode}/{total_episodes} ({pct:.0f}%) | "
                  f"Eval: {mean_reward:.4f} | "
                  f"Steps: {global_step} | "
                  f"Entropy: {agent.ent_coef:.4f} | "
                  f"Best: {best_average_reward:.4f}")

            if mean_reward > best_average_reward:
                best_average_reward = mean_reward
                agent.save()
                print(f"  >> New best model saved: {best_average_reward:.4f}")

    if steps_since_learn > 0:
        agent.learn()

    print(f"\nTraining complete. Best eval reward: {best_average_reward:.4f}")
    plot_learning_curve(evaluation_scores, file_path='curves/job_scheduling/training_ppo.png',
                        title="PPO Training (Episode-Based)")

    with open('curves/job_scheduling/training_ppo.json', 'w') as f:
        json.dump(evaluation_scores, f)

    agent.load()
    agent.actor.eval()
    eval_env = Eval_JobSchedulingEnv("saved_job_sets")
    avg, rewards, set_files = evaluate_model(agent.actor, eval_env, episodes=20)
    print("External evaluation average rewards:", rewards)
    print(f"Mean: {np.mean(rewards):.4f} | Best: {np.max(rewards):.4f} | Std: {np.std(rewards):.4f}")
    print("Evaluation set files:", set_files)

    with open('curves/job_scheduling/evaluation_scores_ppo.json', 'w') as f:
        json.dump(rewards, f)
