"""Shared construction of the trained inner (placement) PPO agent."""

from nero.envs.job_scheduling.eval import Eval_JobSchedulingEnv
from nero.agents.ppo import PPOAgent, flatten_obs
from nero.paths import INNER, TEST_SETS

INNER_DIR = str(INNER)
INNER_HYPERPARAMS = dict(lr_actor=5e-4, lr_critic=1e-3, gamma=0.99, lamda=0.95,
                         clip=0.2, epochs=10, batch_size=128)


def inner_dims(set_dir=str(TEST_SETS)):
    """(state_dim, action_dim) of the inner scheduling problem."""
    probe = Eval_JobSchedulingEnv(set_dir)
    obs, _ = probe.reset()
    return len(flatten_obs(obs)), probe.S * probe.A


def load_inner_agent(checkpoint_dir=INNER_DIR, device=None, set_dir=str(TEST_SETS),
                     eval_mode=True, load=True):
    """Build the inner PPO agent and (by default) restore its trained weights."""
    state_dim, action_dim = inner_dims(set_dir)
    agent = PPOAgent(state_dim, action_dim, checkpoint_dir=checkpoint_dir,
                     **INNER_HYPERPARAMS)
    if load:
        agent.load()
    if device is not None:
        agent.actor.to(device)
        agent.critic.to(device)
    if eval_mode:
        agent.actor.eval()
        agent.critic.eval()
    return agent, state_dim, action_dim
