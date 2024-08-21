# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:52:39 2023

@author: Igor
"""

import os
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer

import matplotlib.pyplot as plt

import DroneMALoad
import DroneMA_Sparse_Load

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
rewardsPlot = []
rewardsPlot2 = []

class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Linear(128, 128), nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Linear(128, 128), nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state

def _get_agents(
    agent_1: Optional[BasePolicy] = None,
    agent_2: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = _get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    
    state_shape= observation_space.shape or observation_space.n,
    action_shape=env.action_space.shape or env.action_space.n,
    
    if agent_1 is None:
        net = Net(state_shape, action_shape)
        optim = torch.optim.Adam(net.parameters(), lr=1e-3)
            
        agent_1 = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=0.9,
            estimation_step=3,
            target_update_freq=320,
        )

    if agent_2 is None:
        net = Net(state_shape, action_shape)
        optim = torch.optim.Adam(net.parameters(), lr=1e-3)
            
        agent_2 = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=0.9,
            estimation_step=3,
            target_update_freq=320,
        )

    agents = [agent_1, agent_2]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents, agents


def _get_env():
    return PettingZooEnv(DroneMALoad.env())


if __name__ == "__main__":
    # ======== Step 1: Environment setup =========
    train_envs = DummyVectorEnv([_get_env for _ in range(20)])
    test_envs = DummyVectorEnv([_get_env for _ in range(10)])

    # seed
    seed = 3
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    # ======== Step 2: Agent setup =========
    policy, optim, agents, temp = _get_agents()

    # ======== Step 3: Collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(20000, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    train_collector.collect(n_step=4000)

    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        torch.save(policy.policies[agents[0]].state_dict(), "policy1_Dense_Combined_Large.pth")
        torch.save(policy.policies[agents[1]].state_dict(), "policy2_Dense_Combined_Large.pth")

    def stop_fn(mean_rewards):
        return mean_rewards >= 8000

    def train_fn(epoch, env_step):
        policy.policies[agents[0]].set_eps(0.1)
        policy.policies[agents[1]].set_eps(0.1)
        
    def test_fn(epoch, env_step):
        policy.policies[agents[0]].set_eps(0.05)
        policy.policies[agents[1]].set_eps(0.05)
        
    def reward_metric(rews):
        agent1_rewards = rews[:, 0]  # Rewards of agent 1
        agent2_rewards = rews[:, 1]
        com_rewards = [(x + y) for x, y in zip(agent1_rewards, agent2_rewards)]
        #print(agent1_rewards, agent2_rewards, com_rewards)
        return np.array(com_rewards)

    # ======== Step 5: Run the trainer =========
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=500,
        step_per_epoch=200,
        step_per_collect=40,
        update_per_step=0.1,
        episode_per_test=100,
        batch_size=128,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        test_in_train=False,
        reward_metric=reward_metric,
    )

    print(f"\n==========Result==========\n{result}")