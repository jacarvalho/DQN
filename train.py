"""
PyTorch implementation of DQN, largely based on https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
Should work on all gym environments with discrete action-spaces
"""
import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Use the gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MDP transition
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'nextstate', 'done'))


class ReplayMemory:

    def __init__(self, capacity=100):
        self._capacity = capacity
        self._memory = []
        self._pos = 0

    def __len__(self):
        return len(self._memory)

    def add(self, t):
        """
        Add a transition to the replay memory.

        :param t: Transition tuple
        """
        if len(self._memory) >= self._capacity:
            # If the memory is full, replace as a FIFO
            self._memory[self._pos] = t
        else:
            self._memory.append(t)

        self._pos = (self._pos + 1) % self._capacity

    def sample(self, batch_size):
        return random.sample(self._memory, batch_size)


class DQN(nn.Module):

    def __init__(self, s_dim, a_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(s_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, a_dim)
        self.a_dim = a_dim

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x.view(x.size(0), -1)


class EpsilonSchedule:

    def __init__(self, eps_start=0.9, eps_end=0.05, eps_decay=0.9999):
        self._eps_start = eps_start
        self._eps_end = eps_end
        self._eps_decay = eps_decay


class Agent:

    def __init__(self, env=None,
                 q_net=None, q_net_target=None, lr=1e-3,
                 replay_memory=None, batch_size=None,
                 gamma=None,
                 eps_start=None, eps_end=None, eps_decay=None, target_update=10):
        self.env = env
        self.policy_net = q_net
        self.target_net = q_net_target
        self.replay_memory = replay_memory
        self.batch_size = batch_size
        self.a_dim = q_net.a_dim
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.steps = 0

        self.loss_criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(q_net.parameters(), lr=1e-3)

    def select_action(self, s):
        eps = np.random.rand(1)
        if eps < self.eps_start:
            action = torch.tensor([[np.random.choice(self.a_dim)]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                action = self.policy_net(s).argmax(1).view(1, 1)
        self.steps += 1
        return action

    def optimize(self):
        transitions = self.replay_memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # TODO: Use not done here
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.nextstate)),
                                      device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.nextstate if s is not None])

        state_batch = torch.cat(batch.state, dim=0)
        action_batch = torch.cat(batch.action, dim=0)
        reward_batch = torch.cat(batch.reward, dim=0)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = reward_batch + self.gamma * next_state_values

        loss = self.loss_criterion(expected_state_action_values.unsqueeze(1), state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self, n_episodes):
        for i_episode in range(n_episodes):
            print("ep: {:4}/{} -> ".format(i_episode+1, n_episodes), end=' ')
            state = env.reset()
            state = torch.from_numpy(state.reshape(1, -1)).to(device, dtype=torch.float32)
            for t in count():
                action = self.select_action(state)
                nextstate, reward, done, _ = self.env.step(action.item())

                reward = torch.tensor([reward], device=device)
                nextstate = torch.from_numpy(nextstate.reshape(1, -1)).to(device, dtype=torch.float32)

                self.replay_memory.add(Transition(state, action, reward, nextstate, done))

                state = nextstate

                if len(self.replay_memory) >= self.batch_size:
                    self.optimize()

                if done:
                    print("{}".format(t))
                    break

            if i_episode % self.target_update:
                self.target_net.load_state_dict(self.policy_net.state_dict())


if __name__ == '__main__':
    env = gym.make('CartPole-v0').unwrapped

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n

    q_net = DQN(s_dim, a_dim).to(device)
    q_net_target = DQN(s_dim, a_dim).to(device)
    q_net_target.load_state_dict(q_net.state_dict())
    q_net_target.eval()

    memory = ReplayMemory(capacity=1000)
    agent = Agent(env=env,
                  q_net=q_net, q_net_target=q_net_target, lr=1e-3,
                  replay_memory=memory, batch_size=100,
                  gamma=0.99,
                  eps_start=0.9, eps_end=0.05, eps_decay=0.9999,
                  target_update=10)

    agent.train(5000)

