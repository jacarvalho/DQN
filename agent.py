import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from itertools import count

from utils import *


class Agent:

    def __init__(self, env=None,
                 q_net=None, q_net_target=None, lr=1e-3,
                 replay_memory=None, batch_size=None,
                 gamma=None,
                 eps_schedule=None,
                 target_update_n_steps=10,
                 evaluate_every_n_steps=100):
        self.env = env
        self.q_net = q_net
        self.q_net_target = q_net_target
        self.replay_memory = replay_memory
        self.batch_size = batch_size
        self.a_dim = q_net.a_dim
        self.gamma = gamma
        self.eps_schedule = eps_schedule
        self.target_update_n_steps = target_update_n_steps
        self.evaluate_every_n_steps = evaluate_every_n_steps
        self._n_steps = 0

        # Loss and gradient descent optimizer
        self.loss_criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def select_action(self, s):
        eps = np.random.rand(1)
        if eps < self.eps_schedule():
            action = torch.tensor([[np.random.choice(self.a_dim)]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                action = self.q_net(s).argmax(1).view(1, 1)
        self._n_steps += 1
        return action

    def optimize(self):
        transitions = self.replay_memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        not_done_mask = torch.tensor(tuple(map(lambda d: not d, batch.done)), device=device, dtype=torch.bool)

        non_final_next_states = torch.cat([nextstate
                                           for nextstate, done in zip(batch.nextstate, batch.done) if not done])

        state_batch = torch.cat(batch.state, dim=0)
        action_batch = torch.cat(batch.action, dim=0)
        reward_batch = torch.cat(batch.reward, dim=0)

        state_action_values = self.q_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[not_done_mask] = self.q_net_target(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = reward_batch + self.gamma * next_state_values

        # Perform a batch gradient descent
        loss = self.loss_criterion(expected_state_action_values.unsqueeze(1), state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data.item()

    def train(self, n_episodes=1000):
        for i_episode in range(n_episodes):
            print("ep: {:4}/{} -> ".format(i_episode+1, n_episodes), end=' ')
            state = self.env.reset()
            state = torch.from_numpy(state.reshape(1, -1)).to(device, dtype=torch.float32)
            ep_avg_bellman_error = 0.
            ret = 0.
            for t in count():
                action = self.select_action(state)
                nextstate, reward, done, _ = self.env.step(action.item())
                ret += reward
                reward = torch.tensor([reward], device=device)
                nextstate = torch.from_numpy(nextstate.reshape(1, -1)).to(device, dtype=torch.float32)

                self.replay_memory.add(Transition(state, action, reward, nextstate, done))

                state = nextstate

                if len(self.replay_memory) >= self.batch_size:
                    ep_avg_bellman_error += self.optimize()

                if done or t >= self.env.spec.max_episode_steps - 1:
                    print("ret: {}, avg bellman error: {:.6f}".format(ret, ep_avg_bellman_error/t))
                    break

            if (i_episode % self.target_update_n_steps) == 0:
                self.q_net_target.load_state_dict(self.q_net.state_dict())

            if (i_episode % self.evaluate_every_n_steps) == 0:
                self.evaluate(render=True)
        self.env.close()

    def evaluate(self, render=False):
        state = self.env.reset()
        state = torch.from_numpy(state.reshape(1, -1)).to(device, dtype=torch.float32)
        ret = 0.
        for t in count():
            if render:
                self.env.render()
            action = self.select_action(state)
            nextstate, reward, done, _ = self.env.step(action.item())
            nextstate = torch.from_numpy(nextstate.reshape(1, -1)).to(device, dtype=torch.float32)
            state = nextstate
            ret += reward

            if done or t >= self.env.spec.max_episode_steps - 1:
                break
        print("ret: {}".format(ret))
