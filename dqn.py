"""
PyTorch implementation of DQN, based on https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
Should work on all gym environments with discrete action-spaces
"""
from networks import *
from agent import *
from envs import *


def main():
    env = gym.make('CartPoleLimit-v0').unwrapped
    # env = gym.make('MountainCar-v0').unwrapped
    # env = gym.make('Acrobot-v1').unwrapped

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n

    q_net = DQN(s_dim, a_dim).to(device)
    q_net_target = DQN(s_dim, a_dim).to(device)
    q_net_target.load_state_dict(q_net.state_dict())
    q_net_target.eval()

    memory = ReplayMemory(capacity=500)
    agent = Agent(env=env,
                  q_net=q_net, q_net_target=q_net_target, lr=1e-4,
                  replay_memory=memory, batch_size=200,
                  gamma=0.99,
                  eps_schedule=EpsilonScheduleExponential(eps_start=0.9, eps_min=0.05, eps_decay=0.999),
                  # eps_schedule=EpsilonScheduleLinear(eps_start=0.9, eps_min=0.05, eps_decay=0.002),
                  target_update_n_steps=20,
                  evaluate_every_n_steps=100)

    agent.train(n_episodes=1000)


if __name__ == '__main__':
    main()
