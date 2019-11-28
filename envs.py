import gym


# Environments
gym.envs.register(
    id='CartPoleLimit-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=300,
)
