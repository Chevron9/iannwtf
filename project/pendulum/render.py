import gym
import numpy as np
from ddpg_tf2 import Agent


env = gym.make('BipedalWalkerHardcore-v3')
#render the result
observation = env.reset()
agent = Agent(input_dims=env.observation_space.shape, env=env,
        n_actions=env.action_space.shape[0],noise = 0)
load_checkpoint = True

if load_checkpoint:
    n_steps = 0
    while n_steps <= agent.batch_size:
        observation = env.reset()
        action = env.action_space.sample()
        observation_, reward, done, info = env.step(action)
        agent.remember(observation, action, reward, observation_, done)
        n_steps += 1
    agent.learn()
    agent.load_models()
    evaluate = True
else:
    evaluate = False


for _ in range(2000):
    env.render()
    action = agent.choose_action(observation, True) # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
env.close()
