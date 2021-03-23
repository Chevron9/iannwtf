import gym
import numpy as np
from agent import Agent


#script to render the result

#environment the agent is in
env = gym.make('BipedalWalker-v3')

observation = env.reset()

agent = Agent(input_dims=env.observation_space.shape, env=env,
        n_actions=env.action_space.shape[0])

#timesteps that are rendered
timesteps = 300


#load the model
n_steps = 0
while n_steps <= agent.batch_size:
    observation = env.reset()
    action = env.action_space.sample()
    observation_, reward, done, info = env.step(action)
    agent.remember(observation, action, reward, observation_, done)
    n_steps += 1
agent.learn()
agent.load_models()




#renders the environment with the actions of the saved model
counter = 0
score = 0
for _ in range(timesteps):
    env.render()
    #agent chooses action without noise
    action = agent.choose_action(observation, evaluate = True)
    observation, reward, done, info = env.step(action)
    counter += 1
    score += reward

    if done:
        observation = env.reset()
        print(f"{counter} steps taken. Score: {score}")
print(f"{counter} steps taken. Score: {score}")

env.close()
