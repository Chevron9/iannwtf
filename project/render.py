import gym
import numpy as np

from td3.agent import Agent
#from ddpg.agent import Agent
#module_dir = "ddpg/"
module_dir = "td3/"

#script to render the result


#environment the agent is in
env = gym.make('BipedalWalkerHardcore-v3')
# env = gym.make('BipedalWalker-v3')


observation = env.reset()

agent = Agent(input_dims=env.observation_space.shape, env=env,
        n_actions=env.action_space.shape[0], module_dir = module_dir)

#timesteps that are rendered
timesteps = 3000


#load the model
n_steps = 0
while n_steps <= agent.batch_size:
    observation = env.reset()
    #action = env.action_space.sample()
    action = agent.choose_action(observation, evaluate = True)
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
        score = 0
print(f"{counter} steps taken. Score: {score}")

env.close()
