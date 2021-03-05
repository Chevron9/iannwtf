import gym
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import random

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
import tensorflow_probability as tfp

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

"""
BipedalWalkerHardcore-v3
https://gym.openai.com/envs/BipedalWalkerHardcore-v2/
Hardcore version with ladders, stumps, pitfalls. 
Time limit is increased due to obstacles. 
Reward is given for moving forward, total 300+ points up to the far end. 
If the robot falls, it gets -100. 
Applying motor torque costs a small amount of points, more optimal agent will get better score. 
State consists of hull angle speed, angular velocity, horizontal speed, vertical speed, 
position of joints and joints angular speed, legs contact with ground, and 10 lidar rangefinder measurements. 
There's no coordinates in the state vector.
"""
env = gym.make("BipedalWalkerHardcore-v3")
env.reset()
print(env.action_space)
print(env.observation_space)

for t in range(1000):
    env.render()
    obs, reward, done, info = env.step(env.action_space.sample()) # take a random action
    if done:
        print(f"Episode finished after {t+1} timesteps, reward at {reward}")
        env.reset()

print(f"Episode finished after {t+1} timesteps, reward at {reward}")
env.close()

#trick: start training by uniform sampling
#OU noise vs mean-0 gauss noise, scale down as time goes on