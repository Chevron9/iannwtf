import sys
print(f"Python version {sys.version}")
print(sys.path)

import os 
print(f"Current working directory is {os.getcwd()}")

import gym
import tensorflow as tf
print(f"TF version: {tf.version.VERSION}")
import numpy as np
import time

# modules
from td3.agent import Agent

from utilities.plot import plot_learning_curve
from utilities.time_converter import timespan_format

module_dir = "td3_hardcore/"



# TD3 is an improvement on DDPG

# Actor network only updated every 2 steps

# 2 separate critic networks (and a target network for each of them)


# target policy smoothing
# TD3 reduces this variance by adding a small amount of random noise to the target and averaging over mini batches. 
# The range of noise is clipped in order to keep the target value close to the original action.

#TODO: Improve the loading loop, make sure the algorithm remembers what came before
# adjusting episodes
# adjusting plots
# adjusting tensorboard graphs etc
# statefulness

if __name__ == '__main__':

    #for the case you just want to load a previous model
    load_checkpoint = False

    #TODO maybe add some error handling if no checkpoint to load exists

    # enable or disable priority replay
    prioritize = False

    #Housekeeping variables
    last_score = 0
    last_avg_score = 0
    last_save = 0
    avg_delta = []
    avg_steps = []

    t = t_start = time.localtime()
    current_time = time.strftime("%Y-%m-%d-%H:%M:%S", t)
    print(f"\n----------------- Training started at {current_time}. -------------------\ncheckpoint: {load_checkpoint}")


    figure_dir = module_dir+f'plots/'
    figure_file = figure_dir+f'walker{current_time.replace(":","_")}.png'

    log_dir = module_dir+'logs/' + current_time.replace(":","_")
    
    #Tensorboard writer  
    writer = tf.summary.create_file_writer(log_dir)


    #initialize the environment for the agent and initialize the agent
    
    #tf.debugging.set_log_device_placement(True)
    # env = gym.make('BipedalWalker-v3')
    env = gym.make('BipedalWalkerHardcore-v3')


    n_actions = env.action_space.shape[0]

    noise = 0.4

    # NEW batch 128
    agent = Agent(alpha=0.00005, beta=0.0005, input_dims=env.observation_space.shape, tau=0.001, env=env,
                  batch_size=128, dense1=512, dense2=512, n_actions=n_actions, noise = noise, module_dir = module_dir, prioritize = prioritize)


    episodes = 5000 #250

    if prioritize:
        prior_beta_scale = (1-agent.priority_beta)/episodes



    #set bestscore to minimum
    best_score = env.reward_range[0]
    score_history = []


    #initializes the model with one random sample batch if model are loaded
    #you can't load an empty model for some reason
    #these are all dummy variables etc until load_models overwrites them
    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset()
            action = agent.choose_action(observation, evaluate = True)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            n_steps += 1
        agent.learn()
        agent.load_models()

    # ---------------------------------------
    # main learning loop
    # ---------------------------------------
    try:
        for i in range(episodes):
            if i == 3:
                tf.profiler.experimental.server.start(6009)
                print("profiler started")
                # launch tensorboard with "tensorboard --logdir logs"
                # capture profile
            #if i == 13:
            #    tf.profiler.experimental.stop
            #    print("profiler stopped")


            current_episode = i
            observation = env.reset()
            done = False
            score = 0
            
            #regulates the noise over the course of training as exponential
            #decay to get smaller noise at the end, the noise is the
            #standarddeviation of a normal distribution
            #(numbers from trial and error)
            agent.noise = noise * np.exp(-i/1500)

            #while the environment is running the model chooses actions, saves states,
            #rewards, actions and observations in the buffer and trains the networks
            #on them
            steps = 0
            while not done:
                action = agent.choose_action(observation)

                observation_, reward, done, info = env.step(action)
                steps += 1
                score += reward
                agent.remember(observation, action, reward, observation_, done)
                agent.learn()

                #saves previous observation
                observation = observation_

            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            #saves the model if the average score is better than the best previous
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()
                last_save = current_episode

            #calculating and giving some info on training progress
            t_new = time.localtime()
            current_time = time.strftime("%H:%M:%S", t_new)
            t_delta = time.mktime(t_new)-time.mktime(t)
            t = t_new
            
            # perhaps add a decaying factor for ETA
            avg_delta.append(t_delta)
            avg_delta_mean = np.mean(avg_delta)
            avg_delta_std = np.var(avg_delta)

            ETA_avg = (episodes-i)*avg_delta_mean
            ETA_min = (episodes-i)*max((avg_delta_mean-avg_delta_std),min(avg_delta))
            ETA_max = (episodes-i)*(avg_delta_mean+avg_delta_std)

            avg_steps.append(steps)
            per_step = t_delta/steps
            steps_per_score = score/steps

            print(f"{current_time} \n"
            f'Episode: **{i+1}**/{episodes}, Score: {score:.0f} (Δ{score-last_score:5.1f})\n'
            f'Average score: {avg_score:.1f} (Δ{avg_score-last_avg_score:5.2f})\n'
            f'Episode time: {t_delta:.1f}s, average: {avg_delta_mean:.1f}s (±{avg_delta_std:4.2f}),', 
            f'ETA: {timespan_format(ETA_avg)} ({timespan_format(ETA_min)} to {timespan_format(ETA_max)})\n'
            f'Steps: {steps}. Time per step: {per_step:.1e}s. Reward per step: {steps_per_score:.2f}.\n' 
            f'It has been {i - last_save} episode(s) since the model was last saved, with a score of {best_score:.0f} (Δ{avg_score-best_score:2.2f}).\n')

            last_score = score
            last_avg_score = avg_score
            

            with writer.as_default():
                tf.summary.scalar('Average Score', avg_score, step=i)
                tf.summary.scalar('Score', score, step=i)
                tf.summary.scalar('ETA', ETA_avg, step=i)
                tf.summary.scalar('Calculation time per step', per_step, step=i)
                tf.summary.scalar('Calculation time per episode', t_delta, step=i)
                tf.summary.scalar('Steps', steps, step=i)
                if ((i+1) % 50) == 0: #writer.flush and learning plot has a large performance impact, so only do it every 50 episodes
                    writer.flush()
                    x = [j+1 for j in range(current_episode+1)]
                    plot_learning_curve(x, score_history, figure_file)

            if prioritize:
                agent.priority_beta += prior_beta_scale
                


            
    except KeyboardInterrupt:
        episodes = current_episode
        print("Manually shutting down training.")
    
    #plots the whole score history
    x = [i+1 for i in range(episodes)]
    plot_learning_curve(x, score_history, figure_file)

t2 = time.localtime()
current_time = time.strftime("%Y-%m-%d-%H:%M:%S", t2)
t_delta = time.mktime(t2)-time.mktime(t_start)
print(f"\n----------------- Training ended at {current_time}. Duration was {t_delta/60:.2f} minutes.-----------------")
