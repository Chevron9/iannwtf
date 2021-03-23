import gym
import tensorflow as tf
import numpy as np
import time

# modules
from agent import Agent
from plot import plot_learning_curve


if __name__ == '__main__':
    #for the case you just want to load a previous model
    load_checkpoint = False

    #Housekeeping variables
    last_score = last_avg_score = 0
    avg_delta = avg_steps = []

    t = t_start = time.localtime()
    current_time = time.strftime("%Y-%m-%d-%H:%M:%S", t)
    print(f"\n----------------- Training started at {current_time}. -------------------\ncheckpoint: {load_checkpoint}")

    #initialize the environment for the agent and initialize the agent
    #tf.debugging.set_log_device_placement(True)
    env = gym.make('BipedalWalker-v3')
    #env = gym.make('BipedalWalkerHardcore-v3')
    noise = 0.4
    # NEW batch 128
    agent = Agent(alpha=0.00005, beta=0.0005, input_dims=[24], tau=0.001, env=env,
                  batch_size=128, dense1=512, dense2=512, n_actions=4, noise = noise)


    episodes = 250 #250

    #where the final plot is saved
    figure_file = f'plots/walker{current_time.replace(":","_")}.png'

    #set bestscore to minimum
    best_score = env.reward_range[0]
    score_history = []


    #initializes the model with one random sample batch if model are loaded
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

    #main learning loop
    try:
        for i in range(episodes):
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
                if not load_checkpoint:
                    agent.learn()

                #saves previous observation
                observation = observation_

            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            #saves the model if the average score is better than the best previous
            if avg_score > best_score:
                best_score = avg_score
                if not load_checkpoint:
                    agent.save_models()
                    last_save = current_episode

            #calculating and giving some info on training progress
            t_new = time.localtime()
            current_time = time.strftime("%H:%M:%S", t_new)
            t_delta = time.mktime(t_new)-time.mktime(t)
            t = t_new
            avg_delta.append(t_delta)
            avg_delta_mean = np.mean(avg_delta)
            avg_delta_std = np.var(avg_delta)

            ETA_avg = (episodes-i)*avg_delta_mean
            ETA_min = (episodes-i)*max((avg_delta_mean-avg_delta_std),min(avg_delta))
            ETA_max = (episodes-i)*(avg_delta_mean+avg_delta_std)

            avg_steps.append(steps) #unused for now
            per_step = t_delta/steps
            steps_per_score = score/steps

            print(f"{current_time} \n"
            f'Episode: **{i}**/{episodes}, Score: {score:.0f} (Δ{score-last_score:5.1f}) \n'
            f'Average score: {avg_score:.0f} (Δ{avg_score-last_avg_score:5.1f})\n'
            f'Episode time: {t_delta:.1f}s, average: {avg_delta_mean:.1f}s (±{avg_delta_std:.2f}),', 
            f'ETA: {ETA_avg/60:.0f}m ({ETA_min/60:.0f}m-{ETA_max/60:.0f}m) \n'
            f'Steps: {steps}. Time per step: {per_step:.1e}s. Reward per step: {steps_per_score:.2f}. \n' 
            f'It has been {i - last_save} episode(s) since the model was last saved, with a score of {best_score:.0f} (Δ{avg_score-best_score:.0f}). \n')

            last_score = score
            last_avg_score = avg_score
            
    except KeyboardInterrupt:
        episodes = current_episode
        print("Shutting down training.")
    
    #plots the whole score history
    if not load_checkpoint:
        x = [i+1 for i in range(episodes)]
        plot_learning_curve(x, score_history, figure_file)

t2 = time.localtime()
current_time = time.strftime("%Y-%m-%d-%H:%M:%S", t2)
t_delta = time.mktime(t2)-time.mktime(t_start)
print(f"\n----------------- Training ended at {current_time}. Duration was {t_delta/60:.2f} minutes.-----------------")
