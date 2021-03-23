import gym
import tensorflow as tf
import numpy as np
import time

# modules
from agent import Agent
from plot import plot_learning_curve

last_score = last_avg = 0


if __name__ == '__main__':
    t = t_start = time.localtime()
    current_time = time.strftime("%Y-%m-%d-%H:%M:%S", t)
    print(f"----------------- Training started at {current_time}. -------------------")

    #initialize the environment for the agent and initialize the agent
    #tf.debugging.set_log_device_placement(True)
    env = gym.make('BipedalWalker-v3')
    noise = 0.4
    agent = Agent(alpha=0.00005, beta=0.0005, input_dims=[24], tau=0.001, env=env,
                  batch_size=64, dense1=512, dense2=512, n_actions=4, noise = noise)


    episodes = 6000 #250

    #where the final plot is saved
    figure_file = f'plots/walker{current_time.replace(":","_")}.png'

    #set bestscore to minimum
    best_score = env.reward_range[0]
    score_history = []

    #for the case you just want to load a previous model
    load_checkpoint = False

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
        evaluate = True
    else:
        evaluate = False

    #main learning loop
    for i in range(episodes):
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
        while not done:
            action = agent.choose_action(observation, evaluate)

            observation_, reward, done, info = env.step(action)
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

        t_new = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        t_delta = time.mktime(t_new)-time.mktime(t)
        t = t_new

        print(f"{current_time} \n"
        f'Episode: {i}, Score: {score:.0f} ({score-last_score:.1f})\n'
        f'Average score: {avg_score:.0f} ({avg_score-last_avg:.1f})\n'
        f'Episode time: {t_delta:.0f}s \n')

        last_score = score
        last_avg = avg_score
    
    #plots the whole score history
    if not load_checkpoint:
        x = [i+1 for i in range(episodes)]
        plot_learning_curve(x, score_history, figure_file)

t2 = time.localtime()
current_time = time.strftime("%Y-%m-%d-%H:%M:%S", t2)
t_delta = time.mktime(t2)-time.mktime(t_start)
print(f"----------------- Training ended at {current_time}. Duration was {t_delta/60:.2f} minutes.-----------------")