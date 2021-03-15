import gym
import tensorflow as tf
import numpy as np
from ddpg_tf2 import Agent
from utils import plot_learning_curve

if __name__ == '__main__':


    tf.debugging.set_log_device_placement(True)
    env = gym.make('BipedalWalker-v3')
    agent = Agent(alpha=0.00005, beta=0.0005, input_dims=[24], tau=0.001, env=env,
                  batch_size=64, fc1=400, fc2=300, n_actions=4)
    #agent = Agent(input_dims=env.observation_space.shape, env=env,
    #        n_actions=env.action_space.shape[0])
    print(env.observation_space.shape)
    n_games = 6000 #250

    figure_file = 'plots/walker.png'

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

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

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        agent.noise = 0.3 * np.exp(-i/1300)
        while not done:
            action = agent.choose_action(observation, evaluate)

            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
