import gym
import numpy as np
from ddpg_tf2 import Agent
from utils import plot_learning_curve
import time




if __name__ == '__main__':
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(f"----------------- Training started at {current_time}. -------------------")

    env = gym.make('BipedalWalker-v3')
    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])
    n_games = 5000

    figure_file = 'walker.png'

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
        print(f"Noise: {agent.noise}")
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation, evaluate)
            
            if evaluate == False:
                agent.noise = agent.starting_noise * np.exp(-i/2500)
            else:
                agent.noise = 0
            
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

t2 = time.localtime()
current_time2 = time.strftime("%H:%M:%S", t)
t_delta = time.mktime(t2)-time.mktime(t)
print(f"----------------- Training ended at {current_time}. Duration was {t_delta/60:.2f} minutes.-----------------")
