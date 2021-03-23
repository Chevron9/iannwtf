import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from replaybuffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork

# the agent class where the all the important parameters and systems of the
# model are managed

class Agent:
    def __init__(self, input_dims, alpha=0.001, beta=0.002, env=None,
            gamma=0.99, n_actions=4, max_size=1000000, tau=0.001,
            dense1=512, dense2=512, batch_size=64, noise=0.3):

        #initializing network-parameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise

        #retrieves the maximum and minimum of the actionvalues
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        #initializes the Replaybuffer which stores what the agents does
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        
        # initializing the Networks with given parameters
        # target_actor and target_critic are just initialized as the actor and
        # critic networks
        self.actor = ActorNetwork(n_actions=n_actions, name='actor', dense1 = dense1, dense2 = dense2)
        self.target_actor = ActorNetwork(n_actions=n_actions, name='target_actor',  dense1 = dense1, dense2 = dense2)

        self.critic = CriticNetwork(name='critic',  dense1 = dense1, dense2 = dense2)
        self.target_critic = CriticNetwork(name='target_critic', dense1 = dense1, dense2 = dense2)


        #compile the networks with learningrates
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1) #Hard copy, since this is the initialization

    #updates the target networks
    #soft copies the target and actor network dependent on tau
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    #stores the state, action, reward transition
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    #saves models in files
    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    #loads models from files
    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    #choose_action with help of the actor network, adds noise if it for training
    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluate:

            actions += tf.random.normal(shape=[self.n_actions],
                    mean=0.0, stddev=self.noise)

        #makes sure that action boundaries are met
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0]

    #learn function of the networks
    def learn(self):
        #starts to learn when there are enough samples to fill a batch
        if self.memory.current_position < self.batch_size:
            return

        #gets batch form memory
        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        #convert np arrays to tensors to feed them to the networks
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        #update critic network
        with tf.GradientTape() as tape:

            #target actor decides which action to take
            target_actions = self.target_actor(states_)
            #target critic evaluates the value of the actions in the given states
            critic_value_ = tf.squeeze(self.target_critic(
                                states_, target_actions), 1)

            #critic network evaluate the actual states and actions the model took
            critic_value = tf.squeeze(self.critic(states, actions), 1)

            #target gives says what value of the action in a certain state should
            #be like
            target = rewards + self.gamma*critic_value_*(1-done)

            #takes the MSE of the target and the actual critic value as the loss
            critic_loss = keras.losses.MSE(target, critic_value)


        #gets the gradients of the loss in respect to the parameters of the network
        critic_network_gradient = tape.gradient(critic_loss,
                                            self.critic.trainable_variables)

        #aplies the gradients to the critic network
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))

        #update the actor network
        with tf.GradientTape() as tape:
            #gets the policy of the actor in a state
            action_policy = self.actor(states)

            #loss of the actor is the negative value of the critic because we
            #want to maximize the value but gradient decent minimizes
            actor_loss = -self.critic(states, action_policy)

            #the loss is a average of all the losses
            actor_loss = tf.math.reduce_mean(actor_loss)

        #gradients of the loss in respect to the parameters of the critic network
        actor_network_gradient = tape.gradient(actor_loss,
                                    self.actor.trainable_variables)


        #optimizing the network the gradients
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()
