import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from .replaybuffer import ReplayBuffer
from .networks import ActorNetwork, CriticNetwork

# the agent class where the all the important parameters and systems of the
# model are managed

class Agent:
    def __init__(self, input_dims, alpha=0.001, beta=0.002, env=None,
            gamma=0.99, n_actions=4, max_size=1000000, tau=0.001,
            dense1=512, dense2=512, batch_size=64, noise=0.3, module_dir = ""):

        #initializing network-parameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        
        self.delay = 0 # this governs if we wanna update the actor this step or not
        self.delay_threshold = 2 #we want to update the actor every x steps

        #retrieves the maximum and minimum of the actionvalues
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        #initializes the Replaybuffer which stores what the agents does
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        
        # initializing the Networks with given parameters
        # target_actor and target_critic are just initialized as the actor and
        # critic networks

        chkpt_dir = module_dir+"/tmp"
        self.actor = ActorNetwork(n_actions=n_actions, name='actor', dense1 = dense1, dense2 = dense2, chkpt_dir=chkpt_dir)
        self.target_actor = ActorNetwork(n_actions=n_actions, name='target_actor',  dense1 = dense1, dense2 = dense2, chkpt_dir=chkpt_dir)



        # The critics might be better handled in a list, but this might slow down tensorflow performance
        self.critic1 = CriticNetwork(name='critic1',  dense1 = dense1, dense2 = dense2, chkpt_dir=chkpt_dir)
        self.target_critic1 = CriticNetwork(name='target_critic1', dense1 = dense1, dense2 = dense2, chkpt_dir=chkpt_dir)

        self.critic2 = CriticNetwork(name='critic2',  dense1 = dense1, dense2 = dense2, chkpt_dir=chkpt_dir)
        self.target_critic2 = CriticNetwork(name='target_critic2', dense1 = dense1, dense2 = dense2, chkpt_dir=chkpt_dir)


        #compile the networks with learning rates
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        
        self.critic1.compile(optimizer=Adam(learning_rate=beta))
        self.target_critic1.compile(optimizer=Adam(learning_rate=beta))

        self.critic2.compile(optimizer=Adam(learning_rate=beta))
        self.target_critic2.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1) #Hard copy, since this is the initialization

    #updates the target networks
    #soft copies the target and actor network dependent on tau
    def update_network_parameters(self, tau=None, delay = False):
        if tau is None:
            tau = self.tau

        if delay:
            weights = []
            targets = self.target_actor.weights
            for i, weight in enumerate(self.actor.weights):
                weights.append(weight * tau + targets[i]*(1-tau))
            self.target_actor.set_weights(weights)


        # critics copying to target critics
        weights = []
        targets = self.target_critic1.weights
        for i, weight in enumerate(self.critic1.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic1.set_weights(weights)

        weights = []
        targets = self.target_critic2.weights
        for i, weight in enumerate(self.critic2.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic2.set_weights(weights)

    #stores the state, action, reward transition
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    #saves models in files
    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic1.save_weights(self.critic1.checkpoint_file)
        self.target_critic1.save_weights(self.target_critic1.checkpoint_file)
        self.critic2.save_weights(self.critic2.checkpoint_file)
        self.target_critic2.save_weights(self.target_critic2.checkpoint_file)

    #loads models from files
    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic1.load_weights(self.critic1.checkpoint_file)
        self.target_critic1.load_weights(self.target_critic1.checkpoint_file)
        self.critic2.load_weights(self.critic2.checkpoint_file)
        self.target_critic2.load_weights(self.target_critic2.checkpoint_file)

    #choose_action with help of the actor network, adds noise if it's for training
    @tf.function
    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        
        # inject eploration noise
        if not evaluate:

            actions += tf.random.normal(shape=[self.n_actions],
                    mean=0.0, stddev=self.noise)

        #makes sure that action boundaries are met
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0]

    #learn function of the networks
    def learn(self):
        # control problem: How to solve this problem? (or: picking policy) -> actor
        # leads to
        # prediction problem: Are my actiosn actually getting me closer to accomplishing my goal? (or: value function of policy) -> critic

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
        with tf.GradientTape(persistent=True) as tape:

            #call target actor to simulate which action to take
            target_actions = self.target_actor(states_)

            # TD3: Add noise regularization to policy
            # This is achieved by adding "clipped noise"
            # Noise is clipped to ensure the noisy actions aren't too far from policy.
            target_actions = target_actions + tf.clip_by_value(np.random.normal(scale=0.2), -0.5, 0.5)

            # Clip again to make sure we aren't violating action space
            target_actions = tf.clip_by_value(target_actions, self.min_action, self.max_action)


            #target critic evaluates the value of the actions in the given states
            critic1_value_ = tf.squeeze(self.target_critic1(states_, target_actions), 1)

            #critic network evaluate the actual states and actions the model took
            critic1_value = tf.squeeze(self.critic1(states, actions), 1)


            critic2_value_ = tf.squeeze(self.target_critic2(states_, target_actions), 1)
            critic2_value = tf.squeeze(self.critic2(states, actions), 1)

            #target says what value of the action in a certain state should
            #be like
            target = rewards + self.gamma*tf.math.minimum(critic1_value_,critic2_value_)*(1-done)

            #takes the MSE of the target and the actual critic value as the loss
            critic1_loss = keras.losses.MSE(target, critic1_value)
            critic2_loss = keras.losses.MSE(target, critic2_value)


        #gets the gradients of the loss in respect to the parameters of the network
        critic1_network_gradient = tape.gradient(critic1_loss,
                                            self.critic1.trainable_variables)

        critic2_network_gradient = tape.gradient(critic2_loss,
                                            self.critic2.trainable_variables)

        #aplies the gradients to the critic network
        self.critic1.optimizer.apply_gradients(zip(
            critic1_network_gradient, self.critic1.trainable_variables))

        self.critic2.optimizer.apply_gradients(zip(
            critic2_network_gradient, self.critic2.trainable_variables))

        # The if/else makes sure we only update the actor every second step.
        if self.delay < (self.delay_threshold-1):
            self.delay += 1
            delayBool = True
        elif self.delay == (self.delay_threshold-1):
            delayBool = False
            #update the actor network
            with tf.GradientTape() as tape:
                #gets the policy of the actor in a state
                action_policy = self.actor(states)

                #loss of the actor is the negative value of the critic because we
                #want to maximize the value but gradient descent minimizes
                # We are using Critic1 Loss here, as this is how it's done in the original paper.
                
                # It would be interesting to plot how c1 and c2 loss differ over time.

                actor_loss = -(self.critic1(states, action_policy))

                #the loss is a average of all the losses
                actor_loss = tf.math.reduce_mean(actor_loss)

            #gradients of the loss in respect to the parameters of the actor network
            actor_network_gradient = tape.gradient(actor_loss,
                                        self.actor.trainable_variables)


            #optimizing the network gradients
            self.actor.optimizer.apply_gradients(zip(
                actor_network_gradient, self.actor.trainable_variables))

            self.delay = 0
        else:
            raise Exception(f"Delay has been set incorrectly. Delay is {self.delay}, Threshold is {self.threshold}.")

        self.update_network_parameters(delay=delayBool)
