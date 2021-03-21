import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

#all the networks of the model


#critic network that takes the state and the action and puts out the value of the
#the action in the given state
class CriticNetwork(keras.Model):
    def __init__(self, dense1=512, dense2=512,
            name='critic', chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        #Dimensions of the dense layers
        self.dense1 = dense1
        self.dense2 = dense2


        #says where the weights are saved
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                    self.model_name+'_ddpg.h5')

        #dense layers with kernel and bias initializer(from an other implementation)
        #and relu activation
        f1 = 1. / np.sqrt(self.dense1)
        self.dense_layer1 = Dense(self.dense1, activation= tf.keras.activations.relu,
        kernel_initializer = tf.keras.initializers.RandomUniform(-f1, f1),
        bias_initializer = tf.keras.initializers.RandomUniform(-f1, f1))


        f2 = 1. / np.sqrt(self.dense2)
        self.dense_layer2 = Dense(self.dense2, activation= tf.keras.activations.relu,
        kernel_initializer = tf.keras.initializers.RandomUniform(-f2, f2),
        bias_initializer = tf.keras.initializers.RandomUniform(-f2, f2))

        #denselayer with 1 neuron that gives the estimated q value of the
        #state-action pair
        f3 = 0.003
        self.q = Dense(1, activation=None, kernel_initializer = tf.keras.initializers.RandomUniform(-f3, f3) ,
        bias_initializer = tf.keras.initializers.RandomUniform(-f3, f3),
        kernel_regularizer=tf.keras.regularizers.l2(0.01))

    @tf.function
    def call(self, state, action, training = True):
        #feeds the network state and action pairs
        action_value = self.dense_layer1(tf.concat([state, action], axis=1))

        action_value = self.dense_layer2(action_value)

        q = self.q(action_value)

        #gives back an estimation of a q value
        return q


#critic network that takes the state and outputs a probability
#distribution over all possible actions
class ActorNetwork(keras.Model):
    def __init__(self, dense1=512, dense2=512, n_actions=4, name='actor',
            chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.dense1 = dense1
        self.dense2 = dense2
        self.n_actions = n_actions


        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                    self.model_name+'_ddpg.h5')

        #first dense layer
        f1 = 1. / np.sqrt(self.dense1)
        self.dense_layer1 = Dense(self.dense1, activation= tf.keras.activations.relu,
        kernel_initializer = tf.keras.initializers.RandomUniform(-f1, f1),
        bias_initializer = tf.keras.initializers.RandomUniform(-f1, f1))


        #second dense layer
        f2 = 1. / np.sqrt(self.dense2)
        self.dense_layer2 = Dense(self.dense2, activation= tf.keras.activations.relu,
        kernel_initializer = tf.keras.initializers.RandomUniform(-f2, f2),
        bias_initializer = tf.keras.initializers.RandomUniform(-f2, f2))


        #output layer with tanh activation to get an output vector of length actionspace
        #with values between -1 and 1 to fit to the action boundaries
        f3 = 0.003
        self.mu = Dense(self.n_actions, activation='tanh', kernel_initializer = tf.keras.initializers.RandomUniform(-f3, f3) , bias_initializer
         = tf.keras.initializers.RandomUniform(-f3, f3))

    @tf.function
    def call(self, state, training = True):

        actions = self.dense_layer1(state)
        actions = self.dense_layer2(actions)

        #gives back the actions the agent should take (deterministic policy)
        actions = self.mu(actions)


        return actions
