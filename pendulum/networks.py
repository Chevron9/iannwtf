import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


#critic network that takes the state and the action and puts out the value of the
#the action in the given state
class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512,
            name='critic', chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.relu = tf.keras.activations.relu

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                    self.model_name+'_ddpg.h5')


        f1 = 1. / np.sqrt(self.fc1_dims)
        self.fc1 = Dense(self.fc1_dims, activation= None, kernel_initializer = tf.keras.initializers.RandomUniform(-f1, f1),
        bias_initializer = tf.keras.initializers.RandomUniform(-f1, f1))
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        f2 = 1. / np.sqrt(self.fc2_dims)
        self.fc2 = Dense(self.fc2_dims, activation= None, kernel_initializer = tf.keras.initializers.RandomUniform(-f2, f2),
        bias_initializer = tf.keras.initializers.RandomUniform(-f2, f2))
        self.batchnorm2 = tf.keras.layers.BatchNormalization()

        f3 = 0.003
        self.q = Dense(1, activation=None, kernel_initializer = tf.keras.initializers.RandomUniform(-f3, f3) ,
        bias_initializer = tf.keras.initializers.RandomUniform(-f3, f3),
        kernel_regularizer=tf.keras.regularizers.l2(0.01))

    @tf.function
    def call(self, state, action, training = True):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        #action_value = self.batchnorm1(action_value, training)
        action_value = self.relu(action_value)

        action_value = self.fc2(action_value)
        #action_value = self.batchnorm2(action_value, training)

        action_value = self.relu(action_value)

        q = self.q(action_value)

        return q
#critic network that takes the state and puts out the a probability
#distribution over all possible actions
class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, n_actions=4, name='actor',
            chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.relu = tf.keras.activations.relu

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                    self.model_name+'_ddpg.h5')

        #first dense layer
        f1 = 1. / np.sqrt(self.fc1_dims)
        self.fc1 = Dense(self.fc1_dims, activation= None, kernel_initializer = tf.keras.initializers.RandomUniform(-f1, f1) , bias_initializer
         = tf.keras.initializers.RandomUniform(-f1, f1))
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        #second dense layer
        f2 = 1. / np.sqrt(self.fc2_dims)
        self.fc2 = Dense(self.fc2_dims, activation= None, kernel_initializer = tf.keras.initializers.RandomUniform(-f2, f2) , bias_initializer
         = tf.keras.initializers.RandomUniform(-f2, f2))
        self.batchnorm2 = tf.keras.layers.BatchNormalization()


        #output
        f3 = 0.003
        self.mu = Dense(self.n_actions, activation='tanh', kernel_initializer = tf.keras.initializers.RandomUniform(-f3, f3) , bias_initializer
         = tf.keras.initializers.RandomUniform(-f3, f3))
    @tf.function
    def call(self, state, training = True):
        prob = self.fc1(state)
        #prob = self.batchnorm1(prob, training)
        prob = self.relu(prob)
        prob = self.fc2(prob)
        #prob = self.batchnorm2(prob, training)
        prob = self.relu(prob)

        mu = self.mu(prob)

        return mu
