import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer


x = tf.zeros((32,400,400,1))

conv_layer_1= tf.keras.layers.Conv2D(filters= 25, kernel_size= 3, strides=(1, 1), padding='valid',  activation=tf.keras.activations.relu)

x = conv_layer_1(x)
print(x)