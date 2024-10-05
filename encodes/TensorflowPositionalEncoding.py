import keras as keras
import numpy as np
import tensorflow as tf

'''

'''
class TensorflowPositionalEncoding(keras.Layer):
    def __init__(self):
        super(TensorflowPositionalEncoding, self).__init__()

    def get_angles(self, pos, i, d_model):
        angles = 1 / np.power(10000., (2 * (i // 2)) / np.float32(d_model))
        return pos * angles  # (seq_lenght, d_model)

    def call(self, inputs):
       seq_length = inputs.shape.aslist()[-2]
       d_model = inputs.shape.as_list()[-1]
       angles = self.get_angles(np.arange(seq_length)[:, np.newaxis],
                                np.arange(d_model[np.newaxis, :], d_model))
       angles[:,0::2] = np.sin(angles[:,0::2])
       angles[:,1::2] = np.cos(angles[:,1::2])
       pos_encoding = angles[np.newaxis, ...]
       return inputs + tf.cast(pos_encoding, tf.float32)
