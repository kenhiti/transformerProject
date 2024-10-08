import keras as keras
import numpy as np
import tensorflow as tf

'''
    POSITIONAL ENCODING
    This to make use of the
    order of the sequence, we must inject some information about the relative or absolute position of the
    tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension dmodel
    as the embeddings, so that the two can be summed.
    
    Formula:
        PE(pos,2i)   = sin(pos/10000 ** 2i/dmodel)
        PE(pos,2i+1) = cos(pos/10000 ** 2i/dmodel)
'''
class TensorflowPositionalEncoding(keras.Layer):
    def __init__(self):
        super(TensorflowPositionalEncoding, self).__init__()

    '''
        pos = position, i = dimension, and d_model = model dimension
    '''
    def get_angles(self, pos, i, d_model):
        angles = 1 / np.power(10000., (2 * (i // 2)) / np.float32(d_model))
        return pos * angles  # (seq_length, d_model)
    get_angles.__doc__ = "A function that calculates the parameter to calculate SINE e COSINEE"

    '''
       This function is required Embeddings Object Parameter.
       The SINE is calculates with all even numbers
       The COSINE is calculates with all odd numbers 
    '''
    def call(self, inputs):
        #The index -2 is imputed because it's the position of the value of the numbers of lines
        seq_length = inputs.shape.aslist()[-2]
        #The index -1 is imputed because it's the position of the value of the numbers of columns.
        d_model = inputs.shape.as_list()[-1]

        angles = self.get_angles(np.arange(seq_length)[:, np.newaxis],
                                np.arange(d_model)[np.newaxis, :], d_model)

        angles[:,0::2] = np.sin(angles[:,0::2])
        angles[:,1::2] = np.cos(angles[:,1::2])
        pos_encoding = angles[np.newaxis, ...]

        #Returns a summed of input embedding and positional encoding
        return inputs + tf.cast(pos_encoding, tf.float32)
    call.__doc__ = keras.Layer.call.__doc__