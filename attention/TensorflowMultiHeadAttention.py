import tensorflow as tf
from keras import layers as layers

from transformer_architecture.attention.tensorflow_attention_mechanism import scaled_dot_product_attention

'''
    MULTI HEAD ATTENTION MECHANISM
    nb_proj = number of projection
'''
class TensorflowMultiHeadAttention(layers.Layer):

    def __init__(self, nb_proj):
        super(TensorflowMultiHeadAttention, self).__init__()
        self.nb_proj = nb_proj

    def build(self, input_shape):
        self.d_model = input_shape[-1]
        assert self.d_model % self.nb_proj == 0

        self.d_proj = self.d_model // self.nb_proj

        self.query_lin = layers.Dense(units=self.d_model)
        self.key_lin = layers.Dense(units=self.d_model)
        self.value_lin = layers.Dense(units=self.d_model)

        self.final_lin = layers.Dense(units=self.d_model)

    def split_proj(self, inputs, batch_size): # inputs: (batch_size, seq_length, d_model)
        shape = (batch_size, -1, self.nb_proj, self.d_proj)
        splitted_inputs = tf.reshape(inputs, shape=shape) # (batch_size, seq_length, nb_proj, d_proj)
        return tf.transpose(splitted_inputs, perm=[0,2,1,3])    # (batch_size, nb_proj, seq_length, d_proj)

    def call(self, queries, keys, values, mask):
        batch_size = tf.shape(queries)[0]

        queries = self.query_lin(queries)
        keys = self.key_lin(keys)
        values = self.value_lin(values)

        queries = self.split_proj(queries, batch_size)
        keys = self.split_proj(keys, batch_size)
        values = self.split_proj(values, batch_size)

        attention = scaled_dot_product_attention(queries, keys, values, mask)
        attention = tf.transpose(attention, perm=[0,2,1,3])

        concat_attention = tf.reshape(attention, shape=(batch_size, -1, self.nb_proj))

        outputs = self.final_lin(concat_attention)
        return outputs

