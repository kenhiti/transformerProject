import tensorflow as tf
from keras import layers as layers

from transformer_architecture.attention.TensorflowMultiHeadAttention import TensorflowMultiHeadAttention


class TensorflowEncoderLayer(layers.Layer):
    def __init__(self, FFN_units, nb_proj, dropout_rate):
        super(TensorflowEncoderLayer, self).__init__()
        self.FFN_units = FFN_units
        self.nb_proj = nb_proj
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.d_model = input_shape[-1]

        self.multi_head_attention = TensorflowMultiHeadAttention(self.nb_proj)
        self.dropout_1 = layers.Dropout(rate=self.dropout_rate)
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)  # 0.0000001

        self.dense_1 = layers.Dense(units=self.FFN_units, activation='relu')
        self.dense_2 = layers.Dense(units=self.d_model, activation='relu')
        self.dropout_2 = layers.Dropout(rate=self.dropout_rate)

        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, mask, training):
        attention = self.multi_head_attention(inputs, inputs, inputs, mask)
        attention = self.dropout_1(attention, training=training)
        attention = self.norm_1(attention + inputs)

        outputs = self.dense_1(attention)
        outputs = self.dense_2(outputs)
        outputs = self.dropout_2(outputs, training=training)
        outputs = self.norm_2(outputs + attention)

        return outputs