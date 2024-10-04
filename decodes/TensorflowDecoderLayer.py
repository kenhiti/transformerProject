import tensorflow as tf
from keras import layers as layers

from transformer_architecture.TransformerArchitecture import outputs
from transformer_architecture.attention.TensorflowMultiHeadAttention import TensorflowMultiHeadAttention


class TensorflowDecoderLayer(layers.Layer):
    def __init__(self, FFN_units, nb_proj, dropout_rate):
        super(TensorflowDecoderLayer, self).__init__()
        self.FFN_units = FFN_units
        self.nb_proj = nb_proj
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.d_model = input_shape[-1]

        self.multi_head_attention_1 = TensorflowMultiHeadAttention(self.nb_proj)
        self.dropout_1 = layers.Dropout(rate=self.dropout_rate)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        self.multi_head_attention_2 = TensorflowMultiHeadAttention(self.nb_proj)
        self.dropout_2 = layers.Dropout(rate=self.dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dense_1 = layers.Dense(units=self.FFN_units, activation='relu')
        self.dense_2 = layers.Dense(units=self.FFN_units, activation='relu')
        self.dropout_3 = layers.Dropout(rate=self.dropout_rate)

        self.norm3 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs,encode_output, mask1, mask2, training):
        attention = self.multi_head_attention_1(inputs, inputs, inputs, mask1)
        attention = self.dropout_1(attention, training)
        attention = self.norm1(attention + inputs)

        attention_2 = self.multi_head_attention_2(attention, encode_output, encode_output, mask2)
        attention_2 = self.dropout_2(attention_2, training)
        attention = self.norm2(attention_2 + attention)

        outputs = self.dense_1(attention_2)
        outputs = self.dense_2(outputs)
        outputs = self.dropout_3(outputs, training)
        outputs = self.norm3(outputs + attention_2)
