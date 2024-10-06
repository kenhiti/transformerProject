import tensorflow as tf
from keras import Model as model
from keras import layers as layers

from transformer_architecture.decodes.TensorFlowDecoder import TensorFlowDecoder
from transformer_architecture.encodes.TensorFlowEncoder import TensorFlowEncoder
from transformer_architecture.utils.tests.function_test import create_padding_mask, create_look_ahead_mask


class TensorFlowTransformer(model):
    def __init__(self,
                 vocab_size_enc,
                 vocab_size_dec,
                 d_model,
                 nb_layers,
                 FFN_units,
                 nb_proj,
                 dropout_rate,
                 name="transformer"):
        super(TensorFlowTransformer, self).__init__(name=name)
        self.encoder = TensorFlowEncoder(nb_layers, FFN_units, nb_proj, dropout_rate, vocab_size_enc, d_model)
        self.decoder = TensorFlowDecoder(nb_layers, FFN_units, nb_proj, dropout_rate, vocab_size_dec, d_model)
        self.last_linear = layers.Dense(units=vocab_size_dec, name='lin_output')

    def create_padding_mask(self, seq): # (batch_size, seq_length) -> (batch_size, nb_proj, seq_lenght, d_proj)
        mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, seq):
        seq_len = tf.shape(seq)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return look_ahead_mask

    def call(self, enc_inputs, dec_inputs, training):
        enc_mask = self.create_padding_mask(enc_inputs)
        dec_mask_1 = tf.maximum(self.create_padding_mask(dec_inputs), self.create_look_ahead_mask(dec_inputs))
        dec_mask_2 = self.create_padding_mask(enc_inputs)

        enc_outputs = self.encoder(enc_inputs,enc_mask, training)
        dec_outputs = self.decoder(dec_inputs, enc_outputs, dec_mask_1, dec_mask_2, training)

        return self.last_linear(dec_outputs)


