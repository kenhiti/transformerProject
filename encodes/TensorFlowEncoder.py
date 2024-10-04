import tensorflow as tf
from keras import layers as layers

from transformer_architecture.encodes.TensorflowEncoderLayer import TensorflowEncoderLayer
from transformer_architecture.encodes.TensorflowPositionalEncoding import TensorflowPositionalEncoding


class TensorFlowEncoder(layers.Layer):

    def __init__(self,
                 nb_layers,
                 FFN_units,
                 nb_proj,
                 dropout_rate,
                 vocab_size,
                 d_model,
                 name="encoder"):
        super(TensorFlowEncoder, self).__init__(name=name)
        self.nb_layers = nb_layers
        self.d_model = d_model

        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = TensorflowPositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.enc_layers = [TensorflowEncoderLayer(FFN_units, nb_proj, dropout_rate) for _ in range(nb_layers)]

    def call(self, inputs, mask, training):
        outputs = self.embedding(inputs)
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs, training)

        for i in range(self.nb_layers):
            outputs = self.enc_layers[i](outputs, mask, training)

        return outputs
