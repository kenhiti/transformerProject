import tensorflow as tf
from keras import layers as layers

from transformer_architecture.decodes.TensorflowDecoderLayer import TensorflowDecoderLayer
from transformer_architecture.encodes.TensorflowPositionalEncoding import TensorflowPositionalEncoding


class TensorflowDecoder(layers.Layer):

    def __init__(self,
                 nb_layers,
                 FFN_units,
                 nb_proj,
                 dropout_rate,
                 vocab_size,
                 d_model,
                 name="decoder"):
        super(TensorflowDecoder, self).__init__(name=name)
        self.d_model = d_model
        self.nb_layers = nb_layers
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = TensorflowPositionalEncoding()
        self.dropout = layers.Dropout(dropout_rate)
        self.dec_layers = [TensorflowDecoderLayer(FFN_units,nb_proj, dropout_rate) for _ in range(nb_layers)]

    def call(self, inputs, enc_outputs, mask1, mask2, training):
        outputs = self.embedding(inputs)
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs, training)

        for i in range(self.nb_layers):
            outputs = self.dec_layers[i](outputs, enc_outputs,mask1, mask2, training)

        return outputs
