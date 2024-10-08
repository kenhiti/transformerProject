from keras import layers as layers

from transformer_architecture.attention.TensorflowMultiHeadAttention import TensorflowMultiHeadAttention

class TensorflowEncoderLayer(layers.Layer):
    def __init__(self, FFN_units, nb_proj, dropout_rate):
        super(TensorflowEncoderLayer, self).__init__()
        self.FFN_units = FFN_units
        self.nb_proj = nb_proj
        self.dropout_rate = dropout_rate

    '''
        Creating all layers of the Encoder.
            -Multi-Head Attention Layer
            -Dropout Layer
            -Add & Normalization Layer
            -Feed Forward Network Layer
            -Add & Normalization Layer
    '''
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
        """
            Insert into Multi-Head Attention Layer the V,K and Q. In Encoder, mask variable is None.
        """
        attention = self.multi_head_attention(inputs, inputs, inputs, mask)
        '''
            Dropout is activated only if training mode. In the training model, link this with the Dropout Layer.
            Otherwise, link this with the Add & Normalization Layer
        '''
        attention = self.dropout_1(attention, training=training)
        #The First Add & Normalization Layer receive the summed of the Positional Encoding and Multi-Head Attention.
        attention = self.norm_1(attention + inputs)

        '''
            Dense Layer 1 represents the Feed Forward Networks and receive Add & Normalization Layer
            and Multi-Head Attention Layer. Receives Dropout Layer if training mode on.
        '''
        outputs = self.dense_1(attention)
        '''
            Dense Layer 2 represents the outputs and receive Dense Layer 1, Feed Forward Networks Layer and Add & Normalization Layer. 
            Receives Dropout Layer if training mode on.
        '''
        outputs = self.dense_2(outputs)
        outputs = self.dropout_2(outputs, training=training)
        '''
            The second Add & Normalization Layer receives the input of the Feed Forward Networks Layer and 
            output of the Dense Layer 1(Feed Forward Networks Layer output or Dropout Layer if training mode on).
        '''
        outputs = self.norm_2(outputs + attention)

        return outputs