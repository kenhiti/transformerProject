import tensorflow as tf

'''
    Attention calculus:
        attention(Q,K,V) = softmax(QK**T/sqrt(dk))*V
'''
def scaled_dot_product_attention(query, key, value, mask=None):
    #QK**T
    dot_product = tf.matmul(query, key, transpose_b=True)
    key_dimension = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_dot_product = dot_product / tf.math.sqrt(key_dimension)

    if mask is not None:
        #look ahead mask
        scaled_dot_product += (mask * -1e9) # 0.0000000001

    attention = tf.matmul(tf.nn.softmax(scaled_dot_product, axis=-1), value)
    return attention