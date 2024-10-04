import tensorflow as tf

def scaled_dot_product_attention(query, key, value, mask=None):
    product = tf.matmul(query, key, transpose_b=True)
    key_dimension = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_product = product / tf.math.sqrt(key_dimension)

    if mask is not None:
        scaled_product += (mask * -1e9) # 0.0000000001

    attention = tf.matmul(tf.nn.softmax(scaled_product, axis=-1), value)
    return attention