import tensorflow as tf

def create_padding_mask(seq): # (batch_size, seq_length) -> (batch_size, nb_proj, seq_lenght, d_proj)
  mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
  return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(seq):
  seq_len = tf.shape(seq)[1]
  look_ahed_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  return look_ahed_mask

seq = tf.cast([[837, 836, 0, 273, 8, 0, 0, 0]], tf.int32)

print(create_padding_mask(seq))

print(create_look_ahead_mask(seq))