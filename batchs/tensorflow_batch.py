import logging

import tensorflow as tf

batch_size = 64
buffer_size = 20000

def create_tensor_slices_load_to_cache_and_do_prefetch(input, output):
    logging.info("Start creating tensor slices...")
    dataset = tf.data.Dataset.from_tensor_slices((input,output))
    logging.info("Start loading to cache...")
    dataset = dataset.shuffle(buffer_size).batch(batch_size)
    logging.info("Start do Prefetching...")
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)


