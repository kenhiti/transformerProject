import logging

import tensorflow as tf

'''
https://www.tensorflow.org/guide/data_performance

Best practice summary
Here is a summary of the best practices for designing performant TensorFlow input pipelines:

V - Use the prefetch transformation to overlap the work of a producer and consumer
X - Parallelize the data reading transformation using the interleave transformation
X - Parallelize the map transformation by setting the num_parallel_calls argument
V - Use the cache transformation to cache data in memory during the first epoch
X - Vectorize user-defined functions passed in to the map transformation
V - Shuffle transformations

IN PROGRESS
'''

batch_size = 64
buffer_size = 20000

def create_tensor_slices_load_to_cache_and_do_prefetch(input, output):
    logging.info("Start creating tensor slices...")
    dataset = tf.data.Dataset.from_tensor_slices((input,output))
    logging.info("Start loading to cache...")
    dataset = dataset.shuffle(buffer_size).batch(batch_size)
    logging.info("Start do Prefetching...")
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)


