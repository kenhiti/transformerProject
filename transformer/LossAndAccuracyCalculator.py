import tensorflow as tf
from keras import metrics as metrics
from keras import losses as losses

def create_loss_object():
    return losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(loss_object, target, prediction):
    mask = tf.math.logical_not(tf.math.equal(target, 0))
    loss_ = loss_object(target, prediction)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

def generate_train_loss_object():
    return metrics.Mean(name='train_loss')

def generate_train_accuracy_object():
    return metrics.SparseCategoricalAccuracy(name='train_accuracy')