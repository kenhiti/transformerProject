import tensorflow as tf
from keras import metrics as metrics
from keras import losses as losses

def create_loss_object():
    """
        When models aren't pass by the activation function, this called logits.
        The parameter from_logits = True, when you want to get data from the last layer

        When the reduction parameter is not None, which means that loss is calculated batch by batch,
        when the reduction is None it means that we need to use a custom reduction calculator
    """
    return losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

"""
    The prediction is the decoder output that indicate the portuguese phrases.
    Target is originals phrases that inserted in database
"""
def loss_function(loss_object, target, prediction):
    """
        Insert mask to calculate loss value.
        Where the zeroes included in the padding step will not be calculated
    """
    mask = tf.math.logical_not(tf.math.equal(target, 0))
    loss_ = loss_object(target, prediction)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

def generate_train_loss_object():
    return metrics.Mean(name='train_loss')

def generate_train_accuracy_object():
    return metrics.SparseCategoricalAccuracy(name='train_accuracy')