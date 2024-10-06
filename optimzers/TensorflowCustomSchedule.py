import tensorflow as tf
from keras import optimizers as optimizers

class TensorflowCustomSchedule(optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(TensorflowCustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model,tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)