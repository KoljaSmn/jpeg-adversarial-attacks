# Collection of utils to complement tensorflow implementation, e.g. converted numpy functions
import numpy as np
import tensorflow as tf

tf_pi = tf.constant(np.pi)


@tf.function
def deg2rad(x):
    x = tf.cast(x, tf.float32)
    return x * (tf_pi / 180.0)


@tf.function
def tf_concat(tf1, tf2):
    if tf1 is None:
        return tf2
    if tf2 is None:
        return tf1

    return tf.concat([tf1, tf2], axis=0)


def log(x, base):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator
