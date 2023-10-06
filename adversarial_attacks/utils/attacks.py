import numpy as np
import tensorflow as tf


@tf.function
def confidence(y_true, y_pred, from_logits=True, labels_infhot=None, targeted=False):
    if not from_logits:
        raise ValueError('Only available using logits.')

    if labels_infhot is None:
        labels_infhot = tf.where(tf.equal(y_true, 1.), np.inf, 0.)

    if not targeted:
        real = tf.math.reduce_sum(y_true * y_pred, axis=1)
        other = tf.math.reduce_max(y_pred - labels_infhot, axis=1)
        return real - other
    else:
        target = tf.math.reduce_sum(y_true * y_pred, axis=1)
        other = tf.math.reduce_max(y_pred - labels_infhot, axis=1)
        return - (target - other)


@tf.function
def confidence_for_model_and_input(model, input, labels, labels_infhot=None, targeted=False):
    logits = model(input, logits_or_softmax='logits')
    return confidence(labels, logits, from_logits=True, labels_infhot=labels_infhot, targeted=targeted)


@tf.function
def _is_adv_confidence(c, confidence_threshold, return_confidence=False):
    is_adv = tf.math.less(c, - confidence_threshold)
    if return_confidence:
        return is_adv, c
    return is_adv


@tf.function
def is_adv_confidence(model, input, labels, confidence_threshold, labels_infhot=None, targeted=False,
                      return_confidence=False):
    c = confidence_for_model_and_input(model, input, labels, labels_infhot, targeted)
    return _is_adv_confidence(c, confidence_threshold, return_confidence=return_confidence)


@tf.function
def is_adv_confidence_for_logits(logits, labels, confidence_threshold, labels_infhot=None, targeted=False,
                                 return_confidence=False):
    c = confidence(labels, logits, labels_infhot=labels_infhot, targeted=targeted)
    return _is_adv_confidence(c, confidence_threshold, return_confidence=return_confidence)
