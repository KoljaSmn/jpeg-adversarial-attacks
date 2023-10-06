import os
import shutil
from datetime import datetime
from random import randrange
from sys import platform

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm as std_tqdm


def save_as_png(rgb_image, filepath):
    """
    Saves an RGB image as png file.
    """
    tf.keras.utils.save_img(filepath, rgb_image)


@tf.function
def concat_tf_tensors(t1: tf.Tensor, t2: tf.Tensor):
    """
    Concatenate two tf.Tensors on axis 0
    If one tensor is None, return the other.
    """
    if t1 is None:
        return t2
    if t2 is None:
        return t1
    return tf.concat([t1, t2], axis=0)


def timestamp_random():
    """
    Timestamp + random number to save evaluation results etc.
    """
    return datetime.now().strftime("%d%m%Y%H%M%S") + str(randrange(0, 1000000))


def remove_dir(path):
    """
    removes a directory.
    """
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


def check_linux() -> bool:
    """
    Check if os is linux. torchjpeg is only available on linux.
    """
    return 'linux' in platform


def makedirs(path):
    """
    Create a directory.
    """
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def shin_rounding_approximation(x):
    """
    See Shin & Song, 2017:
    :param x:
    :return:
    """
    return tf.math.round(x) + tf.math.pow(x - tf.math.round(x), 3)


def round_or_approx(x, round):
    """
    :param x:
    :param round:
    :return:
    """
    if round == 'round':
        return tf.math.round(x)
    elif round == 'shin':
        return shin_rounding_approximation(x)
    elif round is None:
        return x

    else:
        raise ValueError(f'Rounding scheme {round} unknown')
