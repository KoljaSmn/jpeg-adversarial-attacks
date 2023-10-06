import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from adversarial_attacks.utils.tensorflow import deg2rad
from adversarial_attacks.utils.general import round_or_approx

tf_pi = tf.constant(np.pi)


@tf.function
def rgb_to_cielab(rgb: tf.Tensor):
    """
    Converts a rgb image to cielab color space.
    :param rgb: tf.Tensor of shape (w, h, 3) or (bs, w, h, 3). Range [0, 255].
    """
    return tfio.experimental.color.rgb_to_lab(rgb / 255., illuminant="E")


@tf.function
def cielab_to_rgb(cielab: tf.Tensor):
    """
    Converts a cielab image to rgb color space.
    :param cielab: tf.Tensor of shape (w, h, 3) or (bs, w, h, 3).
    """
    return tfio.experimental.color.lab_to_rgb(cielab, illuminant="E") * 255.


@tf.function
def _cart2polar_2pi(x, y):
    """convert cartesian coordinates to polar (uses non-standard theta range!)

    copied from skimage/color/delta_e.py and adapted to tensorflow.

    NON-STANDARD RANGE! Maps to ``(0, 2*pi)`` rather than usual ``(-pi, +pi)``
    """
    x = tf.cast(x, tf.float32)
    r, t = tf.experimental.numpy.hypot(x, y), tf.math.atan2(y, x)
    t += tf.where(t < 0., 2 * tf_pi, 0)
    return r, t


@tf.function
def deltaE_ciede2000(lab1, lab2, kL=1, kC=1, kH=1):
    """
    Assumes the channel axis to be the last axis (-1, 3).
    Assumes batched input.

    Mainly copied from skimage/color/delta_e.py and adapted to tensorflow.
    https://github.com/scikit-image/scikit-image/blob/main/skimage/color/delta_e.py
    """
    lab1, lab2 = tf.cast(lab1, tf.float32), tf.cast(lab2, tf.float32)

    lab1 = tf.transpose(lab1, [3, 0, 1, 2])
    lab2 = tf.transpose(lab2, [3, 0, 1, 2])

    L1, a1, b1 = lab1[0], lab1[1], lab1[2]
    L2, a2, b2 = lab2[0], lab2[1], lab2[2]

    # distort `a` based on average chroma
    # then convert to lch coordines from distorted `a`
    # all subsequence calculations are in the new coordiantes
    # (often denoted "prime" in the literature)
    Cbar = 0.5 * (tf.experimental.numpy.hypot(a1, b1) + tf.experimental.numpy.hypot(a2, b2))
    c7 = Cbar ** 7
    G = 0.5 * (1 - tf.math.sqrt(c7 / (c7 + 25 ** 7)))
    scale = 1 + G
    C1, h1 = _cart2polar_2pi(a1 * scale, b1)
    C2, h2 = _cart2polar_2pi(a2 * scale, b2)
    # recall that c, h are polar coordiantes.  c==r, h==theta

    # ciede2000 has four terms to delta_e:
    # 1) Luminance term
    # 2) Hue term
    # 3) Chroma term
    # 4) hue Rotation term

    # lightness term
    Lbar = 0.5 * (L1 + L2)
    tmp = (Lbar - 50) ** 2
    SL = 1 + 0.015 * tmp / tf.math.sqrt(20 + tmp)
    L_term = (L2 - L1) / (kL * SL)

    # chroma term
    Cbar = 0.5 * (C1 + C2)  # new coordiantes
    SC = 1 + 0.045 * Cbar
    C_term = (C2 - C1) / (kC * SC)

    # hue term
    h_diff = h2 - h1
    h_sum = h1 + h2
    CC = C1 * C2

    dH = tf.identity(h_diff)
    dH = tf.where(h_diff > tf_pi, dH - 2 * tf_pi, dH)
    dH = tf.where(h_diff < -tf_pi, dH + 2 * tf_pi, dH)
    dH = tf.where(CC == 0., 0., dH)  # if r == 0, dtheta == 0
    dH_term = 2 * tf.math.sqrt(CC) * tf.math.sin(dH / 2)

    Hbar = tf.identity(h_sum)
    mask = tf.math.logical_and(CC != 0., tf.math.abs(h_diff) > tf_pi)
    Hbar = tf.where(tf.math.logical_and(mask, h_sum < 2 * tf_pi), Hbar + 2 * tf_pi, Hbar)
    Hbar = tf.where(tf.math.logical_and(mask, h_sum >= 2 * tf_pi), Hbar - 2 * tf_pi, Hbar)
    Hbar = tf.where(CC == 0., Hbar * 2, Hbar)
    Hbar *= 0.5

    T = (1 -
         0.17 * tf.math.cos(Hbar - deg2rad(30)) +
         0.24 * tf.math.cos(2 * Hbar) +
         0.32 * tf.math.cos(3 * Hbar + deg2rad(6)) -
         0.20 * tf.math.cos(4 * Hbar - deg2rad(63))
         )
    SH = 1 + 0.015 * Cbar * T

    H_term = dH_term / (kH * SH)

    # hue rotation
    c7 = Cbar ** 7
    Rc = 2 * tf.math.sqrt(c7 / (c7 + 25 ** 7))
    dtheta = deg2rad(30) * tf.math.exp(-((tf.experimental.numpy.rad2deg(Hbar) - 275) / 25) ** 2)
    R_term = -tf.math.sin(2 * dtheta) * Rc * C_term * H_term

    # put it all together
    dE2 = L_term ** 2
    dE2 += C_term ** 2
    dE2 += H_term ** 2
    dE2 += R_term
    ans = tf.math.sqrt(tf.maximum(dE2, 0))
    return ans


@tf.function
def cielab_ciede2000(img1, img2):
    """
    Returns pixel-wise ciede2000 distance of two images or batches of images (cielab).
    """
    return deltaE_ciede2000(img1, img2)


@tf.function
def rgb_ciede2000(img1, img2):
    """
    Returns pixel-wise ciede2000 distance of two images or batches of images (rgb).
    """
    return cielab_ciede2000(rgb_to_cielab(img1), rgb_to_cielab(img2))


# constant matrix to convert rgb to ycbcr
rgb_to_ycbcr_xform = tf.constant(np.array([[0.299, 0.587, 0.114],
                                           [-0.168736, -0.331264, 0.5],
                                           [0.5, -0.418688, -0.081312]]), dtype=tf.float64)


def rgb_to_ycbcr(rgb: tf.Tensor, round: str = 'round'):
    """
    Converts a rgb image to ycbcr

    :param rgb: rgb image of shape (w, h, 3) or (bs, w, h, 3)
    :return: ycbcr image of shape (w, h, 3) or (bs, w, h, 3)
    """
    value = tf.cast(rgb, tf.float64)
    value = tf.linalg.matmul(value, rgb_to_ycbcr_xform, transpose_b=True)
    value = value + tf.constant([0, 128, 128], value.dtype)

    value = tf.clip_by_value(value, 0., 255.)
    value = round_or_approx(value, round)

    return value


# constant matrix to convert ycbcr to rgb
ycbcr_to_rgb_xform = tf.constant(np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]]), dtype=tf.float32)


def ycbcr_to_rgb(ycbcr, round: str = 'round'):
    """
    Converts a ycbcr image to rgb.

    :param ycbcr: rgb image of shape (w, h, 3) or (bs, w, h, 3)
    :return: rgb image of shape (w, h, 3) or (bs, w, h, 3)
    """

    value = tf.cast(ycbcr, tf.float32)

    value = value - tf.constant([0, 128, 128], value.dtype)
    value = tf.linalg.matmul(value, ycbcr_to_rgb_xform, transpose_b=True)

    value = tf.clip_by_value(value, 0., 255.)
    value = round_or_approx(value, round)

    return value


@tf.function
def tf_ycbcr_to_rgb(input, round: str = 'round'):
    return ycbcr_to_rgb(input, round)


def get_color_transformation_func(input_color_model, output_color_model):
    if output_color_model == input_color_model:
        return lambda x1: x1

    if input_color_model == 'rgb':
        if output_color_model == 'cielab':
            return lambda x1: rgb_to_cielab(x1)
        if output_color_model == 'ycbcr':
            return lambda x1: rgb_to_ycbcr(x1)
    elif input_color_model == 'ycbcr':
        if output_color_model == 'cielab':
            return lambda x1: rgb_to_cielab(ycbcr_to_rgb(x1))
        if output_color_model == 'rgb':
            return lambda x1: ycbcr_to_rgb(x1)

    raise NotImplementedError('Combination of {} as input color model and {} as distance color model'.format
                              (input_color_model, output_color_model))


def get_color_transformation_func_two_images(input_color_model, distance_color_model):
    if distance_color_model != 'ciede2000':
        fn = get_color_transformation_func(input_color_model, distance_color_model)
        return lambda x1, x2: (fn(x1), fn(x2))

    def rgb_ciede(x1, x2):
        ciede = rgb_ciede2000(x1, x2)
        return ciede, tf.zeros_like(ciede)

    if input_color_model == 'rgb' and distance_color_model == 'ciede2000':
        return lambda x1, x2: rgb_ciede(x1, x2)
    elif input_color_model == 'ycbcr' and distance_color_model == 'ciede2000':
        return lambda x1, x2: rgb_ciede(ycbcr_to_rgb(x1), ycbcr_to_rgb(x2))

    raise NotImplementedError('Combination of {} as input color model and {} as distance color model'.format
                              (input_color_model, distance_color_model))
