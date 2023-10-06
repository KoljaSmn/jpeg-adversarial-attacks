import numpy as np
import tensorflow as tf

import adversarial_attacks.models
import adversarial_attacks.models.jpeg_to_rgb
import adversarial_attacks.models.rgb_to_jpeg
import adversarial_attacks.utils
import adversarial_attacks.utils.color
from adversarial_attacks.config.tf_distribution import tf_distribution_strategy
from adversarial_attacks.utils.jpeg import dimensions, quantize_zigzagged_coefficients, \
    dequantize_zigzagged_coefficients, integer_differentiable_shin_rounding_approximation


# dictionaries to save transformation models that have already been built
RGB_TO_JPEG_MODEL = {}
JPEG_TO_RGB_MODEL = {}


@tf.autograph.experimental.do_not_convert
def get_jpeg_to_rgb_model(dataset_name: str,
                          jpeg_quality: int,
                          round: bool = True,
                          chroma_subsampling: bool = True) -> adversarial_attacks.models.jpeg_to_rgb.JpegToRGBModel:
    """
    Returns a jpeg to rgb transformation for the given dataset

    :param dataset_name: in 'cifar10', 'imagenet'
    :param jpeg_quality: the input's jpeg quality
    """
    if (dataset_name, jpeg_quality, round, chroma_subsampling) not in JPEG_TO_RGB_MODEL:
        JPEG_TO_RGB_MODEL[(
            dataset_name, jpeg_quality, round,
            chroma_subsampling)] = adversarial_attacks.models.jpeg_to_rgb.JpegToRGBModel(
            dataset_name=dataset_name,
            jpeg_quality=jpeg_quality,
            round=round,
            chroma_subsampling=chroma_subsampling)
    return JPEG_TO_RGB_MODEL[(dataset_name, jpeg_quality, round, chroma_subsampling)]


@tf.autograph.experimental.do_not_convert
def get_rgb_to_jpeg_model(dataset_name: str,
                          jpeg_quality: int,
                          round: str = 'round',
                          chroma_subsampling: bool = True) -> adversarial_attacks.models.rgb_to_jpeg.RGBToJpegModel:
    """
    Returns a rgb to jpeg transformation for the given dataset

    :param dataset_name: in 'cifar10', 'imagenet'
    :param jpeg_quality: the output's jpeg quality
    """
    if (dataset_name, jpeg_quality, round, chroma_subsampling) not in RGB_TO_JPEG_MODEL:
        RGB_TO_JPEG_MODEL[(dataset_name, jpeg_quality, round, chroma_subsampling)] = \
            adversarial_attacks.models.rgb_to_jpeg.RGBToJpegModel(
                dataset_name,
                jpeg_quality=jpeg_quality,
                round=round,
                chroma_subsampling=chroma_subsampling)
    return RGB_TO_JPEG_MODEL[(dataset_name, jpeg_quality, round, chroma_subsampling)]


def jpeg_to_rgb_batch(image: tuple, dataset: str, jpeg_quality: int, chroma_subsampling: bool = True) -> tf.Tensor:
    """
    Transforms a batch of jpeg images shape to RGB.

    :param image: tuple  (Y, Cb, Cr),
    where Y has shape (bs, w/8, h/8, 64) and the chroma channels have shape (bs, w/16, h/16, 64).
    :param dataset: cifar10 or imagenet
    """
    return get_jpeg_to_rgb_model(dataset, jpeg_quality, chroma_subsampling=chroma_subsampling)(image)


def rgb_to_jpeg_batch(rgb: tf.Tensor, dataset_name: str, jpeg_quality: int = 100,
                      chroma_subsampling: bool = True) -> tuple:
    """
    Transforms a batch of RGB images to JPEG.
    :param rgb: batched rgb images, shape (bs, w, h, 3)
    :param dataset_name: cifar10 or imagenet
    :return: tuple (Y, Cb, Cr) with shapes (bs, w/8, h/8, 3) for Y and (bs, w/16, h/16, 3) for chroma
    """
    model = get_rgb_to_jpeg_model(dataset_name, jpeg_quality, round='round', chroma_subsampling=chroma_subsampling)
    Y, Cb, Cr = model(rgb)
    return Y, Cb, Cr


def ycbcr_tuple_to_rgb(ycbcr_tuple: tuple, round: str = 'round') -> tf.Tensor:
    """
    Transforms ycbcr tuple to a rgb tensor.
    :param ycbcr_tuple: (Y, Cb, Cr). All three channels have shape (w, h) or (bs, w, h).
    :return: rgb tensor of shape (w, h, 3) or (bs, w, h, 3)
    """
    Y, Cb, Cr = ycbcr_tuple
    ycbcr_stack = tf.stack([Y, Cb, Cr], axis=-1)
    return adversarial_attacks.utils.color.ycbcr_to_rgb(ycbcr_stack, round=round)


def rgb_to_ycbcr_tuple(rgb: tf.Tensor) -> tuple:
    """
    Transforms rgb tensor to a ycbcr tuple.
    :param rgb: tensor of shape (bs, w, h, 3)
    :return ycbcr_tuple: (Y, Cb, Cr). All three channels have shape (bs, w, h).
    """

    ycbcr = adversarial_attacks.utils.color.rgb_to_ycbcr(rgb)
    Y, Cb, Cr = ycbcr[:, :, :, 0], ycbcr[:, :, :, 1], ycbcr[:, :, :, 2]
    return (Y, Cb, Cr)


def ycbcr_tuple_to_jpeg(ycbcr: tuple, dataset: str, jpeg_quality: int = 100, chroma_subsampling: bool = True) -> tuple:
    """
    Transforms rgb tensor to a ycbcr tuple.
    :param ycbcr: (Y, Cb, Cr). All three channels have shape (w, h) or (bs, w, h).
    :param dataset: name of dataset, e.g. cifar10 or imagenet
    """
    rgb = ycbcr_tuple_to_rgb(ycbcr)
    jpeg = rgb_to_jpeg_batch(rgb, dataset, jpeg_quality, chroma_subsampling=chroma_subsampling)
    return jpeg


def jpeg_to_ycbcr_tuple(jpeg: tuple, dataset: str, jpeg_quality: int = 100, chroma_subsampling: bool = True):
    """
    Transforms rgb tensor to a ycbcr tuple.
    :param jpeg: (Y, Cb, Cr). luma shape (bs, w/8, h/8, 64), chroma (bs, w/16, h/16, 64)
    :param dataset: e.g. cifar10 or imagenet
    :return ycbcr_tuple: (Y, Cb, Cr). All three channels have shape (w, h) or (bs, w, h).
    """
    rgb = jpeg_to_rgb_batch(jpeg, dataset, jpeg_quality, chroma_subsampling=chroma_subsampling)
    ycbcr = rgb_to_ycbcr_tuple(rgb)
    return ycbcr


class ShinJpegCompressionApproximationForRGBImages:
    """
    Compresses RGB images using the JPEG compression from
    Shin & Song, Jpeg-resistant adversarial images, NIPS 2017.

    """

    def __init__(self, dataset, jpeg_quality, chroma_subsampling):
        # model for jpeg compression. Uses the rounding approximation instead of true rounding.
        self.rgb_to_jpeg_approx = adversarial_attacks.models.rgb_to_jpeg.RGBToJpegModel(dataset,
                                                                                        jpeg_quality=jpeg_quality,
                                                                                        chroma_subsampling=chroma_subsampling,
                                                                                        round='shin')
        # converts JPEG data back into unquantized rgb data
        self.jpeg_to_rgb = adversarial_attacks.models.jpeg_to_rgb.JpegToRGBModel(dataset, jpeg_quality=jpeg_quality,
                                                                                 chroma_subsampling=chroma_subsampling,
                                                                                 round=False)

    def __call__(self, rgb):
        return self._convert(rgb)

    def _convert(self, rgb):
        return self.jpeg_to_rgb(self.rgb_to_jpeg_approx(rgb))
