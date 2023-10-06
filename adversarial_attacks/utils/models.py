import numpy as np
import tensorflow as tf


def _get_jpeg_to_rgb_put_together_blocks_filter(chroma_subsampling):
    """
    Returns a convolution filter that separates or puts together 8*8 blocks of an image,
    depending on whether a standard convolution or transposed convolution is used.

    For technical reasons, either the input/output has shape (w, h, 64) instead of (w, h, 8, 8).
    So, the input/output must be reshaped before/after calling the convolution.

    :param chroma_subsampling:
    :return:
    """
    if not chroma_subsampling:
        luma_filter = np.zeros((8, 8, 1, 64), np.float32)
        for w in range(8):
            for h in range(8):
                luma_filter[w][h][0][w * 8 + h] = 1.0
        return luma_filter
    else:
        chroma_filter = np.zeros((16, 16, 1, 64), np.float32)
        for w in range(0, 16, 2):
            for h in range(0, 16, 2):
                channel_index = w // 2 * 8 + h // 2
                chroma_filter[w][h][0][channel_index] = 1.0
                chroma_filter[w + 1][h][0][channel_index] = 1.0
                chroma_filter[w][h + 1][0][channel_index] = 1.0
                chroma_filter[w + 1][h + 1][0][channel_index] = 1.0

        return chroma_filter


class PutTogetherBlocksJPEGToRGBTransposedConvolution(tf.keras.layers.Conv2DTranspose):
    """
    Transposed Convolution to put together blocks.
    If the input is chroma subsampled, each pixel will be repeated four times in the output.
    """

    def __init__(self, subsampling: bool = False, name='put_together_blocks_transposed_conv'):
        self._subsampling = subsampling
        self._filter = tf.constant_initializer(_get_jpeg_to_rgb_put_together_blocks_filter(self._subsampling))

        mult_w, mult_h = (8, 8) if not self._subsampling else (16, 16)

        super().__init__(1, (mult_h, mult_w),
                         strides=(mult_h, mult_w),
                         kernel_initializer=self._filter,
                         trainable=False, name=name)


class SeparateBlocksRGBToJPEGConvolution(tf.keras.layers.Conv2D):
    """
    Convolution to separate 8*8 blocks inside the images.
    When a channel is to be subsampled, it has to be before calling the layer.
    """

    def __init__(self, name='separate_blocks_conv'):
        self._filter = tf.constant_initializer(_get_jpeg_to_rgb_put_together_blocks_filter(False))
        super().__init__(64, (8, 8),
                         strides=(8, 8),
                         kernel_initializer=self._filter,
                         trainable=False,
                         name=name)
