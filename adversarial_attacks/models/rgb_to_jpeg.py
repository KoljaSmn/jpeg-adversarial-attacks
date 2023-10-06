import tensorflow as tf

import adversarial_attacks.utils.color
import adversarial_attacks.utils.general
import adversarial_attacks.utils.models
from adversarial_attacks.utils import jpeg as jpeg_utils
from adversarial_attacks.utils.general import timestamp_random
import adversarial_attacks.models.models

from adversarial_attacks.utils.models import SeparateBlocksRGBToJPEGConvolution


class RGBToJpegModel(adversarial_attacks.models.models.Model):
    """
    Model that converts RGB data to JPEG.
    """

    def __init__(self, dataset, jpeg_quality: int, round: str = 'round', chroma_subsampling: bool = True):
        """
        :param dataset: the name of the dataset, 'cifar10', 'imagenet'
        :param jpeg_quality: the jpeg quality of the output
        :param round: one of 'round', 'shin', None.
               'round' for correctly rounded, quantized JPEG data,
               'shin' to use the rounding approximation used in
               Shin & Song's paper: Jpeg-resistant adversarial images, NIPS 2017.
               None for no rounding -> unquantized data
               Note that the model is not differentiable when set to 'round'.
        :param chroma_subsampling: Whether the output is chroma subsampled or not
        """
        self.jpeg_quality = jpeg_quality

        if type(round) == bool:
            raise ValueError("round must be one of 'round', 'shin', None.")

        self.round = round
        self.chroma_subsampling = chroma_subsampling

        super().__init__(dataset, save_model=False)

    def __call__(self, rgb):
        """
        Converts RGB data to JPEG.
        :param rgb:
        :return:
        """
        jpeg = self.model(rgb)
        return jpeg[0], jpeg[1], jpeg[2]

    def build_model(self):
        """
        Builds the conversion model as tf.keras.Model
        :return:
        """

        # get the luma and chroma input shapes
        luma_output_shape = self.not_subsampled_input_shape
        chroma_output_shape = self.subsampled_input_shape if self.chroma_subsampling \
            else self.not_subsampled_input_shape
        y_w, y_h = luma_output_shape[0:2]
        c_w, c_h = chroma_output_shape[0:2]

        rgb = tf.keras.Input(shape=self.rgb_input_shape)

        # convert rgb to ycbcr pixels
        ycbcr = adversarial_attacks.utils.color.rgb_to_ycbcr(rgb, round=None)
        Y, Cb, Cr = ycbcr[:, :, :, 0], ycbcr[:, :, :, 1], ycbcr[:, :, :, 2]
        # expand dims, technical reasons as it allows to use ConvLayers (expanded dim=channel dim)
        Y, Cb, Cr = tf.expand_dims(Y, axis=3), tf.expand_dims(Cb, axis=3), tf.expand_dims(Cr, axis=3)

        # if chroma subsampling is used, resize the chroma channels
        if self.chroma_subsampling:
            c_shape = tf.convert_to_tensor([self.rgb_input_shape[0] // 2, self.rgb_input_shape[1] // 2])
            Cb, Cr = tf.image.resize(Cb, c_shape), tf.image.resize(Cr, c_shape)

        # separates 8*8 blocks
        Cr = SeparateBlocksRGBToJPEGConvolution(name='Cr_conv')(Cr)
        # reshapes (c_w, c_h, 64) output to (c_w, c_h, 8, 8)
        # the output has shape (c_w, c_h, 64) for technical reasons as it allows to use the tf conv2d layer
        Cr = tf.keras.layers.Reshape((c_w, c_h, 8, 8))(Cr)
        # dct to convert pixels to coefficients
        Cr = jpeg_utils.block_dct(Cr - 128.)

        Cb = SeparateBlocksRGBToJPEGConvolution(name='Cb_conv')(Cb)
        Cb = tf.keras.layers.Reshape((c_w, c_h, 8, 8))(Cb)
        Cb = jpeg_utils.block_dct(Cb - 128.)

        Y = SeparateBlocksRGBToJPEGConvolution(name='Y_conv')(Y)
        Y = tf.keras.layers.Reshape((y_w, y_h, 8, 8))(Y)
        Y = jpeg_utils.block_dct(Y - 128.)

        Y, Cb, Cr = jpeg_utils.quantize(Y, Cb, Cr, jpeg_quality=self.jpeg_quality, round=self.round)
        Y, Cb, Cr = tf.keras.layers.Reshape((y_w, y_h, 64))(Y), \
                    tf.keras.layers.Reshape((c_w, c_h, 64))(Cb), \
                    tf.keras.layers.Reshape((c_w, c_h, 64))(Cr)
        Y, Cb, Cr = jpeg_utils.zigzag(Y), jpeg_utils.zigzag(Cb), jpeg_utils.zigzag(Cr)

        model = tf.keras.Model(inputs=[rgb], outputs=[Y, Cb, Cr],
                               name='rgb_to_jpeg_{}_{}_{}'.format(self.ds_name, self.jpeg_quality, timestamp_random()))

        model.compile(loss='mse')

        return model

    def _shin_rounding_approximation(self, x):
        return adversarial_attacks.utils.general.shin_rounding_approximation(x)
