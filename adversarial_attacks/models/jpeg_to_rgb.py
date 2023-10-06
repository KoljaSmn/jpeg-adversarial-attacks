import tensorflow as tf

from adversarial_attacks.models.models import Model
from adversarial_attacks.utils.models import PutTogetherBlocksJPEGToRGBTransposedConvolution
import adversarial_attacks.utils.color
from adversarial_attacks.utils import jpeg as jpeg_utils
from adversarial_attacks.utils.general import timestamp_random


class JpegToRGBModel(Model):
    """
    Model that converts JPEG images to RGB data.
    """
    CACHE = {}

    def __init__(self, dataset_name,
                 jpeg_quality: int = 100, round: bool = False, chroma_subsampling: bool = True):
        """
        :param dataset_name: the name of the dataset, 'cifar10', 'imagenet'
        :param jpeg_quality: the jpeg quality of the input
        :param round: whether the output RGB data should be quantized or not,
               note that the model is not differentiable when set to True
        :param chroma_subsampling: Whether the input is chroma subsampled or not
        """
        load_model_name = None
        self.jpeg_quality = jpeg_quality
        self.round = round
        self.chroma_subsampling = chroma_subsampling

        super().__init__(dataset_name=dataset_name, save_model_name=None,
                         load_model_name=load_model_name,
                         save_model=False)

    def __call__(self, inp):
        """
        Converts (Y, Cb, CR) tuple of JPEG data to RGB.
        :param inp:
        :return:
        """
        return self.model(inp)

    def build_model(self):
        """
        Builds the conversion model as tf.keras.Model
        :return:
        """
        rounding_op = 'round' if self.round else None

        # get the luma and chroma input shapes
        luma_input_shape = self.not_subsampled_input_shape
        chroma_input_shape = self.subsampled_input_shape if self.chroma_subsampling \
            else self.not_subsampled_input_shape

        y_w, y_h = luma_input_shape[0:2]
        c_w, c_h = chroma_input_shape[0:2]

        Y_in = tf.keras.Input(luma_input_shape, name='y_in')
        Cb_in = tf.keras.Input(chroma_input_shape, name='cb_in')
        Cr_in = tf.keras.Input(chroma_input_shape, name='cr_in')

        Y, Cb, Cr = Y_in, Cb_in, Cr_in

        # un-zig-zag the coefficients
        Y, Cb, Cr = jpeg_utils.unzigzag(Y), jpeg_utils.unzigzag(Cb), jpeg_utils.unzigzag(Cr)

        # reshapes the coefficients to 8*8 blocks
        Y, Cb, Cr = tf.keras.layers.Reshape((y_w, y_h, 8, 8))(Y), \
                    tf.keras.layers.Reshape((c_w, c_h, 8, 8))(Cb), \
                    tf.keras.layers.Reshape((c_w, c_h, 8, 8))(Cr)

        # dequantizes coefficients (multiplication with quantization matrix)
        Y, Cb, Cr = jpeg_utils.dequantize(Y, Cb, Cr, jpeg_quality=self.jpeg_quality, round=None)

        # idct to convert coefficients to pixel values
        Y = jpeg_utils.block_idct(Y) + 128.
        # reshape each block to 64-vector instead of 8*8
        # this is done for technical reasons as it allows to use the tf conv2d layer
        Y = tf.keras.layers.Reshape((y_w, y_h, 64))(Y)
        Y = PutTogetherBlocksJPEGToRGBTransposedConvolution(subsampling=False,
                                                            name='luma_transposed_conv')(Y)

        Cb = jpeg_utils.block_idct(Cb) + 128.
        Cb = tf.keras.layers.Reshape((c_w, c_h, 64))(Cb)
        Cb = PutTogetherBlocksJPEGToRGBTransposedConvolution(subsampling=self.chroma_subsampling,
                                                             name='Cb_transposed_conv')(Cb)

        Cr = jpeg_utils.block_idct(Cr) + 128.
        Cr = tf.keras.layers.Reshape((c_w, c_h, 64))(Cr)
        Cr = PutTogetherBlocksJPEGToRGBTransposedConvolution(subsampling=self.chroma_subsampling,
                                                             name='Cr_transposed_conv')(Cr)

        clip_min, clip_max = 0, 255
        Y, Cb, Cr = tf.clip_by_value(Y, clip_min, clip_max), tf.clip_by_value(Cb, clip_min, clip_max), \
                    tf.clip_by_value(Cr, clip_min, clip_max)

        # stack three channels
        YCbCr_stacked = tf.stack([Y, Cb, Cr], axis=-2)[:, :, :, :, 0]
        # convert ycbcr to rgb
        RGB = adversarial_attacks.utils.color.ycbcr_to_rgb(YCbCr_stacked, round=rounding_op)

        RGB = tf.clip_by_value(RGB, 0., 255.)

        model = tf.keras.Model(inputs=[Y_in, Cb_in, Cr_in], outputs=[RGB],
                               name='jpeg_to_rgb_{}_{}'.format(self.model_name, timestamp_random()))

        model.compile(loss=tf.keras.losses.MeanSquaredError())

        return model

    def get_metrics(self):
        metrics = {}
        for key in ['train', 'val']:
            metrics[key] = {}
            metrics[key]['loss'] = tf.keras.metrics.MSE()
        return metrics
