from abc import ABC, abstractmethod

import tensorflow as tf

import adversarial_attacks.utils.attacks
import adversarial_attacks.utils.transformation
from adversarial_attacks.attacks.attacks import Attack, ycbcr_clip_and_cast, YCBCR_CLIP_MIN, YCBCR_CLIP_MAX, \
    default_alpha
from adversarial_attacks.config.config import Config
from adversarial_attacks.utils.transformation import ycbcr_tuple_to_rgb

from adversarial_attacks.models.rgb_to_jpeg import RGBToJpegModel
from adversarial_attacks.models.jpeg_to_rgb import JpegToRGBModel


class YCbCrAdversarialAttack(Attack, ABC):
    """
    Generic YCbCr Attack class.
    """
    _model_type = 'ycbcr'

    @abstractmethod
    def __init__(self, dataset, model_name, loss_fn='crossentropy', model=None, output_jpeg_quality=None,
                 output_chroma_subsampling: bool = True):

        self._output_jpeg_quality = output_jpeg_quality
        self._output_chroma_subsampling = output_chroma_subsampling

        self._rgb_to_jpeg_model, self._jpeg_to_rgb_model = None, None
        if self._output_jpeg_quality is not None:
            self._rgb_to_jpeg_model = RGBToJpegModel(dataset, jpeg_quality=self._output_jpeg_quality, round='round',
                                                     chroma_subsampling=self._output_chroma_subsampling)
            self._jpeg_to_rgb_model = JpegToRGBModel(dataset, jpeg_quality=self._output_jpeg_quality,
                                                     chroma_subsampling=self._output_chroma_subsampling, round=False)

        super().__init__(dataset, model_name, loss_fn=loss_fn, model=model)

    def _jpeg_compression(self, images):
        if self._output_jpeg_quality is not None:
            rgb = ycbcr_tuple_to_rgb(images, round=None)
            compressed_rgb = self._jpeg_to_rgb_model(self._rgb_to_jpeg_model(rgb))
            return adversarial_attacks.utils.transformation.rgb_to_ycbcr_tuple(compressed_rgb)
        return images

    def predict(self, images, logits_or_softmax='logits', overwrite_model=None):
        return super(YCbCrAdversarialAttack, self).predict(ycbcr_tuple_to_rgb(images, round=None),
                                                           logits_or_softmax=logits_or_softmax,
                                                           overwrite_model=overwrite_model)

    def batched_output_signature(self):
        w, h = Config.INPUT_SHAPE[self.get_dataset_name()]['rgb'][:2]
        batched_output_signature = (
            (tf.TensorSpec(shape=(
                None,
                w, h
            )),
             tf.TensorSpec(shape=(
                 None,
                 w, h
             )),
             tf.TensorSpec(shape=(
                 None,
                 w, h
             ))
            ))
        return batched_output_signature

    @abstractmethod
    def cache_key(self):
        k = f'ycbcr_{super().cache_key()}'
        if self._output_jpeg_quality is not None:
            k += f'_output_jq_{self._output_jpeg_quality}_output_cs_{self._output_chroma_subsampling}'

        return k

    @tf.function
    def linf_bounds(self, Y, Cb, Cr, eps_Y, eps_Cb, eps_Cr):
        # compute the l_inf bounds for every channel
        Y_clip = Y - eps_Y, Y + eps_Y
        Cb_clip = Cb - eps_Cb, Cb + eps_Cb
        Cr_clip = Cr - eps_Cr, Cr + eps_Cr
        return Y_clip, Cb_clip, Cr_clip

    @tf.function
    def gradient(self, images, labels, ret_loss=False):
        """
        returns the gradient for the coefficient specified with watch
        :param model: The _model to compute the loss for the gradients.
        :param images: batch of images
        :param labels: batch of labels
        :returns the gradient of the loss function of the image and the _model.
                 for Y, Cb, Cr
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([images[0], images[1], images[2]])
            prediction = self.predict(images, logits_or_softmax='logits')  # predict the batch
            loss = self._loss_fn(labels, prediction)  # compute the loss

        # compute the gradients
        Y_grad, Cb_grad, Cr_grad = tape.gradient(loss, images[0]), \
                                   tape.gradient(loss, images[1]), \
                                   tape.gradient(loss, images[2])
        if ret_loss:
            return (Y_grad, Cb_grad, Cr_grad), loss

        return Y_grad, Cb_grad, Cr_grad

    @tf.function
    def gradient_sign(self, images, labels):
        """
        :param _model: The _model to compute the loss for the gradients.
        :param images: batch of images
        :param labels: batch of labels
        :returns the sign of the gradient of the loss function of the image and the _model.
                 for Y, Cb, Cr
        """
        gradient = self.gradient(images, labels)
        return tf.sign(gradient[0]), tf.sign(gradient[1]), tf.sign(gradient[2])

    @tf.function
    def gradient_sign_perturbation(self, images, labels, pert_Y, pert_Cb, pert_Cr, round: bool = True):
        Y_gradient_sign, Cb_gradient_sign, Cr_gradient_sign = self.gradient_sign(images, labels)
        # compute the coefficients gradients

        # perturbation
        Y = images[0] + Y_gradient_sign * pert_Y
        Cb = images[1] + Cb_gradient_sign * pert_Cb
        Cr = images[2] + Cr_gradient_sign * pert_Cr

        return ycbcr_clip_and_cast(Y, Cb, Cr, YCBCR_CLIP_MIN, YCBCR_CLIP_MAX, round=round)

    @tf.function
    def rgb_to_input_datatype_conversion(self, images):
        return adversarial_attacks.utils.transformation.rgb_to_ycbcr_tuple(images)

    @tf.function
    def input_datatype_to_rgb_conversion(self, images):
        return adversarial_attacks.utils.transformation.ycbcr_tuple_to_rgb(images, round=None)


class YCbCrFGSM(YCbCrAdversarialAttack):
    """
    Class for Fast Gradient Sign Method (YCbCr Version).
    Goodfellow et al., 2014: Explaining and Harnessing Adversarial Examples

    To attack images, you can either instantiate this class using a params object and
    then call it with different input images, or you can use the static attack method.
    """
    _attack_name = 'fgsm'

    def __init__(self, dataset, model_name, loss_fn='crossentropy',
                 eps_Y: float = 1, eps_Cb: float = 1, eps_Cr: float = 1, model=None, output_jpeg_quality=None,
                 output_chroma_subsampling: bool = True):
        super().__init__(dataset, model_name, loss_fn=loss_fn, model=model,
                         output_chroma_subsampling=output_chroma_subsampling, output_jpeg_quality=output_jpeg_quality)

        self.eps_Y = eps_Y
        self.eps_Cb = eps_Cb
        self.eps_Cr = eps_Cr

    def cache_key(self):
        return f'{super().cache_key()}_eps_Y_{self.eps_Y}_eps_Cb_{self.eps_Cb}_eps_Cr_{self.eps_Cr}'

    @tf.function
    def attack_datatype(self, images, labels):
        """

        :param images: batch of rgb images
        :param labels: batch of one-hot labels
        :return:
        """
        return self._jpeg_compression(self._attack(images, labels, self.eps_Y, self.eps_Cb, self.eps_Cr))

    def _batch_view_shape(self, batch_size):
        return [batch_size, 1, 1]

    @tf.function
    def _attack(self, images, labels, eps_Y, eps_Cb, eps_Cr):
        Y, Cb, Cr = tf.cast(images[0], tf.float32), tf.cast(images[1], tf.float32), tf.cast(images[2], tf.float32)
        images = (Y, Cb, Cr)

        images = self.gradient_sign_perturbation(images, labels, eps_Y, eps_Cb, eps_Cr,
                                                 round=True)
        return (images[0], images[1], images[2])


class YCbCrBIM(YCbCrFGSM):
    """
    Class for Basic Iterative Method (YCbCr Version).
    Kurakin et al., 2016: Adversarial examples in the physical world

    To attack images, you can either instantiate this class using a params object and
    then call it with different input images, or you can use the static attack method.
    """
    _attack_name = 'bim'

    def __init__(self, dataset, model_name, loss_fn='crossentropy',
                 eps_Y: float = 1, eps_Cb: float = 1, eps_Cr: float = 1,
                 alpha_Y: float = None, alpha_Cb: float = None, alpha_Cr: float = None,
                 T: int = 10, model=None, output_jpeg_quality=None,
                 output_chroma_subsampling: bool = True
                 ):
        alpha_Y, alpha_Cb, alpha_Cr = default_alpha(alpha_Y, eps_Y, T), \
                                      default_alpha(alpha_Cb, eps_Cb, T), \
                                      default_alpha(alpha_Cr, eps_Cr, T)
        super().__init__(dataset, model_name=model_name, loss_fn=loss_fn, eps_Y=eps_Y, eps_Cb=eps_Cb, eps_Cr=eps_Cr,
                         model=model, output_chroma_subsampling=output_chroma_subsampling,
                         output_jpeg_quality=output_jpeg_quality)

        self.alpha_Y = alpha_Y
        self.alpha_Cb = alpha_Cb
        self.alpha_Cr = alpha_Cr
        self.T = T

    def cache_key(self):
        return f'{super().cache_key()}_alpha_Y_{self.alpha_Y}_alpha_Cb_{self.alpha_Cb}_alpha_Cr_{self.alpha_Cr}_T_{self.T}'

    @tf.function
    def attack_datatype(self, images, labels):
        """

        :param images: batch of rgb images
        :param labels: batch of one-hot labels
        :return:
        """
        return self._jpeg_compression(
            self._attack(images, labels, self.eps_Y, self.eps_Cb, self.eps_Cr, self.alpha_Y, self.alpha_Cb,
                         self.alpha_Cr, self.T))

    @tf.function
    def _attack(self, images, labels, eps_Y, eps_Cb, eps_Cr, alpha_Y, alpha_Cb, alpha_Cr, T):
        Y, Cb, Cr = tf.cast(images[0], tf.float32), tf.cast(images[1], tf.float32), tf.cast(images[2], tf.float32)
        images = (Y, Cb, Cr)

        Y_clip, Cb_clip, Cr_clip = self.linf_bounds(Y, Cb, Cr, eps_Y, eps_Cb, eps_Cr)

        for t in range(T):
            # compute the coefficients gradient signs.
            images = self.gradient_sign_perturbation(images, labels,
                                                     alpha_Y,
                                                     alpha_Cb,
                                                     alpha_Cr,
                                                     round=False)

        # clip the image onto the valid range
        Y, Cb, Cr = images[0], images[1], images[2]
        images = ycbcr_clip_and_cast(Y, Cb, Cr, YCBCR_CLIP_MIN, YCBCR_CLIP_MAX, Y_clip, Cb_clip, Cr_clip, round=True)
        return (images[0], images[1], images[2])
