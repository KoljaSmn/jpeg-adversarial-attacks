from abc import ABC, abstractmethod

import tensorflow as tf

from adversarial_attacks.attacks.attacks import Attack, default_alpha
from adversarial_attacks.config.config import Config

from adversarial_attacks.models.rgb_to_jpeg import RGBToJpegModel
from adversarial_attacks.models.jpeg_to_rgb import JpegToRGBModel


class RGBAdversarialAttack(Attack, ABC):
    """
    Abstract RGB Attack class.
    """
    _model_type = 'rgb'

    def __init__(self, dataset, model_name, loss_fn='crossentropy', model=None, output_jpeg_quality=None,
                 output_chroma_subsampling: bool = True):
        """
        :param dataset: the name of the dataset
        :param model_name: the name of the model. If model is not None, model_name will be ignored.
        :param loss_fn: crossentropy or confidence.
        :param model: The model. If model is not None, model_name will be ignored.
                      This is mainly used for adversarial training.
                      It prevents the model from being loaded twice (once for training, once for attacking)
                      such that the images are created on the model that is being trained.
        :param output_jpeg_quality: The jpeg quality, the output is compressed to. None for no compression.
        :param output_chroma_subsampling: Whether to use chroma subsampling when compressing the output.
               only used if output_jpeg_quality is not None.
               Deactivated in the paper's experiments.
        """
        self._output_jpeg_quality = output_jpeg_quality
        self._output_chroma_subsampling = output_chroma_subsampling

        self._rgb_to_jpeg_model, self._jpeg_to_rgb_model = None, None
        if self._output_jpeg_quality is not None:
            # load a rgb to jpeg model for the given jpeg quality
            self._rgb_to_jpeg_model = RGBToJpegModel(dataset, jpeg_quality=self._output_jpeg_quality, round='round',
                                                     chroma_subsampling=self._output_chroma_subsampling)
            # load a jpeg to rgb model for the given jpeg quality
            # compression will be done by calling _jpeg_to_rgb_model(_rgb_to_jpeg_model(images))
            self._jpeg_to_rgb_model = JpegToRGBModel(dataset, jpeg_quality=self._output_jpeg_quality,
                                                     chroma_subsampling=self._output_chroma_subsampling, round=False)

        super().__init__(dataset, model_name, loss_fn=loss_fn, model=model)

    def _jpeg_compression(self, images):
        """
        Compressed the output to the self._output_jpeg_quality
        :param images:
        :return:
        """
        if self._output_jpeg_quality is not None:
            return self._jpeg_to_rgb_model(self._rgb_to_jpeg_model(images))
        return images

    @abstractmethod
    def cache_key(self):
        k = f'rgb_{super().cache_key()}'

        if self._output_jpeg_quality is not None:
            k += f'_output_jq_{self._output_jpeg_quality}_output_cs_{self._output_chroma_subsampling}'

        return k

    @tf.function
    def gradient(self, image, label, ret_loss=False):
        """
        :returns the gradient of the loss function of the image and the _model.
        """
        image = tf.cast(image, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(image)
            prediction = self.predict(image, logits_or_softmax='logits')
            # categorical_crossentropy loss
            loss = self._loss_fn(label, prediction)
        grad = tape.gradient(loss, image)
        grad = tf.where(tf.math.is_nan(grad), tf.random.normal(tf.shape(grad)), grad)
        if not ret_loss:
            return grad
        else:
            return grad, loss

    @tf.function
    def gradient_sign(self, image, label):
        """
        :returns the sign of the gradient of the loss function of the image and the _model.
        """
        return tf.sign(self.gradient(image, label))

    @staticmethod
    @tf.function
    def clip(x, min, max):
        """
        Clips the image x to be within range [min, max]
        :param x:
        :param min:
        :param max:
        :return:
        """
        return tf.clip_by_value(x, clip_value_min=min, clip_value_max=max)

    @tf.function
    def gradient_sign_perturbation(self, image, label, pert):
        """
        Perturbs the image in the gradient's direction by pert.
        :param image:
        :param label:
        :param pert:
        :return:
        """
        gradient_sign = self.gradient_sign(image, label)
        perturbed_x = image + gradient_sign * pert
        return perturbed_x

    @staticmethod
    @tf.function
    def l_inf_clip(x_new, x_old, delta):
        """
        Clips an image such that all values are in range [max(0, x_old-delta), min(255, x_old+delta)].
        Usually, delta should be the allowed perturbation epsilon.
        """
        return RGBAdversarialAttack.clip(RGBAdversarialAttack.clip(x_new, x_old - delta, x_old + delta), 0., 255.)

    @tf.function
    def rgb_to_input_datatype_conversion(self, images):
        """
        Input datatype is RGB. So, it just returns the input images.
        :param images:
        :return:
        """
        return images

    @tf.function
    def input_datatype_to_rgb_conversion(self, images):
        """
        Input datatype is RGB. So, it just returns the input images.
        :param images:
        :return:
        """
        return images


class RGBFGSM(RGBAdversarialAttack):
    """
    Fast gradient sign method.
    Goodfellow et al.: Explaining and Harnessing Adversarial Examples, ILCR 2015
    """
    _attack_name = 'fgsm'

    def __init__(self, dataset, model_name, epsilon, loss_fn='crossentropy', model=None, output_jpeg_quality=None,
                 output_chroma_subsampling: bool = True):
        """
        :param epsilon: the perturbation bound
        :param dataset: the name of the dataset
        :param model_name: the name of the model. If model is not None, model_name will be ignored.
        :param loss_fn: crossentropy or confidence.
        :param model: The model. If model is not None, model_name will be ignored.
                      This is mainly used for adversarial training.
                      It prevents the model from being loaded twice (once for training, once for attacking)
                      such that the images are created on the model that is being trained.
        :param output_jpeg_quality: The jpeg quality, the output is compressed to. None for no compression.
        :param output_chroma_subsampling: Whether to use chroma subsampling when compressing the output.
               only used if output_jpeg_quality is not None.
               Deactivated in the paper's experiments.
        """
        super().__init__(dataset, model_name, loss_fn=loss_fn, model=model,
                         output_chroma_subsampling=output_chroma_subsampling, output_jpeg_quality=output_jpeg_quality)
        self.epsilon = epsilon

    def cache_key(self):
        return f'{super().cache_key()}_epsilon_{self.epsilon}'

    @tf.function
    def attack_datatype(self, x, y):
        """

        :param images: batch of rgb images
        :param labels: batch of one-hot labels
        :return:
        """
        return self._jpeg_compression(self._attack(x, y, self.epsilon))

    @tf.function
    def _attack(self, x, y, epsilon):
        perturbed_x = self.gradient_sign_perturbation(x, y, epsilon)
        perturbed_x = RGBAdversarialAttack.l_inf_clip(perturbed_x, x, epsilon)
        return perturbed_x


class RGBBIM(RGBFGSM):
    """
    Basic Iterative Method (BIM).
    Source: Kurakin et al., Adversarial Examples In The Physical World, ILCR 2017.
    """
    _attack_name = 'bim'

    def __init__(self, dataset, model_name, loss_fn='crossentropy', epsilon: float = 8., alpha: float = None,
                 T: int = 10, model=None, random_init=False, output_jpeg_quality=None,
                 output_chroma_subsampling: bool = True):
        """
        :param alpha: step size. if None, alpha=epsilon/T.
        :param T: number of iterations
        :param epsilon: the perturbation bound
        :param random_init: whether the images will be randomly initialized at the start of the attack.
               Usually used for adversarial training.
        :param dataset: the name of the dataset
        :param model_name: the name of the model. If model is not None, model_name will be ignored.
        :param loss_fn: crossentropy or confidence.
        :param model: The model. If model is not None, model_name will be ignored.
                      This is mainly used for adversarial training.
                      It prevents the model from being loaded twice (once for training, once for attacking)
                      such that the images are created on the model that is being trained.
        :param output_jpeg_quality: The jpeg quality, the output is compressed to. None for no compression.
        :param output_chroma_subsampling: Whether to use chroma subsampling when compressing the output.
               only used if output_jpeg_quality is not None.
               Deactivated in the paper's experiments.
        """
        alpha = default_alpha(alpha, epsilon, T)

        super().__init__(dataset, model_name, epsilon=epsilon, loss_fn=loss_fn, model=model,
                         output_chroma_subsampling=output_chroma_subsampling, output_jpeg_quality=output_jpeg_quality)
        self.alpha = alpha
        self.T = T
        self._random_init = random_init

    def cache_key(self):
        return f'{super().cache_key()}_alpha_{self.alpha}_T_{self.T}_random_init_{self._random_init}'

    @tf.function
    def attack_datatype(self, images, labels):
        """

        :param images: batch of rgb images
        :param labels: batch of one-hot labels
        :return:
        """
        return self._jpeg_compression(self._attack(images, labels, self.epsilon, self.alpha, self.T))

    @tf.function
    def _random_linf_init(self, images, epsilon):
        if self._random_init:
            x_perturbed = images + tf.random.uniform(tf.shape(images), minval=-epsilon, maxval=epsilon)
            return RGBAdversarialAttack.l_inf_clip(x_perturbed, images, epsilon)
        return images

    @tf.function
    def _attack(self, images, labels, epsilon, alpha, T):
        """

        :param images: batch of rgb images
        :param labels: batch of one-hot labels
        :return:
        """
        x_perturbed = tf.identity(images)

        # random initialization
        x_perturbed = self._random_linf_init(x_perturbed, epsilon)

        for t in range(T):
            # perturbation by alpha in the gradient's direction
            x_perturbed = self.gradient_sign_perturbation(x_perturbed, labels, alpha)
            # clip the images to be within the linf ball
            x_perturbed = RGBAdversarialAttack.l_inf_clip(x_perturbed, images, epsilon)

        return x_perturbed
