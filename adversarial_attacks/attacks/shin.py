from abc import ABC, abstractmethod
from typing import List

import tensorflow as tf

import adversarial_attacks.utils.transformation
from adversarial_attacks.attacks.attacks import default_alpha
from adversarial_attacks.attacks.rgb import RGBAdversarialAttack


class ShinAttack(RGBAdversarialAttack, ABC):
    """
    Abstract Class for Shin & Song's attacks:

    Jpeg-resistant adversarial images, NIPS 2017 Workshop on Machine Learning and Computer Security.

    """
    _model_type = 'rgb'

    @abstractmethod
    def __init__(self, dataset, model_name, loss_fn='crossentropy', jpeg_quality: int = 100,
                 chroma_subsampling: bool = True,
                 model=None, output_jpeg_quality=None,
                 output_chroma_subsampling: bool = True):
        """
        :param chroma_subsampling: whether chroma subsampling should be used internally when JPEG compression (approx)
               is included into the target model
        :param jpeg_quality: the jpeg quality to use internally when JPEG compression (approx)
               is included into the target model
        :param output_jpeg_quality: The jpeg quality, the output is compressed to. None for no compression.
        :param output_chroma_subsampling: Whether to use chroma subsampling when compressing the output.
               only used if output_jpeg_quality is not None.
               Deactivated in the paper's experiments.
        :param dataset: the name of the dataset
        :param model_name: the name of the model. If model is not None, model_name will be ignored.
        :param loss_fn: crossentropy or confidence.
        :param model: The model. If model is not None, model_name will be ignored.
                      This is mainly used for adversarial training.
                      It prevents the model from being loaded twice (once for training, once for attacking)
                      such that the images are created on the model that is being trained.

        """

        super().__init__(dataset, model_name, loss_fn=loss_fn, model=model,
                         output_chroma_subsampling=output_chroma_subsampling, output_jpeg_quality=output_jpeg_quality)

        self._jpeg_quality = jpeg_quality
        self._chroma_subsammpling = chroma_subsampling

        self._jpeg_compression_approximation = adversarial_attacks.utils.transformation. \
            ShinJpegCompressionApproximationForRGBImages(self._dataset, jpeg_quality, chroma_subsampling)

    def cache_key(self):
        return f'shin_{super().cache_key()}_jq_{self._jpeg_quality}_cs_{self._chroma_subsammpling}'

    @tf.function
    def gradient(self, image, label, ret_loss=False):
        """
        Computes the gradient. Includes an approximation of JPEG compression into the source model.
        :param image:
        :param label:
        :param ret_loss:
        :return:
        """
        image = tf.cast(image, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(image)
            prediction = self.predict(self._jpeg_compression_approximation(image),  # jpeg compression
                                      logits_or_softmax='logits')
            loss = self._loss_fn(label, prediction)
        grad = tape.gradient(loss, image)
        grad = tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad)
        if not ret_loss:
            return grad
        else:
            return grad, loss


class ShinFGSM(ShinAttack):
    """
    Fast gradient sign method. Shin et al. Version.

    Original FGSM source:
    Goodfellow et al.: Explaining and Harnessing Adversarial Examples, ILCR 2015.
    """
    _attack_name = 'shin_fgsm'

    def __init__(self, dataset, model_name, loss_fn='crossentropy', epsilon: float = 8,
                 jpeg_quality: int = 100, model=None, chroma_subsampling: bool = True, output_jpeg_quality=None,
                 output_chroma_subsampling: bool = True):
        """
        :param epsilon: the rgb perturbation bound
        :param chroma_subsampling: whether chroma subsampling should be used internally when JPEG compression (approx)
               is included into the target model
        :param jpeg_quality: the jpeg quality to use internally when JPEG compression (approx)
               is included into the target model
        :param output_jpeg_quality: The jpeg quality, the output is compressed to. None for no compression.
        :param output_chroma_subsampling: Whether to use chroma subsampling when compressing the output.
               only used if output_jpeg_quality is not None.
               Deactivated in the paper's experiments.
        :param dataset: the name of the dataset
        :param model_name: the name of the model. If model is not None, model_name will be ignored.
        :param loss_fn: crossentropy or confidence.
        :param model: The model. If model is not None, model_name will be ignored.
                      This is mainly used for adversarial training.
                      It prevents the model from being loaded twice (once for training, once for attacking)
                      such that the images are created on the model that is being trained.

        """
        super(ShinFGSM, self).__init__(dataset, model_name, loss_fn=loss_fn, jpeg_quality=jpeg_quality, model=model,
                                       chroma_subsampling=chroma_subsampling,
                                       output_chroma_subsampling=output_chroma_subsampling,
                                       output_jpeg_quality=output_jpeg_quality)
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
        perturbed_x = self.gradient_sign_perturbation(x, y, self.epsilon)
        perturbed_x = RGBAdversarialAttack.l_inf_clip(perturbed_x, x, self.epsilon)
        return self._jpeg_compression(perturbed_x)


class ShinBIM(ShinFGSM):
    """
    Basic Iterative Method (BIM). Shin et al. Version.
    Source: Kurakin et al., Adversarial Examples In The Physical World, ILCR 2017.

    :returns: possible adversarial examples in the same shape as input x
    """
    _attack_name = 'shin_bim'

    def __init__(self, dataset, model_name, loss_fn='crossentropy', epsilon: float = 8, alpha: float = None,
                 T: int = 10, jpeg_quality: int = 100, model=None, chroma_subsampling: bool = True,
                 output_jpeg_quality=None,
                 output_chroma_subsampling: bool = True):
        """
        :param epsilon: the rgb perturbation bound
        :param alpha: the rgb step size. epsilon/T if None
        :param T: number of iterations
        :param chroma_subsampling: whether chroma subsampling should be used internally when JPEG compression (approx)
               is included into the target model
        :param jpeg_quality: the jpeg quality to use internally when JPEG compression (approx)
               is included into the target model
        :param output_jpeg_quality: The jpeg quality, the output is compressed to. None for no compression.
        :param output_chroma_subsampling: Whether to use chroma subsampling when compressing the output.
               only used if output_jpeg_quality is not None.
               Deactivated in the paper's experiments.
        :param dataset: the name of the dataset
        :param model_name: the name of the model. If model is not None, model_name will be ignored.
        :param loss_fn: crossentropy or confidence.
        :param model: The model. If model is not None, model_name will be ignored.
                      This is mainly used for adversarial training.
                      It prevents the model from being loaded twice (once for training, once for attacking)
                      such that the images are created on the model that is being trained.

        """
        alpha = default_alpha(alpha, epsilon, T)
        super().__init__(dataset, model_name, epsilon=epsilon, loss_fn=loss_fn, jpeg_quality=jpeg_quality, model=model,
                         chroma_subsampling=chroma_subsampling, output_chroma_subsampling=output_chroma_subsampling,
                         output_jpeg_quality=output_jpeg_quality)

        self.alpha = alpha
        self.T = T

    def cache_key(self):
        return f'{super().cache_key()}_alpha_{self.alpha}_T_{self.T}'

    @tf.function
    def attack_datatype(self, images, labels):
        """

        :param images: batch of rgb images
        :param labels: batch of one-hot labels
        :return:
        """
        return self._attack(images, labels, self.epsilon, self.alpha, self.T)

    def _attack(self, images, labels, epsilon, alpha, T):
        x_perturbed = images
        for t in range(T):
            x_perturbed = self.gradient_sign_perturbation(x_perturbed, labels, alpha)
            x_perturbed = RGBAdversarialAttack.l_inf_clip(x_perturbed, images, epsilon)

        x_perturbed = RGBAdversarialAttack.l_inf_clip(x_perturbed, images, epsilon)
        return self._jpeg_compression(x_perturbed)


class ShinBIMEnsemble(ShinBIM):
    """
    BIM Ensemble attack that is used in the paper's experiments.

    See Shin & Song,
    Jpeg-resistant adversarial images, NIPS 2017 Workshop on Machine Learning and Computer Security
    for an explanation on how the gradients for every JPEG quality are combined.
    """
    _attack_name = 'shin_ensemble_bim'

    def __init__(self, dataset, model_name, loss_fn='crossentropy', epsilon: float = 8, alpha: float = None,
                 T: int = 10, jpeg_qualities=None, model=None, chroma_subsampling: bool = True,
                 output_jpeg_quality=None,
                 output_chroma_subsampling: bool = True):
        """
        :param epsilon: the rgb perturbation bound
        :param alpha: the rgb step size. epsilon/T if None
        :param T: number of iterations
        :param chroma_subsampling: whether chroma subsampling should be used internally when JPEG compression (approx)
               is included into the target model
        :param jpeg_qualities: ensemble of jpeg qualities as a list
        :param output_jpeg_quality: The jpeg quality, the output is compressed to. None for no compression.
        :param output_chroma_subsampling: Whether to use chroma subsampling when compressing the output.
               only used if output_jpeg_quality is not None.
               Deactivated in the paper's experiments.
        :param dataset: the name of the dataset
        :param model_name: the name of the model. If model is not None, model_name will be ignored.
        :param loss_fn: crossentropy or confidence.
        :param model: The model. If model is not None, model_name will be ignored.
                      This is mainly used for adversarial training.
                      It prevents the model from being loaded twice (once for training, once for attacking)
                      such that the images are created on the model that is being trained.

        """
        super().__init__(dataset, model_name, epsilon=epsilon, alpha=alpha, T=T, loss_fn=loss_fn,
                         jpeg_quality=100, model=model, chroma_subsampling=chroma_subsampling,
                         output_chroma_subsampling=output_chroma_subsampling, output_jpeg_quality=output_jpeg_quality)
        if jpeg_qualities is None:
            jpeg_qualities = [90, 75, 50]
        self._ensemble_jpeg_qualities = jpeg_qualities

        # dict for the ensemble of jpeg qualities
        self._jpeg_compression_approximation = {jq: adversarial_attacks.utils.transformation. \
            ShinJpegCompressionApproximationForRGBImages(self._dataset, jq, chroma_subsampling) for jq in
                                                self._ensemble_jpeg_qualities}

    @tf.function
    def gradient(self, image, label, ret_loss=False):
        """
        Combines the gradients from every JPEG quality from the ensemble of JPEG qualities as
        described in the original paper.
        :param image:
        :param label:
        :param ret_loss:
        :return:
        """
        if ret_loss:
            raise ValueError('ret_loss not available')

        gradients = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        losses = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        # collect gradients and losses for each JPEG quality
        for i in range(len(self._ensemble_jpeg_qualities)):
            jpeg_quality = self._ensemble_jpeg_qualities[i]
            image = tf.cast(image, tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(image)
                prediction = self.predict(self._jpeg_compression_approximation[jpeg_quality](image),
                                          logits_or_softmax='logits')
                loss = self._loss_fn(label, prediction)
            grad = tape.gradient(loss, image)
            grad = tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad)

            gradients = gradients.write(i, grad)
            losses = losses.write(i, loss)

        # stack gradients, losses to tensors and compute combined gradients
        gradients, losses = gradients.stack(), losses.stack()
        exp_losses = tf.math.minimum(tf.math.exp(losses), tf.float32.max)
        sum_exp_losses = tf.math.minimum(tf.math.reduce_sum(exp_losses, axis=0, keepdims=True), tf.float32.max)

        divide = tf.math.divide(exp_losses, sum_exp_losses)
        divide = tf.reshape(divide, [len(self._ensemble_jpeg_qualities), tf.shape(image)[0], 1, 1, 1])

        mult = tf.math.multiply(1. - divide, gradients)
        out = tf.math.reduce_sum(mult, axis=0)

        return out

    def cache_key(self):
        return f'{super().cache_key()}_ensemble_jq_{self._ensemble_jpeg_qualities}'
