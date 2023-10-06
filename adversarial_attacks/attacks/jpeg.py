from abc import ABC, abstractmethod

import tensorflow as tf
import tensorflow_probability as tfp

import adversarial_attacks.attacks.fast_adversarial_rounding
import adversarial_attacks.utils.transformation
import adversarial_attacks.utils.jpeg
import adversarial_attacks.utils.attacks
from adversarial_attacks.attacks.attacks import Attack, ETA_DEFAULT, JPEG_CLIP_MIN, JPEG_CLIP_MAX, \
    ycbcr_round, ycbcr_clip_and_cast, frequency_wise_epsilon_alpha, default_alpha


def _check_lambda(lambda_values, lambda_name):
    """
    Checks whether a lambda (frequency weighting) vector is valid.
    Throws an exception if not.
    """
    if tf.math.reduce_min(lambda_values) < 0. or tf.math.reduce_max(lambda_values) > 1.:
        raise ValueError(f'Invalid lambda value for channel {lambda_name}. Values must be in [0, 1].')


class JpegAdversarialAttack(Attack, ABC):
    """
    Abstract Attack class for JPEG attacks.
    """

    _model_type = 'jpeg'

    @abstractmethod
    def __init__(self, dataset, model_name, loss_fn='crossentropy', jpeg_quality: int = 100,
                 lambda_Y=1., lambda_Cb=1., lambda_Cr=1.,
                 fast_adversarial_rounding: bool = False, eta: float = ETA_DEFAULT, chroma_subsampling: bool = True,
                 model=None):
        """
        :param dataset: the name of the dataset
        :param model_name: the name of the model. If model is not None, model_name will be ignored.
        :param loss_fn: crossentropy or confidence.
        :param model: The model. If model is not None, model_name will be ignored.
                      This is mainly used for adversarial training.
                      It prevents the model from being loaded twice (once for training, once for attacking)
                      such that the images are created on the model that is being trained.
        :param jpeg_quality: The JPEG quality used in the attack. The output will also be of the given JPEG quality.
        :param lambda_Y: frequency weighting vector for the Y channel.
        :param lambda_Cb: frequency weighting vector for the Cb channel.
        :param lambda_Cr: frequency weighting vector for the Cr channel.
        :param fast_adversarial_rounding: whether to use fast adversarial rounding
               instead of rounding every coefficient to the nearest integer at the end of the attack.
        :param eta: The eta for the fast adversarial rounding.
        :param chroma_subsampling: Whether to use chroma subsampling in the attack.
               Deactivated in the experiments in the paper.
        """
        super().__init__(dataset=dataset, model_name=model_name, loss_fn=loss_fn, model=model)

        self.jpeg_quality = jpeg_quality
        self.chroma_subsampling = chroma_subsampling

        # determines whether gradients for each channel are to be computed
        # will be set in subclasses
        self._compute_Y_grad = True
        self._compute_Cb_grad = True
        self._compute_Cr_grad = True

        self._fix_zero_coefficients = True
        self._frequency_gradients_norm = None
        self.quantized = True

        _check_lambda(lambda_Y, 'Y'), _check_lambda(lambda_Cb, 'Cb'), _check_lambda(lambda_Cr, 'Cr')

        self.lambda_Y, self.lambda_Cb, self.lambda_Cr = lambda_Y, lambda_Cb, lambda_Cr

        # loads a model that converts rgb data to quantized JPEG data of quality self.jpeg_quality
        self._rgb_to_jpeg_model = adversarial_attacks.utils.transformation.get_rgb_to_jpeg_model(self._dataset,
                                                                                                 self.jpeg_quality,
                                                                                                 round='round' if self.quantized else None,
                                                                                                 chroma_subsampling=
                                                                                                 self.chroma_subsampling)

        # loads a model that converts JPEG data to unquantized RGB data
        self._jpeg_to_rgb_model = adversarial_attacks.utils.transformation.get_jpeg_to_rgb_model(self._dataset,
                                                                                                 self.jpeg_quality,
                                                                                                 round=False,
                                                                                                 chroma_subsampling=self.chroma_subsampling)

        self.use_fast_adversarial_rounding = fast_adversarial_rounding
        self.eta = eta

        self._prediction_func = lambda images: self.predict(images,
                                                            logits_or_softmax='logits')

        # init the fast adversarial rounding class if it is used.
        if self.use_fast_adversarial_rounding:
            loss_fn_for_image_label_input = lambda images, label: self._loss_fn(label,
                                                                                self._jpeg_prediction_func(images))
            self._fast_adversarial_rounding = adversarial_attacks.attacks.fast_adversarial_rounding. \
                FastAdversarialRounding(self.eta, self.jpeg_quality, loss_fn_for_image_label_input)
        else:
            self._fast_adversarial_rounding = None

    def predict(self, images, logits_or_softmax='softmax'):
        """
        Returns logits or softmax for an input batch of JPEG data.
        First, the JPEG data is converted to RGB, Then the super.predict method is called on RGB data.

        :param images: JPEG data. Tuple (Y, Cb, Cr)
        :param logits_or_softmax:
        :return: logits or softmax output
        """
        return super().predict(self._jpeg_to_rgb_model(images), logits_or_softmax=logits_or_softmax)

    @abstractmethod
    def cache_key(self):
        key = f'jpeq_{super().cache_key()}_jq_{self.jpeg_quality}_cs_{self.chroma_subsampling}_ly_{self.lambda_Y}_' \
              f'lcb_{self.lambda_Cb}_lcr_{self.lambda_Cr}_far_{self.use_fast_adversarial_rounding}_eta_{self.eta}_' \
              f'oneBlockPerFrequency_{self._frequency_gradients_norm}'
        if not self.quantized:
            key += 'quantized_False'

        if self._fix_zero_coefficients:
            key += 'fix_zero_coefficients_True'

        return key

    @tf.function
    def linf_bounds(self, Y, Cb, Cr, abs_eps_Y, abs_eps_Cb, abs_eps_Cr):
        """
        computes the l_inf bounds for every channel with given epsilon values.
        abs_eps_Y has the same shape as Y and correspondingly for Cb, Cr.

        :param Y:
        :param Cb:
        :param Cr:
        :param abs_eps_Y:
        :param abs_eps_Cb:
        :param abs_eps_Cr:
        :return:
        """
        Y_clip = tf.math.floor(Y - abs_eps_Y), tf.math.ceil(Y + abs_eps_Y)
        Cb_clip = tf.math.floor(Cb - abs_eps_Cb), tf.math.ceil(Cb + abs_eps_Cb)
        Cr_clip = tf.math.floor(Cr - abs_eps_Cr), tf.math.ceil(Cr + abs_eps_Cr)
        return Y_clip, Cb_clip, Cr_clip

    @tf.function
    def _jpeg_prediction_func(self, jpeg_adversarial):
        """
        For JPEG attacks, the _model expects unquantized coefficients.
        Therefore, the quantized coefficients have to be dequantized.
        :param jpeg_adversarial: quantized coefficients.
        :return:
        """
        return self.predict(adversarial_attacks.utils.jpeg.dequantize_zigzagged_coeff_tuple(jpeg_adversarial,
                                                                                            self.jpeg_quality,
                                                                                            True, None))

    @tf.function
    def _get_absolute_perturbation_budgets(self, Y, Cb, Cr, pb_rel_Y, pb_rel_Cb, pb_rel_Cr):
        """
        Computes absolute perturbation budgets.

        :param Y:
        :param Cb:
        :param Cr:
        :param pb_rel_Y:
        :param pb_rel_Cb:
        :param pb_rel_Cr:
        :return:
        """
        if self._fix_zero_coefficients:
            # this is what is used in the paper
            mult_Y, mult_Cb, mult_Cr = tf.math.abs(Y), tf.math.abs(Cb), tf.math.abs(Cr)
        else:
            mult_Y = tf.math.maximum(tf.math.abs(Y), 1)
            mult_Cb = tf.math.maximum(tf.math.abs(Cb), 1)
            mult_Cr = tf.math.maximum(tf.math.abs(Cr), 1)

        abs_pb_Y = pb_rel_Y * mult_Y
        abs_pb_Cb = pb_rel_Cb * mult_Cb
        abs_pb_Cr = pb_rel_Cr * mult_Cr
        return abs_pb_Y, abs_pb_Cb, abs_pb_Cr

    @tf.function
    def fast_adversarial_rounding_or_nearest_integer_round(self, images, labels):
        """
        Rounds the image at the end of the attack.
        Either to the nearest integer or using the fast adversarial rounding.

        :param images:
        :param labels:
        :return:
        """
        if self.quantized:
            if self.use_fast_adversarial_rounding:
                Y, Cb, Cr = self._fast_adversarial_rounding(images, labels)
            else:
                Y, Cb, Cr = images[0], images[1], images[2]
            # clip the image onto the valid range and round it to integer values
            # the clipping step should be done even if fast adversarial rounding is enabled
            return ycbcr_round(Y, Cb, Cr)
        return images

    def l_inf_and_channel_bounds_clip(self, images, Y_clip, Cb_clip, Cr_clip):
        return ycbcr_clip_and_cast(images[0], images[1], images[2], JPEG_CLIP_MIN, JPEG_CLIP_MAX,
                                   Y_clip, Cb_clip, Cr_clip,
                                   round=False)

    @tf.function
    def clip_and_round(self, images, labels, Y_clip, Cb_clip, Cr_clip):
        """
        Clips the coefficients to be within the JPEG bounds and then rounds the coefficients.
        :param images:
        :param labels:
        :param Y_clip:
        :param Cb_clip:
        :param Cr_clip:
        :return:
        """
        images = self.l_inf_and_channel_bounds_clip(images, Y_clip, Cb_clip, Cr_clip)
        images = self.fast_adversarial_rounding_or_nearest_integer_round(images, labels)
        return tuple((images[0], images[1], images[2]))

    def _norm_frequency_grads(self, Y_grad, Cb_grad, Cr_grad):
        """
        Allows the gradients to be normed such that, e.g. just one block has non-zero gradients for each frequency.
        Was NOT used in the paper. (self._frequency_gradients_norm=None)

        :param Y_grad:
        :param Cb_grad:
        :param Cr_grad:
        :return:
        """
        if self._frequency_gradients_norm == 'one_block_per_frequency':
            def norm_grads(grad):
                return tf.where(grad == tf.transpose(
                    tf.math.reduce_max(
                        tf.math.reduce_max(tf.transpose(tf.math.abs(grad), [0, 3, 1, 2]), axis=3, keepdims=True),
                        axis=2,
                        keepdims=True), [0, 2, 3, 1]),
                                grad,
                                0
                                )

            return norm_grads(Y_grad), norm_grads(Cb_grad), norm_grads(Cr_grad)
        elif self._frequency_gradients_norm == 'mean_over_blocks_per_frequency':
            def norm_grads(grad):
                return tf.zeros_like(grad) + tf.math.reduce_mean(tf.math.reduce_mean(grad, axis=1, keepdims=True),
                                                                 axis=2, keepdims=True)

            return norm_grads(Y_grad), norm_grads(Cb_grad), norm_grads(Cr_grad)
        elif self._frequency_gradients_norm == 'one_block':
            def norm_grads(grad):
                frequency_sum = tf.math.reduce_sum(tf.math.abs(grad), axis=3, keepdims=True)
                return \
                    tf.where(frequency_sum == tf.math.reduce_max(frequency_sum, axis=[1, 2], keepdims=True), grad, 0)[0]

            return norm_grads(Y_grad), norm_grads(Cb_grad), norm_grads(Cr_grad)

        elif self._frequency_gradients_norm == 'half_of_the_blocks':
            def norm_grads(grad):
                return tf.where(grad >= tfp.stats.percentile(grad, 50, axis=[1, 2], keepdims=True), grad, 0)

            return norm_grads(Y_grad), norm_grads(Cb_grad), norm_grads(Cr_grad)

        elif self._frequency_gradients_norm == 'max_direction_or_zero':
            def norm_grads(grad):
                """
                Every gradient that is not directed in the same direction as the one with the highest abs value for the same frequency is set to zero.
                """
                abs_a = tf.math.abs(grad)
                return tf.where(tf.sign(grad) == tf.math.reduce_sum(
                    tf.where(tf.math.abs(grad) == tf.math.reduce_max(abs_a, axis=[1, 2], keepdims=True), tf.sign(grad),
                             0.),
                    axis=[1, 2], keepdims=True), grad, 0)

            return norm_grads(Y_grad), norm_grads(Cb_grad), norm_grads(Cr_grad)
        elif self._frequency_gradients_norm == 'use_max_abs_grad_per_frequency':
            def norm_grads(grad):
                """
                Every gradient that is not directed in the same direction as the one with the highest abs value for the same frequency is set to zero.
                """
                return tf.zeros_like(grad) + tf.math.reduce_sum(
                    tf.where(tf.math.reduce_max(tf.math.abs(grad), axis=[1, 2], keepdims=True) == tf.math.abs(grad),
                             grad, 0), axis=[1, 2], keepdims=True)

            return norm_grads(Y_grad), norm_grads(Cb_grad), norm_grads(Cr_grad)
        elif self._frequency_gradients_norm is None:
            return Y_grad, Cb_grad, Cr_grad
        raise ValueError(f'frequency gradients norm {self._frequency_gradients_norm} not available.')

    def _get_grads(self, tape, loss, images):
        """
        Given a tf.GradientTape and a loss object, and the input images,
        it returns the computed gradients for each channel.

        If gradients for a channel do not have to be computed (because not perturbation is made on the channel anyway),
        it returns a tensor of zeros.

        :param tape:
        :param loss:
        :param images:
        :return:
        """
        Y_grad, Cb_grad, Cr_grad = tf.zeros_like(images[0]), tf.zeros_like(images[1]), tf.zeros_like(images[2])

        if self._compute_Y_grad:
            Y_grad = tape.gradient(loss, images[0])
        if self._compute_Cb_grad:
            Cb_grad = tape.gradient(loss, images[1])
        if self._compute_Cr_grad:
            Cr_grad = tape.gradient(loss, images[2])

        return Y_grad, Cb_grad, Cr_grad

    @tf.function
    def gradient(self, images, labels, ret_loss=False):
        """
        returns the gradient for the three channels.

        :param images: batch of images (jpeg coefficients)
        :param labels: batch of labels
        :returns the gradient of the loss function for Y, Cb, Cr
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([images[0], images[1], images[2]])

            # predict the batch
            prediction = self._prediction_func(images)
            loss = self._loss_fn(labels, prediction)  # compute the loss

        # compute the gradients
        Y_grad, Cb_grad, Cr_grad = self._get_grads(tape, loss, images)

        Y_grad = tf.where(tf.math.is_nan(Y_grad), tf.random.normal(tf.shape(Y_grad)), Y_grad)
        Cb_grad = tf.where(tf.math.is_nan(Cb_grad), tf.random.normal(tf.shape(Cb_grad)), Cb_grad)
        Cr_grad = tf.where(tf.math.is_nan(Cr_grad), tf.random.normal(tf.shape(Cr_grad)), Cr_grad)

        Y_grad, Cb_grad, Cr_grad = self._norm_frequency_grads(Y_grad, Cb_grad, Cr_grad)

        if ret_loss:
            return (Y_grad, Cb_grad, Cr_grad), loss

        return Y_grad, Cb_grad, Cr_grad

    @tf.function
    def gradient_sign_perturbation(self, images, labels, pert_Y, pert_Cb, pert_Cr, round: bool = True):
        """
        Perturbes a batch of images by pert_Y, pert_Cb, pert_Cr in the gradient's direction.
        
        :param images: (Y, Cb, Cr) tuple
        :param labels:
        :param pert_Y: Should have the same shape as Y
        :param pert_Cb:
        :param pert_Cr:
        :param round: whether the coefficients should be rounded after the perturbation.
        :return:
        """
        Y_gradient_sign, Cb_gradient_sign, Cr_gradient_sign = self.gradient_sign(images, labels)
        # compute the coefficients gradients

        # perturbation
        Y = images[0] + Y_gradient_sign * pert_Y
        Cb = images[1] + Cb_gradient_sign * pert_Cb
        Cr = images[2] + Cr_gradient_sign * pert_Cr

        return ycbcr_clip_and_cast(Y, Cb, Cr, JPEG_CLIP_MIN, JPEG_CLIP_MAX, round=round)

    @tf.function
    def gradient_sign(self, images, labels):
        """
        Returns the sign of the gradients.
        
        :param images: batch of images
        :param labels: batch of labels
        :returns the sign of the gradient of the loss function of the image and the _model.
                 for Y, Cb, Cr
        """
        gradient = self.gradient(images, labels)
        return tf.sign(gradient[0]), tf.sign(gradient[1]), tf.sign(gradient[2])

    @tf.function
    def rgb_to_input_datatype_conversion(self, images):
        return self._rgb_to_jpeg_model(images)

    @tf.function
    def rgb_to_output_datatype_conversion(self, images):
        return self._rgb_to_jpeg_model(images)

    @tf.function
    def input_datatype_to_rgb_conversion(self, images):
        return self._jpeg_to_rgb_model(images)

    @tf.function
    def output_datatype_to_rgb_conversion(self, images):
        return self._jpeg_to_rgb_model(images)


class JpegFGSM(JpegAdversarialAttack):
    """
    FGSM Attack on JPEG coefficients.

    Original FGSM Paper: Goodfellow et al., Explaining and Harnessing Adversarial Examples, ILCR 2015
    """
    _attack_name = 'fgsm'

    def __init__(self, dataset, model_name, loss_fn='crossentropy',
                 jpeg_quality: int = 100,
                 eps_Y: float = 1, eps_Cb: float = 1, eps_Cr: float = 1,
                 lambda_Y=1., lambda_Cb=1., lambda_Cr=1.,
                 fast_adversarial_rounding: bool = False, eta: float = ETA_DEFAULT, chroma_subsampling: bool = True,
                 model=None
                 ):
        """
        :param eps_Y: relative perturbation bound for Y (will be multiplied with lambda_Y)
        :param eps_Cb: relative perturbation bound for Cb (will be multiplied with lambda_Cb)
        :param eps_Cr: relative perturbation bound for Cr (will be multiplied with lambda_Cr)
        :param dataset: the name of the dataset
        :param model_name: the name of the model. If model is not None, model_name will be ignored.
        :param loss_fn: crossentropy or confidence.
        :param model: The model. If model is not None, model_name will be ignored.
                      This is mainly used for adversarial training.
                      It prevents the model from being loaded twice (once for training, once for attacking)
                      such that the images are created on the model that is being trained.
        :param jpeg_quality: The JPEG quality used in the attack. The output will also be of the given JPEG quality.
        :param lambda_Y: frequency weighting vector for the Y channel.
        :param lambda_Cb: frequency weighting vector for the Cb channel.
        :param lambda_Cr: frequency weighting vector for the Cr channel.
        :param fast_adversarial_rounding: whether to use fast adversarial rounding
               instead of rounding every coefficient to the nearest integer at the end of the attack.
        :param eta: The eta for the fast adversarial rounding.
        :param chroma_subsampling: Whether to use chroma subsampling in the attack.
               Deactivated in the experiments in the paper.
        """

        super().__init__(dataset, model_name=model_name, loss_fn=loss_fn, jpeg_quality=jpeg_quality,
                         lambda_Y=lambda_Y, lambda_Cb=lambda_Cb, lambda_Cr=lambda_Cr,
                         fast_adversarial_rounding=fast_adversarial_rounding, eta=eta,
                         chroma_subsampling=chroma_subsampling, model=model)
        self.eps_Y = eps_Y
        self.eps_Cb = eps_Cb
        self.eps_Cr = eps_Cr

        # gradients for a channel have to be computed only if the channel will be perturbed
        self._compute_Y_grad = self.eps_Y != 0
        self._compute_Cb_grad = self.eps_Cb != 0
        self._compute_Cr_grad = self.eps_Cr != 0

    def cache_key(self):
        return f'{super().cache_key()}_eps_Y_{self.eps_Y}_eps_Cb_{self.eps_Cb}_eps_Cr_{self.eps_Cr}'

    @tf.function
    def attack_datatype(self, images, labels):
        """

        :param images: batch of rgb images
        :param labels: batch of one-hot labels
        :return:
        """
        return self._attack(images, labels, self.eps_Y, self.eps_Cb, self.eps_Cr)

    @tf.function
    def _attack(self, images, labels, eps_Y, eps_Cb, eps_Cr):
        Y, Cb, Cr = tf.cast(images[0], tf.float32), tf.cast(images[1], tf.float32), tf.cast(images[2], tf.float32)

        # First, multiply epsilon with the frequency weighting vectors.
        eps_Y = frequency_wise_epsilon_alpha(eps_Y, self.lambda_Y)
        eps_Cb = frequency_wise_epsilon_alpha(eps_Cb, self.lambda_Cb)
        eps_Cr = frequency_wise_epsilon_alpha(eps_Cr, self.lambda_Cr)

        # Then, compute absolute epsilon values and the resulting bounds
        abs_eps_Y, abs_eps_Cb, abs_eps_Cr = self._get_absolute_perturbation_budgets(Y, Cb, Cr, eps_Y, eps_Cb, eps_Cr)
        Y_clip, Cb_clip, Cr_clip = self.linf_bounds(Y, Cb, Cr, abs_eps_Y, abs_eps_Cb, abs_eps_Cr)

        images = (Y, Cb, Cr)
        # Perturb the images using the absoulte epsilon values
        images = self.gradient_sign_perturbation(images, labels,
                                                 abs_eps_Y,
                                                 abs_eps_Cb,
                                                 abs_eps_Cr,
                                                 round=False)

        # Clip the coefficients to be within the valid range and round them
        return self.clip_and_round(images, labels, Y_clip, Cb_clip, Cr_clip)


class JpegBIM(JpegFGSM):
    """
    Class for Basic Iterative Method (Jpeg Version).
    Kurakin et al.: Adversarial examples in the physical world, ILCR 2017.
    """
    _attack_name = 'bim'

    def __init__(self, dataset, model_name, loss_fn='crossentropy',
                 jpeg_quality: int = 100,
                 eps_Y: float = 1, eps_Cb: float = 1, eps_Cr: float = 1,
                 alpha_Y: float = None, alpha_Cb: float = None, alpha_Cr: float = None,
                 lambda_Y=1., lambda_Cb=1., lambda_Cr=1.,
                 T: int = 10, fast_adversarial_rounding: bool = False, eta: float = ETA_DEFAULT,
                 chroma_subsampling: bool = True, model=None, random_start: bool = False
                 ):
        """
        :param alpha_Y: relative step size for Y (will be multiplied with lambda_Y)
        :param alpha_Cb: relative step size for Cb (will be multiplied with lambda_Cb)
        :param alpha_Cr: relative step size for Cr (will be multiplied with lambda_Cr)
        :param T: number of iterations
        :param random_start: whether to use a random initialization inside the L_inf ball.
               Usually used for adversarial training only.
        :param eps_Y: relative perturbation bound for Y (will be multiplied with lambda_Y)
        :param eps_Cb: relative perturbation bound for Cb (will be multiplied with lambda_Cb)
        :param eps_Cr: relative perturbation bound for Cr (will be multiplied with lambda_Cr)
        :param dataset: the name of the dataset
        :param model_name: the name of the model. If model is not None, model_name will be ignored.
        :param loss_fn: crossentropy or confidence.
        :param model: The model. If model is not None, model_name will be ignored.
                      This is mainly used for adversarial training.
                      It prevents the model from being loaded twice (once for training, once for attacking)
                      such that the images are created on the model that is being trained.
        :param jpeg_quality: The JPEG quality used in the attack. The output will also be of the given JPEG quality.
        :param lambda_Y: frequency weighting vector for the Y channel.
        :param lambda_Cb: frequency weighting vector for the Cb channel.
        :param lambda_Cr: frequency weighting vector for the Cr channel.
        :param fast_adversarial_rounding: whether to use fast adversarial rounding
               instead of rounding every coefficient to the nearest integer at the end of the attack.
        :param eta: The eta for the fast adversarial rounding.
        :param chroma_subsampling: Whether to use chroma subsampling in the attack.
               Deactivated in the experiments in the paper.
        """
        alpha_Y, alpha_Cb, alpha_Cr = default_alpha(alpha_Y, eps_Y, T), \
                                      default_alpha(alpha_Cb, eps_Cb, T), \
                                      default_alpha(alpha_Cr, eps_Cr, T)
        super().__init__(dataset, model_name=model_name, loss_fn=loss_fn, jpeg_quality=jpeg_quality,
                         eps_Y=eps_Y, eps_Cb=eps_Cb, eps_Cr=eps_Cr,
                         lambda_Y=lambda_Y, lambda_Cb=lambda_Cb, lambda_Cr=lambda_Cr,
                         fast_adversarial_rounding=fast_adversarial_rounding, eta=eta,
                         chroma_subsampling=chroma_subsampling, model=model)

        self.alpha_Y = alpha_Y
        self.alpha_Cr = alpha_Cr
        self.alpha_Cb = alpha_Cb

        self._compute_Y_grad = self.alpha_Y != 0
        self._compute_Cb_grad = self.alpha_Cb != 0
        self._compute_Cr_grad = self.alpha_Cr != 0

        self.T = T
        self._random_start = random_start

    def cache_key(self):
        return f'{super().cache_key()}_alpha_Y_{self.alpha_Y}_alpha_Cb_{self.alpha_Cb}_alpha_Cr_{self.alpha_Cr}_T_{self.T}_random_start_{self._random_start}'

    @tf.function
    def attack_datatype(self, images, labels):
        """

        :param images: batch of rgb images
        :param labels: batch of one-hot labels
        :return:
        """
        eps_Y, eps_Cb, eps_Cr = self.eps_Y, self.eps_Cb, self.eps_Cr
        alpha_Y, alpha_Cb, alpha_Cr = self.alpha_Y, self.alpha_Cb, self.alpha_Cr
        T = self.T

        Y, Cb, Cr = tf.cast(images[0], tf.float32), tf.cast(images[1], tf.float32), tf.cast(images[2], tf.float32)
        images = (Y, Cb, Cr)

        # compute frequency wise epsilon by multiplying eps with the frequency masking vector
        eps_Y = frequency_wise_epsilon_alpha(eps_Y, self.lambda_Y)
        eps_Cb = frequency_wise_epsilon_alpha(eps_Cb, self.lambda_Cb)
        eps_Cr = frequency_wise_epsilon_alpha(eps_Cr, self.lambda_Cr)

        # compute absolute perturbation budgets and resulting bounds
        abs_eps_Y, abs_eps_Cb, abs_eps_Cr = self._get_absolute_perturbation_budgets(Y, Cb, Cr, eps_Y, eps_Cb, eps_Cr)
        Y_clip, Cb_clip, Cr_clip = self.linf_bounds(Y, Cb, Cr, abs_eps_Y, abs_eps_Cb, abs_eps_Cr)

        if self._random_start:
            Y_random_noise = tf.random.uniform(tf.shape(abs_eps_Y), minval=-abs_eps_Y, maxval=abs_eps_Y)
            Cb_random_noise = tf.random.uniform(tf.shape(abs_eps_Cb), minval=-abs_eps_Cb, maxval=abs_eps_Cb)
            Cr_random_noise = tf.random.uniform(tf.shape(abs_eps_Cr), minval=-abs_eps_Cr, maxval=abs_eps_Cr)

            images = self.l_inf_and_channel_bounds_clip(
                (Y + Y_random_noise, Cb + Cb_random_noise, Cr + Cr_random_noise), Y_clip, Cb_clip, Cr_clip)

        # compute frequency wise step size by multiplying alpha with the frequency masking vector
        alpha_Y = frequency_wise_epsilon_alpha(alpha_Y, self.lambda_Y)
        alpha_Cb = frequency_wise_epsilon_alpha(alpha_Cb, self.lambda_Cb)
        alpha_Cr = frequency_wise_epsilon_alpha(alpha_Cr, self.lambda_Cr)

        # compute absolute step sizes
        abs_alpha_Y, abs_alpha_Cb, abs_alpha_Cr = self._get_absolute_perturbation_budgets(Y, Cb, Cr, alpha_Y, alpha_Cb,
                                                                                          alpha_Cr)

        for t in range(T):
            # perturb the coefficients in the gradient's direction
            images = self.gradient_sign_perturbation(images, labels,
                                                     abs_alpha_Y,
                                                     abs_alpha_Cb,
                                                     abs_alpha_Cr,
                                                     round=False)
        # clip the coefficients to be within the linf bounds and round them
        return self.clip_and_round(images, labels, Y_clip, Cb_clip, Cr_clip)
