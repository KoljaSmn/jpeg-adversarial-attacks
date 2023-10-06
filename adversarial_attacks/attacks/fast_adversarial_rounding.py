import tensorflow as tf
import tensorflow_probability as tfp

import adversarial_attacks.utils.jpeg


@tf.function
def convert_eta(eta_in):
    return int((1. - eta_in) * 100)


class FastAdversarialRounding:
    """
    This class implements the fast adversarial rounding from
    Shi et al.: On generating JPEG adversarial images, 2021.
    """

    def __init__(self, eta, jpeg_quality, loss_fn_for_image_label_input):
        # both will be set in inheriting classes
        self.eta = eta
        self.jpeg_quality = jpeg_quality
        self._loss_fn_for_image_label_input = loss_fn_for_image_label_input

    @tf.function
    def _first_rounding_step(self, jpeg_adversarial_unrounded, Y_grad, Cb_grad, Cr_grad):
        """
        The first rounding step.
        If the gradient's direction is the same as the nearest integer,
        the coefficients will be rounded to that integer.
        :param jpeg_adversarial_unrounded:
        :param Y_grad:
        :param Cb_grad:
        :param Cr_grad:
        :return:
        """
        Y, Cb, Cr = jpeg_adversarial_unrounded[0], jpeg_adversarial_unrounded[1], jpeg_adversarial_unrounded[2]
        Y_nearest, Cb_nearest, Cr_nearest = tf.math.round(Y), tf.math.round(Cb), tf.math.round(Cr)

        Y_first_step_rounding = tf.where(tf.math.logical_or(
            tf.math.logical_and(
                tf.math.greater_equal(Y_nearest, Y),
                tf.math.greater_equal(Y_grad, 0.)
            ),
            tf.math.logical_and(
                tf.math.less_equal(Y_nearest, Y),
                tf.math.less_equal(Y_grad, 0.)
            )
        ),
            Y_nearest, Y)

        Cb_first_step_rounding = tf.where(tf.math.logical_or(
            tf.math.logical_and(
                tf.math.greater_equal(Cb_nearest, Cb),
                tf.math.greater_equal(Cb_grad, 0.)
            ),
            tf.math.logical_and(
                tf.math.less_equal(Cb_nearest, Cb),
                tf.math.less_equal(Cb_grad, 0.)
            )
        ),
            Cb_nearest, Cb)

        Cr_first_step_rounding = tf.where(tf.math.logical_or(
            tf.math.logical_and(
                tf.math.greater_equal(Cr_nearest, Cr),
                tf.math.greater_equal(Cr_grad, 0.)
            ),
            tf.math.logical_and(
                tf.math.less_equal(Cr_nearest, Cr),
                tf.math.less_equal(Cr_grad, 0.)
            )
        ),
            Cr_nearest, Cr)

        return Y_first_step_rounding, Cb_first_step_rounding, Cr_first_step_rounding

    @tf.function
    def _jpeg_prediction_func(self, jpeg_adversarial):
        """
        Function that predicts the classes of jpeg images.
        """
        pass

    @tf.function
    def _fast_adversarial_rounding_gradients(self, jpeg_adversarial, labels):
        """
        Computes the gradients of quantized jpeg coefficients.
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([jpeg_adversarial[0], jpeg_adversarial[1], jpeg_adversarial[2]])
            loss = self._loss_fn_for_image_label_input(jpeg_adversarial, labels)

        # compute the gradients
        Y_grad, Cb_grad, Cr_grad = tape.gradient(loss, jpeg_adversarial[0]), \
                                   tape.gradient(loss, jpeg_adversarial[1]), \
                                   tape.gradient(loss, jpeg_adversarial[2])

        return Y_grad, Cb_grad, Cr_grad

    @tf.function
    def _second_rounding_step(self, Y_first_step_rounding, Cb_first_step_rounding, Cr_first_step_rounding,
                              Y_grad, Cb_grad, Cr_grad):
        """
        The second rounding step uses an importance
        indicator that measures the importance of coefficients being rounded
        in the gradient's direction.
        A certain number of coefficients (determined by eta) will be rounded
        in this direction.

        :param Y_first_step_rounding:
        :param Cb_first_step_rounding:
        :param Cr_first_step_rounding:
        :param Y_grad:
        :param Cb_grad:
        :param Cr_grad:
        :return:
        """
        Y_d_diff = tf.math.subtract(tf.math.subtract(tf.math.ceil(Y_first_step_rounding), Y_first_step_rounding),
                                    tf.math.subtract(Y_first_step_rounding, tf.math.floor(Y_first_step_rounding)))
        Cb_d_diff = tf.math.subtract(tf.math.subtract(tf.math.ceil(Cb_first_step_rounding), Cb_first_step_rounding),
                                     tf.math.subtract(Cb_first_step_rounding,
                                                      tf.math.floor(Cb_first_step_rounding)))
        Cr_d_diff = tf.math.subtract(tf.math.subtract(tf.math.ceil(Cr_first_step_rounding), Cr_first_step_rounding),
                                     tf.math.subtract(Cr_first_step_rounding,
                                                      tf.math.floor(Cr_first_step_rounding)))

        dequantized_differences = adversarial_attacks.utils.jpeg.dequantize_zigzagged_coefficients(Y_d_diff,
                                                                                                   Cb_d_diff,
                                                                                                   Cr_d_diff,
                                                                                                   jpeg_quality=
                                                                                                   self.jpeg_quality,
                                                                                                   batched=True,
                                                                                                   round=None)
        # for those coefficients that have been rounded in the first step, the denominator is zero
        # therefore, they will not be selected
        Y_tau = tf.math.divide_no_nan(tf.math.abs(Y_grad), tf.math.pow(dequantized_differences[0], 2))
        Cb_tau = tf.math.divide_no_nan(tf.math.abs(Cb_grad), tf.math.pow(dequantized_differences[1], 2))
        Cr_tau = tf.math.divide_no_nan(tf.math.abs(Cr_grad), tf.math.pow(dequantized_differences[2], 2))

        # computes percentile resulting from eta
        # eta=0.05 means that every coefficients with an importance value
        # that is higher than the 95% percentile is rounded in the gradient's direction while the others are rounded to the nearest integer
        eta_percentile = convert_eta(self.eta)
        Y_percentile = tfp.stats.percentile(Y_tau, eta_percentile, axis=[1, 2, 3], keepdims=True)
        Cb_percentile = tfp.stats.percentile(Cb_tau, eta_percentile, axis=[1, 2, 3], keepdims=True)
        Cr_percentile = tfp.stats.percentile(Cr_tau, eta_percentile, axis=[1, 2, 3], keepdims=True)

        Y_second_rounding_step = tf.where(tf.math.greater(Y_tau, Y_percentile),  # if coefficient > computed_percentile
                                          tf.where(tf.math.greater_equal(Y_grad, 0),  # round in the gradient direction
                                                   tf.math.ceil(Y_first_step_rounding),
                                                   tf.math.floor(Y_first_step_rounding)),
                                          tf.math.round(Y_first_step_rounding))  # else: round to the nearest integer

        Cb_second_rounding_step = tf.where(
            tf.math.greater(Cb_tau, Cb_percentile),
            tf.where(tf.math.greater_equal(Cb_grad, 0),
                     tf.math.ceil(Cb_first_step_rounding),
                     tf.math.floor(Cb_first_step_rounding)),
            tf.math.round(Cb_first_step_rounding))

        Cr_second_rounding_step = tf.where(
            tf.math.greater(Cr_tau, Cr_percentile),
            tf.where(tf.math.greater_equal(Cr_grad, 0),
                     tf.math.ceil(Cr_first_step_rounding),
                     tf.math.floor(Cr_first_step_rounding)),
            tf.math.round(Cr_first_step_rounding))

        return Y_second_rounding_step, Cb_second_rounding_step, Cr_second_rounding_step

    @tf.function
    def _fast_adversarial_rounding(self, jpeg_adversarial_unrounded, labels):
        """
        This function combines both rounding steps to implement the fast adversarial rounding.
        :param jpeg_adversarial_unrounded:
        :param labels:
        :return:
        """
        Y_grad, Cb_grad, Cr_grad = self._fast_adversarial_rounding_gradients(jpeg_adversarial_unrounded, labels)

        Y_first_step_rounding, \
        Cb_first_step_rounding, \
        Cr_first_step_rounding = self._first_rounding_step(jpeg_adversarial_unrounded, Y_grad, Cb_grad, Cr_grad)

        first_step_rounding_images = Y_first_step_rounding, Cb_first_step_rounding, Cr_first_step_rounding
        Y_grad, Cb_grad, Cr_grad = self._fast_adversarial_rounding_gradients(first_step_rounding_images, labels)
        return self._second_rounding_step(Y_first_step_rounding, Cb_first_step_rounding, Cr_first_step_rounding,
                                          Y_grad, Cb_grad, Cr_grad)

    @tf.function
    def __call__(self, jpeg_adversarial_unrounded, labels):
        return self._fast_adversarial_rounding(jpeg_adversarial_unrounded, labels)
