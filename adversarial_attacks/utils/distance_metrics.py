import numpy as np
import tensorflow as tf

from adversarial_attacks.utils import color as color_utils
from adversarial_attacks.models.lpips import LossNetwork


class SingleValueMetric(tf.keras.metrics.Metric):
    def __init__(self, initial_value, dtype):
        super(SingleValueMetric, self).__init__()
        self._dtype = dtype
        self._initial_value = tf.constant(initial_value)
        self._val = tf.Variable(tf.cast(self._initial_value, self._dtype))

    def reduce_operation(self, values):
        raise NotImplementedError("")

    def update_state(self, values):
        """Accumulates statistics for computing the metric.
        Args:
          values: Per-example value.
        Returns:
          Update op.
        """
        values = tf.cast(values, self._dtype)

        value_opt = self.reduce_operation(values)
        with tf.control_dependencies([value_opt]):
            self._val.assign(self.reduce_operation([value_opt, self._val]))

    def result(self):
        return tf.identity(self._val)

    def reset_state(self):
        self._val = tf.Variable(tf.cast(self._initial_value, self._dtype))


class MinimumMetric(SingleValueMetric):
    def __init__(self, dtype):
        super(MinimumMetric, self).__init__(np.inf, dtype)

    def reduce_operation(self, values):
        return tf.reduce_min(values)


class MaximumMetric(SingleValueMetric):
    def __init__(self, dtype):
        super(MaximumMetric, self).__init__(-np.inf, dtype)

    def reduce_operation(self, values):
        return tf.reduce_max(values)


class ImageDistanceMetric:
    """
    Metric to compute mean, max and min of a list of Distances of images.
    """

    def __init__(self, dtype=tf.float32):
        super().__init__()
        self._dtype = dtype
        # Min, Max have been removed since it only worked when using eager execution
        # self._min_metric = MinimumMetric(dtype)
        # self._max_metric = MaximumMetric(dtype)
        self._mean_metric = tf.keras.metrics.Mean()

    def reset_state(self):
        self._mean_metric.reset_state()
        # self._min_metric.reset_state()
        #self._max_metric.reset_state()

    def update_state(self, distances):
        distances = tf.cast(distances, self._dtype)
        self._mean_metric.update_state(distances)
        # self._min_metric.update_state(distances)
        #self._max_metric.update_state(distances)

    def result(self):
        return {
            #'max': self._max_metric.result().numpy().astype(np.float64),
            'avg': self._mean_metric.result().numpy().astype(np.float64),
            #'min': self._min_metric.result().numpy().astype(np.float64),
        }


class FunctionWrapperMetric(ImageDistanceMetric):
    """
    Metric to compute mean, max and min of a list of some distance value for images.
    See Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
    Image quality assessment: from error visibility to structural similarity.

    A possible color transform must be part of fn.
    """

    def __init__(self, fn,  dtype=tf.float32):
        super().__init__(dtype)
        self._fn = fn

    def __call__(self, original, adversarial):
        return self._fn(original, adversarial)

    def update_state(self, original, adversarial):
        super().update_state(self(original, adversarial))


class SSIMMetric(FunctionWrapperMetric):
    """
    Metric to compute mean, max and min of a list of SSIM values of images.
    See Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
    Image quality assessment: from error visibility to structural similarity.

    A possible color transform must be part of fn.

    Assumes RGB input. -> max_val 255.
    For other color models, the max_val must be adapted.

    We use the negative ssim, such that big distances result in bigger values, as for the l2 and perceptual metrics
    """
    def __init__(self, dtype=tf.float32):
        def ssim(x1, x2):
            """
            Bigger perturbations should have higher values.
            Therefore, we use the version of ssim that is used by Zhang et al.: The Unreasonable Effectiveness of Deep Features as a Perceptual Metric:
            See https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/__init__.py
            """
            return (1. - tf.image.ssim(x1, x2, max_val=255.)) / 2.

        super().__init__(ssim, dtype)


class PSNRMetric(FunctionWrapperMetric):
    """
    Assumes RGB input. -> max_val 255.
    For other color models, the max_val must be adapted.

    We use the negative psnr, such that big distances result in bigger values, as for the l2 and perceptual metrics
    """
    def __init__(self, dtype=tf.float32):
        super().__init__(lambda x1, x2: - tf.image.psnr(x1, x2, max_val=255.), dtype)


class PerceptualLossMetric:
    """
    Loss Metrics for perceptual loss.

    Uses LPIPS (Zhang et al., The Unreasonable Effectiveness of Deep Features as a Perceptual Metric, CVPR 2018)
    with a pretrained vgg16 net.
    See adversarial_attacks.models.lpips for implementation

    """

    def __init__(self, dataset_name, use_cpu=False):
        self._loss_network = LossNetwork.get_loss_network(dataset_name, lpips=True, net='vgg16', use_cpu=use_cpu)
        self._perceptual_distances = ImageDistanceMetric(tf.float64)

    def reset_state(self):
        self._perceptual_distances.reset_state()

    def update_state(self, original: tf.Tensor, adversarial: tf.Tensor):
        """
        updates the current distances by computing the perceptual loss for the original and adversarial image and
        concatenating it to self._distances.

        :param original: batched, color model rgb
        :param adversarial: batched, color model rgb.
        """
        perceptual_loss = self._loss_network(original, adversarial)[:, 0]
        self._perceptual_distances.update_state(perceptual_loss)

    def result(self):
        perceptual = self._perceptual_distances.result()
        return perceptual


class LpDistanceMetric(ImageDistanceMetric):
    """
    Distance metric for lp-norms.
    """

    def __init__(self, dtype=tf.float32):
        super(LpDistanceMetric, self).__init__(dtype)

        # Set in subclasses. Computes the image-wise distances between the original and adversarial image
        self._fn = None

    def __call__(self, original: tf.Tensor, adversarial: tf.Tensor):
        """
        Computes the image-wise distances between the original and adversarial image
        """
        return self._fn(original, adversarial)

    def update_state(self, original, adversarial):
        distances = self._fn(original, adversarial)
        super(LpDistanceMetric, self).update_state(distances)


class L0DistanceMetric(LpDistanceMetric):

    def __init__(self, dtype=tf.float32):
        super().__init__(dtype)

        def l0(original: tf.Tensor, adversarial: tf.Tensor):
            norm = tf.where(tf.equal(original, adversarial), tf.zeros_like(original), tf.zeros_like(original) + 1.)
            norm = tf.keras.layers.Flatten()(norm)
            norm = tf.math.reduce_sum(norm, axis=1)
            return norm

        self._fn = l0


class L1DistanceMetric(LpDistanceMetric):
    def __init__(self, dtype=tf.float32):
        super().__init__(dtype)

        def l1(original: tf.Tensor, adversarial: tf.Tensor):
            return tf.norm(tf.keras.layers.Flatten()(adversarial - original),
                           ord=1,
                           axis=1)

        self._fn = l1


class L2DistanceMetric(LpDistanceMetric):
    def __init__(self, dtype=tf.float32):
        super().__init__(dtype)

        def l2(original: tf.Tensor, adversarial: tf.Tensor):
            return tf.norm(tf.keras.layers.Flatten()(adversarial - original),
                           ord=2,
                           axis=1)

        self._fn = l2


class LInfDistanceMetric(LpDistanceMetric):
    def __init__(self, dtype=tf.float32):
        super().__init__(dtype)

        def linf(original: tf.Tensor, adversarial: tf.Tensor):
            return tf.norm(tf.keras.layers.Flatten()(adversarial - original),
                           ord=np.inf,
                           axis=1)

        self._fn = linf


class LpNorms:
    """
    Collection of L2, L1, L0, Linf norms for some color model.
    """

    def __init__(self, distance_color_model, input_color_model='rgb', ords=None):
        """

        :param distance_color_model: rgb, ciede2000, cielab or ycbcr
        :param input_color_model: rgb or ycbcr. Usually rgb (for our experiments).
        :param ords:
        """
        if ords is None:
            ords = ['l2', 'l1', 'l0', 'linf']

        self._metrics = {}
        if 'l0' in ords:
            self._metrics['l0'] = L0DistanceMetric()
        if 'l1' in ords:
            self._metrics['l1'] = L1DistanceMetric()
        if 'l2' in ords:
            self._metrics['l2'] = L2DistanceMetric()
        if 'linf' in ords:
            self._metrics['linf'] = LInfDistanceMetric()

        self._color_transformation_func = color_utils.get_color_transformation_func_two_images(input_color_model,
                                                                                               distance_color_model)

    def reset_state(self):
        for metric in self._metrics.values():
            metric.reset_state()

    def update_state(self, original_img, adversarial_img):
        color_transform = self._color_transformation_func(original_img, adversarial_img)
        for metric in self._metrics.values():
            metric.update_state(color_transform[0], color_transform[1])

    def __call__(self, original_img, adversarial_img):
        color_transform = self._color_transformation_func(original_img, adversarial_img)
        return {key: metric(color_transform[0], color_transform[1]) for key, metric in self._metrics.items()}

    def result(self):
        return {key: metric.result() for key, metric in self._metrics.items()}


class ExperimentDistanceWrapper:
    """
    Collects distance metrics for experiments.
    """

    def __init__(self, dataset_name, ords=None, perceptual_use_cpu=False):
        self._perceptual_loss_metric = PerceptualLossMetric(dataset_name, use_cpu=perceptual_use_cpu)
        self._rgb_metrics = LpNorms(distance_color_model='rgb', input_color_model='rgb', ords=ords)
        # self._cielab_metrics = LpNorms(distance_color_model='cielab', input_color_model='rgb', ords=ords)
        self._ciede2000_metrics = LpNorms(distance_color_model='ciede2000', input_color_model='rgb', ords=ords)
        self._ssim_metric = SSIMMetric()
        # self._psnr_metric = PSNRMetric()

    def update_state(self, original_rgb: tf.Tensor, adversarial_rgb: tf.Tensor):
        self._perceptual_loss_metric.update_state(original_rgb, adversarial_rgb)
        self._rgb_metrics.update_state(original_rgb, adversarial_rgb)
        # self._cielab_metrics.update_state(original_rgb, adversarial_rgb)
        self._ciede2000_metrics.update_state(original_rgb, adversarial_rgb)
        self._ssim_metric.update_state(original_rgb, adversarial_rgb)
        #self._psnr_metric.update_state(original_rgb, adversarial_rgb)

    def reset_state(self):
        self._perceptual_loss_metric.reset_state()
        self._rgb_metrics.reset_state()
        #self._cielab_metrics.reset_state()
        self._ciede2000_metrics.reset_state()
        self._ssim_metric.reset_state()
        #self._psnr_metric.reset_state()

    def result(self):
        return {
            'perceptual': self._perceptual_loss_metric.result(),
            'rgb': self._rgb_metrics.result(),
            #'cielab': self._cielab_metrics.result(),
            'ciede2000': self._ciede2000_metrics.result(),
            'ssim': self._ssim_metric.result(),
            #'psnr': self._psnr_metric.result()
        }


def tf_concat(tf1, tf2):
    if tf1 is None:
        return tf2
    if tf2 is None:
        return tf1

    return tf.concat([tf1, tf2], axis=0)


class MetricFnWrapperWithSavedDistances:
    def __init__(self, fn):
        self._fn = fn
        self._distances = None

    def update_state(self, original_rgb, adversarial_rgb):
        self._distances = tf_concat(self._distances, self._fn(original_rgb, adversarial_rgb))

    def reset_state(self):
        self._distances = None

    def result(self):
        return self._distances
