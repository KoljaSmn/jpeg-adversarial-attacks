import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import lpips

from adversarial_attacks.config.tf_distribution import tf_distribution_strategy
from adversarial_attacks.utils.distance_metrics import MetricFnWrapperWithSavedDistances
from adversarial_attacks.utils.tensorflow import tf_concat
from adversarial_attacks.models.models import Model


def softmax_to_classification(softmax_out: tf.Tensor) -> tf.Tensor:
    """
    Returns the classifications for a batch of images one-hot encoded.
    Returns a tensor with 1 at the max value and 0 everywhere else, across axis 1.
    
    :param softmax_out: shape (bs, num_classes)
    :return: classification with shape (bs, num_classes)
    """
    return tf.where(
        tf.equal(tf.math.reduce_max(softmax_out, axis=1, keepdims=True), softmax_out),
        tf.constant(1., shape=softmax_out.shape),
        tf.constant(0., shape=softmax_out.shape)
    )


def labels_match(tensor1: tf.Tensor, tensor2: tf.Tensor) -> tf.Tensor:
    """
    For a batch of images, this function returns if tensor1 and tensor2 match for every individual image.

    :param tensor1: shape (bs, num_classes)
    :param tensor2: shape (bs, num_classes)
    :return: shape (bs, )
    """
    return tf.math.reduce_prod(tf.where(tf.equal(tensor1,
                                                 tensor2),
                                        1,
                                        0), axis=1)


class AttackSuccessRate(tf.keras.metrics.CategoricalAccuracy):
    """
    Metric to measure the success rate of attacks.
    Success rate is defined as the relative number of all adversarial images classified incorrectly,
    with the original image being clasified correctly.
    """

    def reset_state(self):
        super(AttackSuccessRate, self).reset_state()

    def merge_state(self):
        raise NotImplementedError('merge_state is not implemented for this Metric.')

    def result(self):
        return super(AttackSuccessRate, self).result()

    def update_state(self, original_labels, original_softmax, adversarial_softmax):
        original_classification = softmax_to_classification(original_softmax)
        adversarial_classification = softmax_to_classification(adversarial_softmax)

        original_correct = labels_match(original_labels, original_classification)
        adversarial_incorrect = tf.math.subtract(1, labels_match(original_labels, adversarial_classification))
        sample_weight = original_correct
        original_correct = tf.one_hot(original_correct, 2)
        adversarial_incorrect = tf.one_hot(adversarial_incorrect, 2)

        super(AttackSuccessRate, self).update_state(original_correct, adversarial_incorrect,
                                                    sample_weight=sample_weight)


class EvaluationMetricCollection:
    """
    Collects evaluation metrics for one model.
    """

    def __init__(self, dataset_name, model_name: str = None, model=None):
        """

        :param dataset_name: 'cifar10', 'imagenet'
        :param model_name: the name of the model. If none, the model will be used.
        :param model:
        """

        self._acc = tf.keras.metrics.CategoricalAccuracy()
        self._top5Acc = tf.keras.metrics.TopKCategoricalAccuracy()
        self._categorical_crossentropy = tf.keras.metrics.CategoricalCrossentropy()
        self._success_rate = AttackSuccessRate()
        self._dataset_name = dataset_name
        if model_name is not None:
            self._model_name = model_name
            with tf_distribution_strategy.scope():
                self._model = Model(dataset_name=self._dataset_name, load_model_name=self._model_name, save_model=False)
        else:
            self._model = model

    def reset_state(self):
        self._acc.reset_state()
        self._top5Acc.reset_state()
        self._categorical_crossentropy.reset_state()
        self._success_rate.reset_state()

    def update_state(self, original_labels, original_img, adversarial_img):
        original_softmax = self._model(original_img, training=False, logits_or_softmax='softmax')
        adversarial_softmax = self._model(adversarial_img, training=False, logits_or_softmax='softmax')
        self._acc.update_state(original_labels, adversarial_softmax)
        self._top5Acc.update_state(original_labels, adversarial_softmax)
        self._categorical_crossentropy.update_state(original_labels, adversarial_softmax)
        self._success_rate.update_state(original_labels, original_softmax, adversarial_softmax)

    def result(self):
        return {'Acc': self._acc.result().numpy().astype(np.float64),
                'Top5Acc': self._top5Acc.result().numpy().astype(np.float64),
                'Loss': self._categorical_crossentropy.result().numpy().astype(np.float64),
                'success_rate': self._success_rate.result().numpy().astype(np.float64)}


class AFC2MetricWrapper:
    """Class that evaluates a distance function (fn) using the 2AFC metrics from the paper
       Zhang et al.: The Unreasonable Effectiveness of Deep Features as a Perceptual Metric, CVPR 2018
    """

    def __init__(self, fn):
        self._fn = fn
        self.m1, self.m2 = MetricFnWrapperWithSavedDistances(self._fn), MetricFnWrapperWithSavedDistances(self._fn)
        self.judge = None

    def update_state(self, ref, p0, p1, judge):
        self.judge = tf_concat(self.judge, judge)
        self.m1.update_state(p0, ref)
        self.m2.update_state(p1, ref)

    def reset_state(self):
        self.judge = None
        self.m1, self.m2 = MetricFnWrapperWithSavedDistances(self._fn), MetricFnWrapperWithSavedDistances(self._fn)

    def result(self):
        return self.m1.result(), self.m2.result(), self.judge

    def correlation(self):
        m1, m2, judge = self.result()
        judge = judge[:, 0]
        return tfp.stats.correlation(judge, tf.cast(m1 - m2, tf.float32), event_axis=None)

    def score(self, return_result_dict=False):
        d0s, d1s, gts = self.result()
        gts = gts[:, 0]
        scores = tf.where(d0s < d1s, 1. - gts, tf.where(d1s < d0s, gts, .5))

        if return_result_dict:
            return (tf.math.reduce_mean(scores), dict(d0s=d0s, d1s=d1s, gts=gts, scores=scores))

        return tf.math.reduce_mean(scores)


class JNDMetricWrapper:
    """
    Class that evaluates a distance function (fn) using the JND metrics from the paper
    Zhang et al.: The Unreasonable Effectiveness of Deep Features as a Perceptual Metric, CVPR 2018
    """

    def __init__(self, fn):
        self._fn = fn
        self.m1 = MetricFnWrapperWithSavedDistances(self._fn)
        self.same = None

    def update_state(self, p0, p1, same):
        self.same = tf_concat(self.same, same)
        self.m1.update_state(p0, p1)

    def reset_state(self):
        self.same = None
        self.m1 = MetricFnWrapperWithSavedDistances(self._fn)

    def result(self):
        return self.m1.result(), self.same

    def correlation(self):
        m1, same = self.result()
        return tfp.stats.correlation(same, tf.cast(m1, tf.float32), event_axis=None)

    def score(self, return_result_dict=False):
        # see score_jnd_dataset in https://github.com/richzhang/PerceptualSimilarity/blob/31bc1271ae6f13b7e281b9959ac24a5e8f2ed522/lpips/trainer.py#L243
        ds, sames = self.result()

        sames = sames.numpy()
        ds = ds.numpy()

        sorted_inds = np.argsort(ds)
        ds_sorted = ds[sorted_inds]
        sames_sorted = sames[sorted_inds]

        TPs = np.cumsum(sames_sorted)
        FPs = np.cumsum(1 - sames_sorted)
        FNs = np.sum(sames_sorted) - TPs

        precs = TPs / (TPs + FPs)
        recs = TPs / (TPs + FNs)
        score = lpips.voc_ap(recs, precs)

        if return_result_dict:
            return (score, dict(ds=ds, sames=sames))

        return score
