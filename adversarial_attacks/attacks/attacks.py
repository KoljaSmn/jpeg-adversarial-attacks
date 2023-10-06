from abc import ABC, abstractmethod

import tensorflow as tf

import adversarial_attacks.attacks
import adversarial_attacks.utils.attacks
import adversarial_attacks.utils.jpeg
import adversarial_attacks.utils.transformation
import adversarial_attacks.utils.color
from adversarial_attacks.models.models import Model
import adversarial_attacks.attacks.fast_adversarial_rounding

JPEG_CLIP_MIN, JPEG_CLIP_MAX = -1024.0, 1018.0
YCBCR_CLIP_MIN, YCBCR_CLIP_MAX = 0., 255.
ETA_DEFAULT = 0.05


def default_alpha(alpha, epsilon, T):
    """
    The step size (alpha) is by default set to epsilon/T.
    """
    if alpha is None:
        if epsilon is None:
            return None
        return epsilon / T
    return alpha


@tf.function
def frequency_wise_epsilon_alpha(eps_rel, lambda_channel):
    """
    Returns a vector of length 64.
    """
    return lambda_channel * eps_rel


class Attack(ABC):
    """
    Abstract Attack class.
    """
    _attack_name = None  # to be set in subclasses
    _model_type = None  # to be set in subclasses

    @abstractmethod
    def __init__(self, dataset, model_name, loss_fn='crossentropy', model=None):
        """
        :param dataset: the name of the dataset
        :param model_name: the name of the model. If model is not None, model_name will be ignored.
        :param loss_fn: crossentropy or confidence.
        :param model: The model. If model is not None, model_name will be ignored.
                      This is mainly used for adversarial training.
                      It prevents the model from being loaded twice (once for training, once for attacking)
                      such that the images are created on the model that is being trained.

        """
        self._dataset = dataset

        if self._attack_name is None:
            raise ValueError("_attack_name variable must be set in subclass.")

        if self._model_type is None:
            raise ValueError("_model_type variable must be set in subclass.")

        if model is None:
            self._model_name = model_name
            self._model = self._load_model()
        else:
            self._model_name = None
            self._model = model

        self._loss_fn_name = loss_fn
        self._loss_fn = lambda y_true, y_pred: Attack._get_tf_loss_fn(self._loss_fn_name)(y_true, y_pred,
                                                                                          from_logits=True)
        self.chroma_subsampling = True  # will be set in subclasses
        self.jpeg_quality = 100  # will be set in subclasses

    def type_plus_attack_name(self):
        return f'{self._model_type}_{self._attack_name}'

    def get_jpeg_quality(self):
        return self.jpeg_quality

    def get_use_chroma_subsampling(self):
        return self.chroma_subsampling

    def get_model_name(self):
        return self._model_name

    def get_model_type(self):
        return self._model_type

    def get_dataset_name(self):
        return self._dataset

    def get_white_box_model(self):
        return self._model

    def cache_key(self):
        """
        The cache key can be used to save the experiment's results to files.
        """
        if self._model_name is None:
            raise ValueError(
                'Results for attacks that have not been initialized using the model_name parameters can not be cached.')

        return f'{self._attack_name}_dataset_{self._dataset}_model_{self._model_name}_loss_{self._loss_fn_name}'

    def _load_model(self):
        return Model(self._dataset, save_model_name=self._model_name, load_model_name=self._model_name,
                     save_model=False)

    @staticmethod
    def _get_tf_loss_fn(_loss_fn_name):
        if _loss_fn_name == 'crossentropy':
            return tf.keras.losses.categorical_crossentropy
        if _loss_fn_name == 'confidence':
            return adversarial_attacks.utils.attacks.confidence
        if _loss_fn_name is None:
            return None
        raise ValueError('Loss function {} not implemented'.format(_loss_fn_name))

    @tf.function
    def predict(self, images, logits_or_softmax='softmax'):
        """
        Predicts a batch of images of the same type as the _model.

        :param images: unquantized, batched RGB data
        :param logits_or_softmax: 'logits', 'softmax'

        :return: the output (logits or softmax)
        """
        return self._model(images, training=False, logits_or_softmax=logits_or_softmax)

    def __call__(self, images, labels):
        """
        The call function expects batched RGB data and one-hot encoded labels.

        :param images: batch of rgb images
        :param labels: batch of one-hot labels
        :return:
        """
        return self._attack_rgb(images, labels)

    @abstractmethod
    def rgb_to_input_datatype_conversion(self, images):
        """
        Conversion of RGB input to the input datatype of each attack (e.g. JPEG).
        Implemented in subclasses.
        """
        pass

    def rgb_to_output_datatype_conversion(self, images):
        """
        Conversion of RGB output to the output datatype of each attack (e.g. JPEG).
        Can be overwritten in subclasses.
        Default case is that input_datatype==output_datatype
        """
        return self.rgb_to_input_datatype_conversion(images)

    @abstractmethod
    def input_datatype_to_rgb_conversion(self, images):
        """
        Conversion of input datatype (e.g. JPEG) to RGB data.
        Implemented in subclasses.
        """
        pass

    def output_datatype_to_rgb_conversion(self, images):
        """
        Conversion of input datatype (e.g. JPEG) to RGB data.
        Can be overwritten in subclasses.
        Default case is that input_datatype==output_datatype
        """
        return self.input_datatype_to_rgb_conversion(images)

    def _attack_rgb(self, images, labels):
        """
        Attacks an RGB Batch. First, it converts the RGB input to input datatype (e.g. JPEG),
        then it performs an attack and converts the output to unquantized RGB data.

        :param images: batch of rgb images
        :param labels: batch of one-hot labels
        :return:
        """
        return self.output_datatype_to_rgb_conversion(
            self.attack_datatype(self.rgb_to_input_datatype_conversion(images), labels))

    @abstractmethod
    def attack_datatype(self, images, labels):
        """
        Here, the main attack is implemented.
        It already expects the correct input datatype for each attack (e.g. JPEG).
        """
        pass


@tf.function
def ycbcr_jpeg_clone(images):
    """
    Clones a (Y, Cb, Cr) tuple of coefficents or pixels.
    """
    return tf.identity(images[0]), tf.identity(images[1]), tf.identity(images[2])


@tf.function
def ycbcr_round(Y, Cb, Cr):
    """
    Rounds (Y, Cb, Cr) tuples to the nearest integer.
    """
    return tf.math.round(Y), tf.math.round(Cb), tf.math.round(Cr)


@tf.function
def ycbcr_clip_and_cast(Y, Cb, Cr, clip_min, clip_max, Y_clip=None, Cb_clip=None, Cr_clip=None, round=True):
    """
    Clips the images onto the valid range
    :param Y:
    :param Cb:
    :param Cr:
    :param clip_min: min clipping values used for all three channels
    :param clip_max: max clipping values used for all three channels
    :param Y_clip: Bounds for Y. Additionally to the default clipping ranges.
    :param Cb_clip: Bounds for Cb.
    :param Cr_clip: Bounds for Cr.
    :param round: whether to round the values to integers. Should be used at the attack's end.
     But not during the iterations.
    :return: The clipped (and rounded) channels.
    """

    if Y_clip is not None:
        Y = tf.clip_by_value(Y, Y_clip[0], Y_clip[1])
    if Cb_clip is not None:
        Cb = tf.clip_by_value(Cb, Cb_clip[0], Cb_clip[1])
    if Cr_clip is not None:
        Cr = tf.clip_by_value(Cr, Cr_clip[0], Cr_clip[1])

    Y = tf.clip_by_value(Y, clip_min, clip_max)
    Cb = tf.clip_by_value(Cb, clip_min, clip_max)
    Cr = tf.clip_by_value(Cr, clip_min, clip_max)

    if round: Y, Cb, Cr = ycbcr_round(Y, Cb, Cr)
    return Y, Cb, Cr
