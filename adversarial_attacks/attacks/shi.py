from abc import ABC, abstractmethod

import tensorflow as tf

import adversarial_attacks.attacks.fast_adversarial_rounding
import adversarial_attacks.utils.transformation
from adversarial_attacks.attacks.attacks import Attack, ETA_DEFAULT
from adversarial_attacks.attacks.rgb import RGBFGSM, RGBBIM
from adversarial_attacks.config.config import Config


class SHIAttack(Attack, ABC):
    """
    Abstract class for the attacks from
    Shi et al., On Generating JPEG Adversarial Images, ICME 2021
    """
    _model_type = 'jpeg'

    @abstractmethod
    def __init__(self, dataset, model_name, loss_fn='crossentropy', eta: float = ETA_DEFAULT,
                 jpeg_quality: int = 100, chroma_subsampling: bool = True, model=None):
        """
        :param eta: the eta value used for the attack, determines how many coefficients
        are rounded in the gradient's direction

        :param dataset: the name of the dataset
        :param model_name: the name of the model. If model is not None, model_name will be ignored.
        :param loss_fn: crossentropy or confidence.
        :param model: The model. If model is not None, model_name will be ignored.
                      This is mainly used for adversarial training.
                      It prevents the model from being loaded twice (once for training, once for attacking)
                      such that the images are created on the model that is being trained.


        """
        super().__init__(dataset, model_name, loss_fn=loss_fn, model=model)

        self.eta = eta
        self.jpeg_quality = jpeg_quality
        self.chroma_subsampling = chroma_subsampling

        # load a model the converts JPEG to RGB data
        self.jpeg_to_rgb_model = adversarial_attacks.utils.transformation.get_jpeg_to_rgb_model(self._dataset,
                                                                                                self.jpeg_quality,
                                                                                                round=None,
                                                                                                chroma_subsampling=
                                                                                                chroma_subsampling)
        # load a model that converts rgb to unquantized (intermediate) JPEG data
        self.rgb_to_jpeg_model = adversarial_attacks.utils.transformation.get_rgb_to_jpeg_model(self._dataset,
                                                                                                self.jpeg_quality,
                                                                                                round=None,
                                                                                                chroma_subsampling=
                                                                                                chroma_subsampling)
        # defines the rgb attack, will be set in subclasses
        self.rgb_attack = None

        # initialize the fast adversarial rounding object
        loss_fn_for_image_label_input = lambda images, label: self._loss_fn(label, self._jpeg_prediction_func(images))
        self._fast_adversarial_rounding = adversarial_attacks.attacks.fast_adversarial_rounding. \
            FastAdversarialRounding(self.eta, self.jpeg_quality, loss_fn_for_image_label_input)

    def predict(self, images, logits_or_softmax='softmax'):
        """
        Predicts a batch of JPEG images
        :param images:
        :param logits_or_softmax:
        :return:
        """
        return super().predict(self.jpeg_to_rgb_model(images), logits_or_softmax=logits_or_softmax)

    def cache_key(self):
        return f'shi_{super().cache_key()}_eta_{self.eta}_jq_{self.jpeg_quality}_cs_{self.chroma_subsampling}'

    @tf.function
    def _jpeg_prediction_func(self, jpeg_adversarial):
        return self.predict(jpeg_adversarial, logits_or_softmax='logits')

    @tf.function
    def rgb_to_jpeg_and_fast_adversarial_rounding(self, rgb_adversarial, labels):
        """
        Converts rgb image to unquantized (intermediate) JPEG data and rounds it using fast adversarial rounding
        :param rgb_adversarial:
        :param labels:
        :return:
        """
        return self._fast_adversarial_rounding(self.rgb_to_jpeg_model(rgb_adversarial), labels)

    @tf.function
    def attack_datatype(self, images, labels):
        """
        Performs the attack. First, the rgb attack is performed.
        The output is then converted to intermediate coefficients and rounded using fast adversarial rounding.

        :param images: batch of rgb images
        :param labels: batch of one-hot labels
        :return:
        """
        rgb_adversarial = self.rgb_attack(images, labels)
        jpeg_adversarial = self.rgb_to_jpeg_and_fast_adversarial_rounding(rgb_adversarial, labels)
        return jpeg_adversarial[0], jpeg_adversarial[1], jpeg_adversarial[2]

    @tf.function
    def rgb_to_input_datatype_conversion(self, images):
        return images
    
    @tf.function
    def input_datatype_to_rgb_conversion(self, images):
        return images
    
    @tf.function
    def output_datatype_to_rgb_conversion(self, images):
        return self.jpeg_to_rgb_model(images)


class SHIFGSM(SHIAttack):
    """
    Shi attack with rgb fgsm as starting point.
    """
    _attack_name = 'shi_fgsm'

    def __init__(self, dataset, model_name, loss_fn='crossentropy', epsilon: float = 8.,
                 eta: float = ETA_DEFAULT, jpeg_quality: int = 100, model=None,
                 chroma_subsampling: bool = True):
        """
        :param epsilon: rgb perturbation bound
        :param eta: the eta value used for the attack, determines how many coefficients
        are rounded in the gradient's direction

        :param dataset: the name of the dataset
        :param model_name: the name of the model. If model is not None, model_name will be ignored.
        :param loss_fn: crossentropy or confidence.
        :param model: The model. If model is not None, model_name will be ignored.
                      This is mainly used for adversarial training.
                      It prevents the model from being loaded twice (once for training, once for attacking)
                      such that the images are created on the model that is being trained.


        """
        super().__init__(dataset, model_name, loss_fn=loss_fn, eta=eta, jpeg_quality=jpeg_quality, model=model,
                         chroma_subsampling=chroma_subsampling)

        self.epsilon = epsilon

        self.rgb_attack = RGBFGSM(dataset=self._dataset, model_name=self._model_name, epsilon=self.epsilon,
                                  loss_fn=self._loss_fn_name)

    def cache_key(self):
        return f'{super().cache_key()}_epsilon_{self.epsilon}'


class SHIBIM(SHIFGSM):
    """
    Shi attack with rgb bim as starting point.
    """
    _attack_name = 'shi_bim'

    def __init__(self, dataset, model_name, epsilon: float = 8., alpha: float = None, T: int = 10,
                 eta: float = ETA_DEFAULT, jpeg_quality: int = 100, model=None,
                 chroma_subsampling: bool = True):
        """
        :param epsilon: rgb perturbation bound
        :param alpha: rgb step size
        :param T: Iterations of the RGB attack
        :param eta: the eta value used for the attack, determines how many coefficients
        are rounded in the gradient's direction

        :param dataset: the name of the dataset
        :param model_name: the name of the model. If model is not None, model_name will be ignored.
        :param loss_fn: crossentropy or confidence.
        :param model: The model. If model is not None, model_name will be ignored.
                      This is mainly used for adversarial training.
                      It prevents the model from being loaded twice (once for training, once for attacking)
                      such that the images are created on the model that is being trained.


        """
        super().__init__(dataset, model_name, epsilon=epsilon, eta=eta, jpeg_quality=jpeg_quality, model=model,
                         chroma_subsampling=chroma_subsampling)

        self.alpha = alpha
        self.T = T

        self.rgb_attack = RGBBIM(dataset=self._dataset, model_name=self._model_name, epsilon=self.epsilon,
                                 alpha=self.alpha, T=self.T,
                                 loss_fn=self._loss_fn_name)

    def cache_key(self):
        return f'{super().cache_key()}_alpha_{self.alpha}_T_{self.T}'
