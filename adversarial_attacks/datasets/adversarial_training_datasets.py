from abc import ABC, abstractmethod
import math
import tensorflow as tf
import random

from adversarial_attacks.config.tf_distribution import tf_distribution_strategy as distribution_strategy
from adversarial_attacks.utils.tensorflow import tf_concat
from adversarial_attacks.datasets.original import RGBDataset

from adversarial_attacks.utils.logging import warning


def sum_normalize_weights(dictionary):
    w_sum = sum(list(dictionary.values()))
    for key in dictionary:
        dictionary[key] = dictionary[key] / w_sum
    return dictionary


class _AdversarialTrainingDataset(ABC):
    @abstractmethod
    def __init__(self, original_ds, dynamic_adversarial_training_attacks, internal_batch_size, shuffle: bool = True,
                 distribute_attack_wise: bool = False):
        if not isinstance(original_ds, RGBDataset):
            raise ValueError('original_ds must be an RGBDataset.')

        self._original_ds = original_ds
        self._internal_batch_size = internal_batch_size
        self._original_n_images = self._original_ds.number_of_images

        self._n_internal_batches = math.ceil(self._original_n_images / self._internal_batch_size)
        self._shuffle = shuffle

        self._distribute_attack_wise = distribute_attack_wise

        if not self._shuffle and original_ds.train_or_test == 'train':
            warning('Dataset shuffling is turned off but is recommended for the train dataset. '
                    'Otherwise the same images will be original or adversarial in every epoch.')

        self._use_dynamic_attacks = dynamic_adversarial_training_attacks is not None
        l = list(dynamic_adversarial_training_attacks.keys())
        self._device_for_attack = {}
        for i in range(len(l)):
            device_nr = i % distribution_strategy.num_replicas_in_sync
            self._device_for_attack[l[i]] = f'/device:gpu:{device_nr}'

        self._original_tf_dataset = self._original_ds.ds.take(self._original_n_images).prefetch(tf.data.AUTOTUNE)

        if self._use_dynamic_attacks:
            self._dynamic_attacks = sum_normalize_weights(dynamic_adversarial_training_attacks)
        else:
            self._dynamic_attacks = None

        self.original_ds = self._original_tf_dataset
        if self._shuffle:
            buffer_size = max([1000, self._original_n_images // 1000])
            self.original_ds = self.original_ds.shuffle(buffer_size)
        self.original_ds = self.original_ds.batch(self._internal_batch_size, drop_remainder=False).take(
            self._n_internal_batches).repeat()
        # the adversarial training iterates through this original ds and
        # creates adversarial batch using the get_adversarial_batch method

        # the adversarial dataset (tf.data.Dataset) will be set in subclasses
        self.ds = None

    def get_n_batches(self):
        return self._n_internal_batches

    def get_batch_size(self):
        return self._internal_batch_size

    @abstractmethod
    def _full_generator(self):
        pass

    @abstractmethod
    def get_adversarial_batch(self, entry):
        pass

    def _get_indexed_dataset(self, ds, batched=False):
        """
        Adds an index to every entry of the input ds.
        :param ds: tf.data.Dataset with images, labels entries
        :param batched: whether the input (and output) is batched or not
        :return:
        """

        def gen():
            idx = 0.
            for images, labels in ds:
                yield idx, (images, labels)
                idx += 1.

        if batched:
            return tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec([], dtype=tf.int64),
                                                                         self._original_ds.batched_output_signature))
        return tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec([], dtype=tf.int64),
                                                                     self._original_ds.output_signature))

    def _attack_batch(self, images, labels, attack_dict):
        """
        Attacks a batch of images and returns the adversarial images.
        The attack is randomly chosen from the attack dict.


        :param images: rgb image batch
        :param labels: one-hot encoded labels batch
        :param attack_dict: dict: attack1 -> weight1, attack2 -> weight2 etc. weights must sum up to 1.
        :return:
        """
        attack = random.choices(list(attack_dict.keys()), weights=attack_dict.values(), k=1)[0]

        if self._distribute_attack_wise:
            with tf.device(self._device_for_attack[attack]):
                adv_images = attack(images, labels)
        else:
            adv_images = attack(images, labels)
        return adv_images


class AdversarialTrainingDataset(_AdversarialTrainingDataset):
    """
    Adversarial Examples are created dynamically during training.
    """

    def __init__(self, original_ds, dynamic_adversarial_training_attacks, internal_batch_size, shuffle: bool = True,
                 distribute_attack_wise: bool = False):
        super(AdversarialTrainingDataset, self).__init__(original_ds, dynamic_adversarial_training_attacks,
                                                         internal_batch_size, shuffle,
                                                         distribute_attack_wise=distribute_attack_wise)
        self.ds = tf.data.Dataset.from_generator(self._full_generator,
                                                 output_signature=original_ds.batched_output_signature).prefetch(
            tf.data.AUTOTUNE)

    def _full_generator(self):
        for entry in self.original_ds:
            yield self.get_adversarial_batch(entry)

    @tf.function
    def get_adversarial_batch(self, entry):
        """
        Returns an adversarial batch for an original batch.

        Half the original batch is attacked and replaced by the adversarial examples.
        This function is used in the adversarial training.

        :param entry: original images, labels
        :return:
        """
        images_batch, labels_batch = entry

        bs = tf.shape(images_batch)[0]

        n_original_images = tf.cast(tf.math.ceil(bs / 2), tf.int64)

        original_images = images_batch[:n_original_images]
        original_labels = labels_batch[:n_original_images]

        adv_labels = labels_batch[n_original_images:]
        adv_images = self._attack_batch(images_batch[n_original_images:], adv_labels, self._dynamic_attacks)

        return tf_concat(original_images, adv_images), tf_concat(original_labels, adv_labels)
