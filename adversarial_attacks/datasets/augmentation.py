import tensorflow as tf
import numpy as np
import math


class RandomCropAndResizeLayer(tf.keras.layers.RandomCrop):

    def __init__(self, crop_probability, original_height, original_width, crop_height, crop_width, **kwargs):
        super().__init__(crop_height, crop_width, **kwargs)
        if crop_probability < 0 or crop_probability > 1:
            raise ValueError('Crop Probability invalid.')
        self._crop_probability = crop_probability
        self._resize_layer = tf.keras.layers.Resizing(original_height, original_width)

    def __call__(self, images, training=False):
        bs = tf.shape(images)[0]
        return tf.where(tf.random.uniform([bs, 1, 1, 1]) < self._crop_probability,
                        self._resize_layer(super(RandomCropAndResizeLayer, self).__call__(images)),
                        images
                        )


def rgb_cast_to_float(x, y):
    return tf.cast(x, tf.float32), y


class RandomFlipWithProb(tf.keras.layers.RandomFlip):
    def __init__(self, prob, mode, seed=None):
        super(RandomFlipWithProb, self).__init__(mode, seed)
        self._prob = prob

    def get_random_transformation(
            self, image=None, label=None, bounding_box=None
    ):
        flip_horizontal = False
        flip_vertical = False
        if self.horizontal:
            flip_horizontal = np.random.choice([True, False], p=[self._prob, 1. - self._prob])
        if self.vertical:
            flip_vertical = np.random.choice([True, False], p=[self._prob, 1. - self._prob])
        return {
            "flip_horizontal": flip_horizontal,
            "flip_vertical": flip_vertical,
        }


class RandomDataAugmentation(tf.keras.Sequential):
    """
    This Model randomly augments the data.
    """

    def __init__(self, w, h, crop_w, crop_h):
        super().__init__(tf.keras.Sequential([
            RandomCropAndResizeLayer(.5, w, h, crop_w, crop_h),
            RandomFlipWithProb(0.5, 'horizontal'),
            tf.keras.layers.RandomRotation(0.1),
        ]))
        for layer in self.layers:
            layer.trainable = False
        self.trainable = False

    def __call__(self, images):
        return super(RandomDataAugmentation, self).__call__(images, training=True)


def dataset_augmentation(batched_ds, rgb_img_shape, augmentation):
    """
    Augments an original dataset.
    :param batched_ds: batched tf.data.Dataset
    :param rgb_img_shape:
    :param augmentation: if 0, no augmentation will be applied.
           if > 0, the original images will be passed through the RandomDataAugmentation model augmentation times
           and the output will be added to the original ds
           if < 0, the original images will be passed through the RandomDataAugmentation model augmentation times,
           but the original images will not be included in the output dataset
    :return:
    """
    if augmentation == 0:
        raise ValueError("augmentation must be != 0.")

    w, h = rgb_img_shape[0], rgb_img_shape[1]
    crop_w, crop_h = math.ceil(0.75 * w), math.ceil(0.75 * h)

    random_data_augmentation = RandomDataAugmentation(w, h, crop_w, crop_h)

    def _augment_batch(images, labels):
        return random_data_augmentation(images), labels

    if augmentation < 0:
        n_repeats = abs(augmentation) - 1
        augmented_ds = batched_ds.map(_augment_batch)
    else:
        n_repeats = augmentation
        augmented_ds = batched_ds

    for i in range(n_repeats):
        augmented_ds = augmented_ds.concatenate(batched_ds.map(_augment_batch))

    return augmented_ds
