import os
from abc import ABC, abstractmethod
import math

import tensorflow as tf
import tensorflow_datasets as tfds

from adversarial_attacks.config.config import Config
from adversarial_attacks.utils import logging as logging
from adversarial_attacks.utils.general import makedirs
import adversarial_attacks.utils.color
import adversarial_attacks.utils.transformation
import adversarial_attacks.attacks.attacks
import adversarial_attacks.datasets.augmentation
import adversarial_attacks.models.rgb_to_jpeg


class Dataset(ABC):
    """
    Abstract class to load cifar10 or imagenet dataset.

    Wraps a tf.data.Dataset.

    """
    # caching of created datasets
    created_datasets = {}

    @abstractmethod
    def __init__(self, dataset_name, train_or_test, number_of_images=None, augmentation: int = 0,
                 shuffle=True, repeat=True, cache=None):
        """
        :param dataset_name: the name of the dataset, 'cifar10' or 'imagenet'
        :param train_or_test: the type of the dataset
                              'train', 'test' for cifar10 or
                              'train', 'validation' for imagenet ('test' is also available but without labels)
        :param number_of_images: the number of images. None for full dataset.
        :param augmentation: 0 for no augmentation (only original), 1 for original + 1 augmented dataset, 2 for original + 2 augmented
                                -1 for only 1 augmented dataset (no original), -2 for only 2 augmented datasets (no original)
        :param shuffle: whether the dataset should be shuffled. Should be true for training datasets.
        :param repeat: whether the dataset should be repeated infinitely.
        :param cache: whether the dataset should be cached
        """
        self.dataset_name = dataset_name
        self.train_or_test = train_or_test
        self.repeat = repeat
        
        if cache is None:
            self._cache = None
        else:
            self._cache = cache
            
        self.shuffle = shuffle
            
        if number_of_images is None:
            self.number_of_images = Config.DATASET_SIZES[self.get_dataset_name()][self.train_or_test]
        else:
            self.number_of_images = number_of_images
            
        if self._cache is None and not self.shuffle and augmentation == 0:
            self._cache = self.number_of_images < 20000

        if self._cache:
            self._cache_dir = os.path.join(Config.DATASETS_CACHE_DIR, dataset_name)
        else:
            self._cache_dir = None

        self.augmentation = augmentation

        self.number_of_original_images = self.number_of_images

        if self.augmentation != 0:
            self.number_of_images *= self._augmentation_factor()

        # load dataset shapes
        # for imagenet, all images are resized to 224*224
        self.not_subsampled_dct_shape = Config.INPUT_SHAPE[self.get_dataset_name()]['Y']
        self.subsampled_dct_shape = Config.INPUT_SHAPE[self.get_dataset_name()]['C']
        self.rgb_img_shape = Config.INPUT_SHAPE[self.get_dataset_name()]['rgb']

        if self._cache and self._tf_dataset_key() in Dataset.created_datasets:
            self.ds = Dataset.created_datasets[self._tf_dataset_key()]
        else:
            self.ds: tf.data.Dataset = self._create_dataset()
            Dataset.created_datasets[self._tf_dataset_key()] = self.ds

        self.original_rgb_dataset = self._get_original_rgb_dataset(tf_dataset=True)

        self.output_signature = None
        self.batched_output_signature = None
        self._ds_type = None

    def _cache_key(self):
        return '{}_{}_{}_{}_{}'.format(self.ds_type(),
                                       self.train_or_test,
                                       self.number_of_original_images,
                                       self.augmentation,
                                       self.dataset_name)

    def cache_key(self):
        return self._cache_key()

    def get_n_batches(self, batch_size):
        return math.ceil(self.number_of_images / batch_size)

    @abstractmethod
    def _tf_dataset_key(self):
        return '{}dsName{}trainOrTest{}nImages{}augmentation{}shuffle{}repeat{}cache{}'.format(self.ds_type(),
                                                                                               self.dataset_name,
                                                                                               self.train_or_test,
                                                                                               self.number_of_original_images,
                                                                                               self.augmentation,
                                                                                               self.shuffle,
                                                                                               self.repeat,
                                                                                               self._cache)

    def get_dataset_name(self):
        return self.dataset_name

    def ds_type(self):
        return self._ds_type

    def num_classes(self):
        return Config.DATASET_NUM_CLASSES[self.get_dataset_name()]

    def dataset_loading_name(self):
        return Config.DATASET_LOADING_NAME[self.get_dataset_name()]

    def _augment(self, batched_ds):
        batched_ds = batched_ds.map(adversarial_attacks.datasets.augmentation.rgb_cast_to_float)
        if self.augmentation != 0:
            return adversarial_attacks.datasets.augmentation.dataset_augmentation(batched_ds, self.rgb_img_shape,
                                                                                  augmentation=self.augmentation)
        return batched_ds

    def _augmentation_factor(self):
        """
        Returns a factor with which the number of images must be multiplied for a given augmentation parameter.
        :return:
        """
        if self.augmentation < 0:
            return abs(self.augmentation)
        return self.augmentation + 1

    def _load_full_tf_rgb_dataset(self):
        """
        Loads the original dataset as a tf.data.Dataset.
        :return:
        """
        type_name = self.train_or_test
        ds = tfds.load(self.dataset_loading_name(),
                       data_dir=Config.TF_DATASETS_DIR,
                       split=[type_name],
                       as_supervised=True,
                       shuffle_files=self.shuffle)[0]
        return ds

    def get_label_str(self, y):
        """
        Returns the label string for numerical labels y.
        :param y:
        :return:
        """
        if len(tf.shape(y)) == 2:  # batched, one hot
            y = tf.math.argmax(y, axis=1)
        elif len(tf.shape(y)) == 0:  # single scalar label
            y = tf.convert_to_tensor([y])
        elif len(tf.shape(y) == 1):  # batched, not one-hot
            y = tf.convert_to_tensor(y)
        else:
            raise ValueError("Error.")
        y = y.numpy().tolist()

        return [Config.LABELS[self.dataset_name][label] for label in y]

    def _get_equally_distributed_dataset_subset(self, tf_ds, n_images):
        """
        Returns an equally distributed dataset subset with the given number of images.

        :param tf_ds:
        :param n_images:
        :return:
        """
        n_classes = self.num_classes()
        n_images_per_class = math.floor(n_images / n_classes)
        n_classes_extra_image = n_images % n_classes

        def concat_datasets(ds1, ds2):
            if ds1 is None:
                return ds2
            if ds2 is None:
                return ds1
            return ds1.concatenate(ds2)

        concatenated_ds = None

        for c in range(n_classes):
            if c < n_classes_extra_image:
                n_this_class = n_images_per_class + 1
            else:
                n_this_class = n_images_per_class

            def class_filter_fn(x, y):
                return tf.math.equal(y, c)

            filtered_ds = tf_ds.filter(class_filter_fn).take(n_this_class)
            concatenated_ds = concat_datasets(concatenated_ds, filtered_ds)

        return concatenated_ds

    def _create_dataset(self):
        """
        Creates the tf.data.Dataset.

        :return:
        """

        # First, the original dataset is loaded
        ds = self._load_full_tf_rgb_dataset()
        # then, the dataset subset is created
        if self.number_of_original_images != Config.DATASET_SIZES[self.get_dataset_name()][self.train_or_test]:
            ds = self._get_equally_distributed_dataset_subset(ds, self.number_of_original_images)
        else:
            ds = ds.take(self.number_of_original_images)

        target_shape = Config.INPUT_SHAPE[self.get_dataset_name()]['rgb']

        # defines a function that resizes images (imagenet -> 224*224)
        def image_map(img, label):
            return tf.image.resize(img, size=[target_shape[0], target_shape[1]]), label

        # defines a function that converts the images to the datatype required (RGB, JPEG, YCbCr)
        # and one-hot encodes the labels
        def batch_map(images, labels):
            return self.rgb_to_datatype_transformation(images), tf.one_hot(labels, self.num_classes())

        # applies resizing, data augmentation, datatype conversion, and one-hot encoding
        prep_batch_size = Config.PREPARATION_BATCH_SIZE[self.get_dataset_name()]
        ds = self._augment(ds.map(image_map).batch(prep_batch_size, num_parallel_calls=tf.data.AUTOTUNE)).map(
            batch_map).unbatch()

        # creates cache dir and caches the dataset (when it is iterated for the first time)
        if self._cache:
            makedirs(self._cache_dir)
            cache_file = os.path.join(self._cache_dir, self._cache_key())
            logging.info(f'Caching dataset at {cache_file}')
            ds = ds.cache(cache_file)

        # shuffles the dataset
        if self.shuffle:
            buffer_size = max(self.number_of_images // 1000, 1000)
            ds = ds.shuffle(buffer_size)

        # repeats the dataset infinitely
        if self.repeat:
            ds = ds.repeat()

        return ds.prefetch(tf.data.AUTOTUNE)

    def _get_original_rgb_dataset(self, number_of_images=None, tf_dataset=True):
        """
        Returns the dataset with type RGB.
        :param number_of_images:
        :param tf_dataset:
        :return:
        """
        number_of_images = self.number_of_images if number_of_images is None else number_of_images
        dataset = RGBDataset(self.get_dataset_name(),
                             train_or_test=self.train_or_test,
                             number_of_images=number_of_images,
                             augmentation=self.augmentation,
                             shuffle=False,
                             repeat=self.repeat,
                             cache=False)
        if tf_dataset:
            return dataset.ds
        else:
            return dataset

    @abstractmethod
    def rgb_to_datatype_transformation(self, rgb):
        """
        Converts RGB data to the required datatype. implemented in subclasses.
        :param rgb:
        :return:
        """
        pass

    @abstractmethod
    def datatype_to_rgb_transformation(self, datatype_image):
        """
        Converts required datatype to RGB. implemented in subclasses.
        :param datatype_image:
        :return:
        """
        pass


class RGBDataset(Dataset):
    """
    RGB dataset.
    """
    _ds_type = 'rgb'

    def __init__(self, dataset_name, train_or_test, number_of_images=None, augmentation: int = 0,
                 shuffle=True, repeat=True, cache=None):
        """
        :param dataset_name: the name of the dataset, 'cifar10' or 'imagenet'
        :param train_or_test: the type of the dataset
                              'train', 'test' for cifar10 or
                              'train', 'validation' for imagenet ('test' is also available but without labels)
        :param number_of_images: the number of images. None for full dataset.
        :param augmentation: 0 for no augmentation (only original), 1 for original + 1 augmented dataset, 2 for original + 2 augmented
                                -1 for only 1 augmented dataset (no original), -2 for only 2 augmented datasets (no original)
        :param shuffle: whether the dataset should be shuffled. Should be true for training datasets.
        :param repeat: whether the dataset should be repeated infinitely.
        :param cache: whether the dataset should be cached
        """
        super().__init__(dataset_name, train_or_test, number_of_images, augmentation, shuffle=shuffle, repeat=repeat,
                         cache=cache)

        w, h, c = self.rgb_img_shape
        output_shape = [w, h, c]

        self.output_signature = (
            tf.TensorSpec(shape=output_shape)
            ,
            tf.TensorSpec(shape=(self.num_classes()))
        )

        self.batched_output_signature = (
            tf.TensorSpec(shape=
                          [None] + output_shape
                          )
            ,
            tf.TensorSpec(shape=(None,
                                 self.num_classes()))
        )

    def _tf_dataset_key(self):
        return 'rgb{}'.format(super()._tf_dataset_key())

    def _get_original_rgb_dataset(self, number_of_images=None, tf_dataset=True):
        if tf_dataset:
            return self.ds
        return self

    @tf.function
    def rgb_to_datatype_transformation(self, rgb):
        return rgb

    def datatype_to_rgb_transformation(self, datatype_image):
        return datatype_image


class JpegDataset(Dataset):
    """
    Dataset of type JPEG.
    """
    _ds_type = 'jpeg'

    def __init__(self, dataset_name, train_or_test, number_of_images=None, augmentation: int = 0,
                 shuffle=True, repeat=True, jpeg_quality: int = 100, chroma_subsampling: bool = True, cache=None):
        """
        :param dataset_name: the name of the dataset, 'cifar10' or 'imagenet'
        :param train_or_test: the type of the dataset
                              'train', 'test' for cifar10 or
                              'train', 'validation' for imagenet ('test' is also available but without labels)
        :param number_of_images: the number of images. None for full dataset.
        :param augmentation: 0 for no augmentation (only original), 1 for original + 1 augmented dataset, 2 for original + 2 augmented
                                -1 for only 1 augmented dataset (no original), -2 for only 2 augmented datasets (no original)
        :param shuffle: whether the dataset should be shuffled. Should be true for training datasets.
        :param repeat: whether the dataset should be repeated infinitely.
        :param cache: whether the dataset should be cached
        :param jpeg_quality: the JPEG quality to use
        :param chroma_subsampling: whether to use chroma subsampling or not
        """
        self._jpeg_quality = jpeg_quality
        self.chroma_subsampling = chroma_subsampling
        self.rgb_to_jpeg_model = adversarial_attacks.models.rgb_to_jpeg.RGBToJpegModel(dataset_name,
                                                                                       self._jpeg_quality,
                                                                                       round='round',
                                                                                       chroma_subsampling=
                                                                                       self.chroma_subsampling, )
        super().__init__(dataset_name, train_or_test, number_of_images, augmentation, shuffle=shuffle, repeat=repeat,
                         cache=cache)

        Y_shape = self.not_subsampled_dct_shape
        C_shape = self.subsampled_dct_shape if self.chroma_subsampling else self.not_subsampled_dct_shape

        Y_w, Y_h = Y_shape[0], Y_shape[1]
        C_w, C_h = C_shape[0], C_shape[1]

        output_shape_Y = [Y_w, Y_h, 64]
        output_shape_C = [C_w, C_h, 64]

        self.output_signature = (
            (tf.TensorSpec(shape=output_shape_Y),
             tf.TensorSpec(shape=output_shape_C),
             tf.TensorSpec(shape=output_shape_C)
             )
            ,
            tf.TensorSpec(shape=(self.num_classes()))
        )

        self.batched_output_signature = (
            (tf.TensorSpec(shape=[None] + output_shape_Y
                           ),
             tf.TensorSpec(shape=[None] + output_shape_C),
             tf.TensorSpec(shape=[None] + output_shape_C),
             )
            ,
            tf.TensorSpec(shape=(None,
                                 self.num_classes()))
        )

    def _cache_key(self):
        return f'{super()._cache_key()}_jq_{self._jpeg_quality}_cs_{self.chroma_subsampling}'

    def _tf_dataset_key(self):
        return 'jpeg{}jq{}cs{}'.format(super()._tf_dataset_key(),
                                       self._jpeg_quality,
                                       self.chroma_subsampling)

    @tf.function
    def rgb_to_datatype_transformation(self, rgb):
        """
        Transforms an rgb image to quantized JPEG data.
        :param rgb:
        :return:
        """
        Y, Cb, Cr = self.rgb_to_jpeg_model(rgb)
        return Y, Cb, Cr

    def datatype_to_rgb_transformation(self, datatype_image):
        return adversarial_attacks.utils.transformation.jpeg_to_rgb_batch(datatype_image, self.dataset_name,
                                                                          self._jpeg_quality, self.chroma_subsampling)


class YCbCrDataset(Dataset):
    """
    Dataset of YCbCr pixels.
    """
    _ds_type = 'ycbcr'

    def __init__(self, dataset_name, train_or_test, number_of_images=None, augmentation: int = 0,
                 shuffle=True, repeat=True, cache=None):
        """
        :param dataset_name: the name of the dataset, 'cifar10' or 'imagenet'
        :param train_or_test: the type of the dataset
                              'train', 'test' for cifar10 or
                              'train', 'validation' for imagenet ('test' is also available but without labels)
        :param number_of_images: the number of images. None for full dataset.
        :param augmentation: 0 for no augmentation (only original), 1 for original + 1 augmented dataset, 2 for original + 2 augmented
                                -1 for only 1 augmented dataset (no original), -2 for only 2 augmented datasets (no original)
        :param shuffle: whether the dataset should be shuffled. Should be true for training datasets.
        :param repeat: whether the dataset should be repeated infinitely.
        :param cache: whether the dataset should be cached
        """
        super().__init__(dataset_name, train_or_test,
                         number_of_images=number_of_images, augmentation=augmentation,
                         shuffle=shuffle, repeat=repeat, cache=cache)

        w, h = self.rgb_img_shape[0], self.rgb_img_shape[1]
        output_shape = [w, h]

        self.output_signature = (
            (tf.TensorSpec(shape=(
                w, h
            )),
             tf.TensorSpec(shape=(
                 w, h
             )),
             tf.TensorSpec(shape=(
                 w, h
             ))
            )
            ,
            tf.TensorSpec(shape=(self.num_classes()))
        )

        self.batched_output_signature = (
            (tf.TensorSpec(shape=(
                None,
                w, h
            )),
             tf.TensorSpec(shape=(
                 None,
                 w, h
             )),
             tf.TensorSpec(shape=(
                 None,
                 w, h
             ))
            )
            ,
            tf.TensorSpec(shape=(None,
                                 self.num_classes()))
        )

    def _tf_dataset_key(self):
        return 'ycbcr_{}'.format(super()._tf_dataset_key())

    @tf.function
    def rgb_to_datatype_transformation(self, rgb):
        """
        Converts an rgb image batch to YCbCr pixels.
        :param rgb:
        :return:
        """
        Y, Cb, Cr = adversarial_attacks.utils.transformation.rgb_to_ycbcr_tuple(rgb)
        return Y, Cb, Cr

    def datatype_to_rgb_transformation(self, datatype_image):
        return adversarial_attacks.utils.transformation.ycbcr_tuple_to_rgb((datatype_image))
