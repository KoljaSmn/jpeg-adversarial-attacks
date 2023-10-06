# Loads the  Berkeley Adobe Perceptual Patch Similarity (BAPPS) dataset as tf.data.Dataset
# See https://github.com/richzhang/PerceptualSimilarity for information on the dataset

import os
import numpy as np
import tensorflow as tf
import math
import wget
import tarfile
from abc import abstractmethod, ABC

from adversarial_attacks.utils.general import makedirs
from adversarial_attacks.config.config import Config
from adversarial_attacks.utils.logging import info

_img_w, _img_h = Config.INPUT_SHAPE['bapps']['rgb'][:2]
_bapps_dir = Config.BAPPS_DIR
_afc2_dir = os.path.join(_bapps_dir, '2afc')
_jnd_dir = os.path.join(_bapps_dir, 'jnd')
_downloaded_files = {'jnd_val': os.path.join(_bapps_dir, 'downloaded_jnd_val'),
                    '2afc_val': os.path.join(_bapps_dir, 'downloaded_2afc_val'),
                    '2afc_train': os.path.join(_bapps_dir, 'downloaded_2afc_train'),}# files will be created when the dataset download is completed


def _index_to_string(i):
    return '%06d' % i


def download_and_extract_bapps_dataset(typ):
    makedirs(_bapps_dir)

    info_str = 'Berkeley Adobe Perceptual Patch Similarity (BAPPS). ' + \
               'See https://github.com/richzhang/PerceptualSimilarity for information on the dataset.'

    with open(os.path.join(_bapps_dir, 'readme.txt'), 'w') as f:
        f.write(info_str)

    makedirs(_afc2_dir)
    makedirs(_jnd_dir)

    if typ == 'jnd_val':
        # download jnd dataset
        info('Downloading JND dataset')
        wget.download("https://people.eecs.berkeley.edu/~rich.zhang/projects/2018_perceptual/dataset/jnd.tar.gz",
                      _bapps_dir)
        # open tar file
        file = tarfile.open(os.path.join(_bapps_dir, 'jnd.tar.gz'))
        # extracting file
        info('Extracting JND dataset')
        file.extractall(_bapps_dir)
        file.close()
        os.remove(os.path.join(_bapps_dir, 'jnd.tar.gz'))
        info('Downloaded and extracted JND dataset')
        
    
    elif typ == '2afc_val':
        # download 2afc val dataset
        info('Downloading 2afc val dataset')
        wget.download("https://people.eecs.berkeley.edu/~rich.zhang/projects/2018_perceptual/dataset/twoafc_val.tar.gz",
                      _bapps_dir)
        # open tar file
        file = tarfile.open(os.path.join(_bapps_dir, 'twoafc_val.tar.gz'))
        # extracting file
        info('Extracting 2afc val dataset')
        file.extractall(_afc2_dir)
        file.close()
        os.remove(os.path.join(_bapps_dir, 'twoafc_val.tar.gz'))
        info('Downloaded and extracted 2afc val dataset')
        
    elif typ == '2afc_train':
        # download 2afc train dataset
        info('Downloading 2afc train dataset')
        wget.download("https://people.eecs.berkeley.edu/~rich.zhang/projects/2018_perceptual/dataset/twoafc_train.tar.gz",
                      _bapps_dir)
        # open tar file
        file = tarfile.open(os.path.join(_bapps_dir, 'twoafc_train.tar.gz'))
        # extracting file
        file.extractall(_afc2_dir)
        file.close()
        os.remove(os.path.join(_bapps_dir, 'twoafc_train.tar.gz'))
        info('Downloaded and extracted 2afc train dataset')
    open(_downloaded_files[typ], 'a').close()


def check_and_download_and_extract_bapps_dataset(typ):
    if not os.path.exists(_downloaded_files[typ]):
        download_and_extract_bapps_dataset(typ)


class _BappsDataset(ABC):
    """
    Abstract class to load the BAPPS dataset from Zhang et al.:
    The Unreasonable Effectiveness of Deep Features as a Perceptual Metric, CVPR 2018
    """

    _test_set_name = 'val'
    _val_choices, _train_choices = None, None
    _ds_type_dir = None
    _output_signature = None

    @abstractmethod
    def __init__(self, train_or_test, choices=None):
        self._typ = 'train' if train_or_test == 'train' else self._test_set_name
        check_and_download_and_extract_bapps_dataset(f'{self._ds_type}_{self._typ}')
        self._dataset_dir = os.path.join(self._ds_type_dir, self._typ)
        if choices:
            for choice in choices:
                if (self._typ == 'train' and choice not in self._train_choices) or \
                        (self._typ == self._test_set_name and choice not in self._val_choices):
                    raise ValueError('Choice {} unknown for type {}.'.format(choice, self._typ))

        else:
            choices = self._train_choices if self._typ == 'train' else self._val_choices

        self._choices = choices

        self._choice_datasets_generators = {
            self._choices[i]: (
                self._choice_generator(self._choices[i]), self._n_images_for_choice(self._choices[i])) for i in
            range(len(self._choices))}

        def merged_generator():
            # https://stackoverflow.com/questions/243865/how-do-i-merge-two-python-iterators
            i_per_choice = {choices[i]: 0 for i in range(len(choices))}

            while True:
                choice_found = False
                for choice in self._choice_datasets_generators:
                    if i_per_choice[choice] >= self._choice_datasets_generators[choice][1]:
                        continue
                    i_per_choice[choice] = i_per_choice[choice] + 1
                    choice_found = True
                    yield next(self._choice_datasets_generators[choice][0])
                if not choice_found:
                    i_per_choice = {choices[i]: 0 for i in range(len(choices))}

        self.ds = tf.data.Dataset.from_generator(merged_generator, output_signature=self._output_signature). \
            take(self.n_images()).prefetch(tf.data.AUTOTUNE)

    @abstractmethod
    def _choice_generator(self, choice):
        pass

    def n_images(self):
        n = 0
        for choice in self._choices:
            n += self._n_images_for_choice(choice)

        return n

    def n_batches(self, bs):
        return math.ceil(self.n_images() / bs)

    def _n_images_for_choice(self, choice):
        ds_dir = os.path.join(self._dataset_dir, choice)
        return len(os.listdir(os.path.join(ds_dir, 'p0')))


class AFC2(_BappsDataset):
    """
    Loads the AFC2 dataset from Zhang et al.:
    The Unreasonable Effectiveness of Deep Features as a Perceptual Metric, CVPR 2018

    An entry consists of an original image, two perturbed images p0, p1 and a number in [0, 1]
    If all human judges preferred p0, the number is 0,
    if all human judges preferred p1, the number is 1.
    The number is the percentage of judges that preferred p1 over p0
    """

    _train_choices = ['traditional', 'cnn']
    _val_choices = ['traditional', 'cnn', 'superres', 'deblur', 'color', 'frameinterp']
    _ds_type_dir = _afc2_dir
    _output_signature = (tf.TensorSpec(shape=(_img_w, _img_h, 3), dtype=tf.float32),
                         tf.TensorSpec(shape=(_img_w, _img_h, 3), dtype=tf.float32),
                         tf.TensorSpec(shape=(_img_w, _img_h, 3), dtype=tf.float32),
                         tf.TensorSpec(shape=(1), dtype=tf.float32))
    _ds_type = '2afc'

    def __init__(self, train_or_test, choices=None):
        super(AFC2, self).__init__(train_or_test, choices)

    def _choice_generator(self, choice):
        ds_dir = os.path.join(self._dataset_dir, choice)
        n_images = self._n_images_for_choice(choice)

        def generator():
            i = 0

            while True:
                filename = _index_to_string(i % n_images)

                judge = np.load(os.path.join(ds_dir, 'judge', filename + '.npy'))
                ref = tf.io.decode_png(tf.io.read_file(os.path.join(ds_dir, 'ref', filename + '.png')),
                                       channels=3)  # channels=3 to output RGB image, otherwise RGBA would be used
                p0 = tf.io.decode_png(tf.io.read_file(os.path.join(ds_dir, 'p0', filename + '.png')), channels=3)
                p1 = tf.io.decode_png(tf.io.read_file(os.path.join(ds_dir, 'p1', filename + '.png')), channels=3)

                ref, p0, p1 = tf.image.resize(ref, [_img_w, _img_h]), \
                              tf.image.resize(p0, [_img_w, _img_h]), \
                              tf.image.resize(p1, [_img_w, _img_h])

                yield ref, p0, p1, judge
                i += 1

        return generator()


class JND(_BappsDataset):
    """
    Loads the JND dataset from Zhang et al.:
    The Unreasonable Effectiveness of Deep Features as a Perceptual Metric, CVPR 2018

    An entry consists of two images p0, p1 and a number in [0, 1]
    If all human judges thought p0 and p1 are not the same image, the number is 0,
    if all human judges preferred thought p0 and p1 are the same image, the number is 1.
    The number is the percentage of judges that thought p0, p1 were the same image
    Only a validation dataset is available
    """

    _val_choices = ['traditional', 'cnn']
    _ds_type_dir = _jnd_dir

    _output_signature = (tf.TensorSpec(shape=(_img_w, _img_h, 3), dtype=tf.float32),
                         tf.TensorSpec(shape=(_img_w, _img_h, 3), dtype=tf.float32),
                         tf.TensorSpec(shape=(), dtype=tf.float32))
    _ds_type = 'jnd'

    def __init__(self, choices=None):
        super().__init__('val', choices)

    def _choice_generator(self, choice):
        ds_dir = os.path.join(self._dataset_dir, choice)
        n_images = self._n_images_for_choice(choice)

        def generator():
            i = 0
            while True:
                filename = _index_to_string(i % n_images)

                same = np.load(os.path.join(ds_dir, 'same', filename + '.npy'))
                # channels=3 to output RGB image, otherwise RGBA would be used
                p0 = tf.io.decode_png(tf.io.read_file(os.path.join(ds_dir, 'p0', filename + '.png')), channels=3)
                p1 = tf.io.decode_png(tf.io.read_file(os.path.join(ds_dir, 'p1', filename + '.png')), channels=3)

                p0, p1 = tf.image.resize(p0, [_img_w, _img_h]), tf.image.resize(p1, [_img_w, _img_h])

                yield p0, p1, same
                i += 1

        return generator()
