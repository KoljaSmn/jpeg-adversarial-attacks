"""
Loads the datasets from Ilyas et al., 2019:  Adversarial Examples Are Not Bugs, They Are Features, NeurIPS 2019
as tf.data.Dataset
"""

import wget
from adversarial_attacks.config.config import Config

import os
from adversarial_attacks.utils.general import makedirs
import tarfile
from adversarial_attacks.utils.logging import info
import torch as ch
import numpy as np
import tensorflow as tf

makedirs(Config.ILYAS_DIR)

downloaded_file = os.path.join(Config.ILYAS_DIR, 'downloaded')


def download_and_extract():
    tar_filename = os.path.join(Config.ILYAS_DIR, 'ilyasetal.tar')
    info('Downloading Ilyas et al. dataset')
    wget.download('http://andrewilyas.com/datasets.tar',
                  tar_filename)

    # open tar file
    file = tarfile.open(tar_filename)
    # extracting file
    info('Extracting Ilyas et al. dataset')
    file.extractall(Config.ILYAS_DIR)
    file.close()
    os.remove(tar_filename)
    open(downloaded_file, 'a').close()


def check_if_downloaded_and_do():
    if not os.path.exists(downloaded_file):
        download_and_extract()


def get_typ_dir(typ):
    return os.path.join(Config.ILYAS_DIR, 'release_datasets', typ)


def get_extraction_path_for_typ(typ):
    return os.path.join(get_typ_dir(typ), 'extracted')


class IlyasEtAlDataset:
    """
    Loads the datasets from Ilyas et al., 2019:  Adversarial Examples Are Not Bugs, They Are Features, NeurIPS 2019
    as tf.data.Dataset
    """

    def __init__(self, typ):
        """
        typ in d_non_robust_CIFAR, d_robust_CIFAR, ddet_CIFAR, drand_CIFAR
        """
        valid_options = ['d_non_robust_CIFAR', 'd_robust_CIFAR', 'ddet_CIFAR', 'drand_CIFAR']
        if typ not in valid_options:
            raise ValueError('typ must be in {}'.format(valid_options))

        check_if_downloaded_and_do()

        self._typ = typ

        train_data = ch.cat(ch.load(os.path.join(get_typ_dir(typ), f"CIFAR_ims")))
        train_labels = ch.cat(ch.load(os.path.join(get_typ_dir(typ), f"CIFAR_lab")))

        np_imgs = tf.cast(np.transpose(train_data, [0, 2, 3, 1]) * 255., tf.uint8)
        np_labels = np.array(train_labels)
        one_hot_targets = np.eye(10)[np_labels.reshape(-1)]

        self.ds = tf.data.Dataset.from_tensor_slices((np_imgs, one_hot_targets))
