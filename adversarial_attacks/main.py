import logging as pylogging
import os
import traceback
import warnings
from sys import platform

import torch
import tensorflow as tf

from adversarial_attacks.config.config import Config
from adversarial_attacks.config.tf_distribution import init_strategy

from adversarial_attacks.utils.general import makedirs

makedirs(Config.LOG_DIR)
makedirs(Config.TENSORBOARD_LOG_DIR)

import adversarial_attacks.utils.logging as logging


def init(use_cpu=False, gpu_nrs=None, run_eagerly=False, tf_strategy='default', enable_warnings=False):
    if not enable_warnings:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(1)
        warnings.filterwarnings('ignore')
        tf.get_logger().setLevel(pylogging.ERROR)
        tf.config.run_functions_eagerly(run_eagerly)
    
    if use_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # Hide GPU from visible devices
        tf.config.set_visible_devices([], 'GPU')
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                if gpu_nrs is not None:
                    tf.config.experimental.set_visible_devices([gpus[gpu_nr] for gpu_nr in gpu_nrs], 'GPU')
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                gpu_config_str = str(len(gpus)) + " Physical GPUs, " + str(len(logical_gpus)) + " Logical GPUs"
                logging.info(gpu_config_str)
            except RuntimeError as e:
                logging.error(e)
                print(traceback.format_exc())

    init_strategy(tf_strategy)

    import adversarial_attacks.utils.general as utils

    if not utils.check_linux():
        logging.warning(
            'Not all functions will work in {} because torchjpeg is only availabe on linux.'.format(platform))
