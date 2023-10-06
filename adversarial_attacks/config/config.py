import json
import os
from adversarial_attacks.utils.general import makedirs, remove_dir


def load_json(file):
    f = open(file)
    data = json.load(f)
    f.close()
    return data


class Config:
    LOGGING_LEVEL = 'INFO'

    DATA_DIR = '../data/jpeg_adversarial_attack/'
    MODEL_PATH = '../models/jpeg_adversarial_attack'
    LPIPS_MODEL_PATH = os.path.join(MODEL_PATH, 'lpips')
    LPIPS_LOSS_MODEL_PATH = os.path.join(LPIPS_MODEL_PATH, 'loss_networks')
    LPIPS_TRAINING_MODEL_PATH = os.path.join(LPIPS_MODEL_PATH, 'training_models')

    LOG_DIR = '../logs/jpeg_adversarial_attack'

    DATASETS_DIR = '../data/datasets/jpeg-adversarial-attacks'
    TF_DATASETS_DIR = '../data/datasets/tensorflow_datasets'
    DATASETS_CACHE_DIR = os.path.join(DATASETS_DIR, 'cached/jpeg_adversarial_attacks')
    BAPPS_DIR = os.path.join(DATASETS_DIR, 'bapps')
    LAIDLAW_DIR = os.path.join(DATASETS_DIR, 'laidlawetal')
    FEZZA_DIR = os.path.join(DATASETS_DIR, 'fezzaetal')
    ILYAS_DIR = os.path.join(DATASETS_DIR, 'ilyasetal')

    DATASET_MODELS = {'cifar10': os.path.join(MODEL_PATH, 'cifar10'),
                      'imagenet': os.path.join(MODEL_PATH, 'imagenet'),
                      'bapps': os.path.join(MODEL_PATH, 'bapps')}

    TENSORBOARD_LOG_DIR = os.path.join(LOG_DIR, 'tensorboard')
    LOG_FILE = os.path.join(LOG_DIR, 'adversarial_attacks.log')


    QUANTIZATION_JSON = os.path.join(DATA_DIR, 'quantization.json')
    DIMENSION_JSON = os.path.join(DATA_DIR, 'dimension.json')
    TEMP_DATA_DIR = os.path.join(DATA_DIR, 'temp/')

    PREPARATION_BATCH_SIZE = {'cifar10': 500, 'imagenet': 200}

    INPUT_SHAPE = {'cifar10': {'rgb': (32, 32, 3), 'Y': (4, 4, 64), 'C': (2, 2, 64)},
                   'imagenet': {'rgb': (224, 224, 3), 'Y': (28, 28, 64), 'C': (14, 14, 64)},
                   'bapps': {'rgb': (256, 256, 3), 'Y': (32, 32, 64), 'C': (16, 16, 64)},
                   'laidlawetal': {'rgb': (224, 224, 3), 'Y': (28, 28, 64), 'C': (14, 14, 64)},
                   }

    IMAGENET_LABELS_FILE = os.path.join('./adversarial_attacks/config', 'imagenet_labels.json')
    CIFAR_LABELS_FILE = os.path.join('./adversarial_attacks/config', 'cifar_labels.json')

    LABELS = {'cifar10': load_json(CIFAR_LABELS_FILE),
              'imagenet': load_json(IMAGENET_LABELS_FILE)}

    DATASET_LOADING_NAME = {'cifar10': 'cifar10', 'imagenet': 'imagenet2012'}
    DATASET_NUM_CLASSES = {'cifar10': 10, 'imagenet': 1000, 'bapps': 0}
    DATASET_SIZES = {'cifar10': {'train': 50000, 'test': 10000},
                     'imagenet': {'train': 1281167, 'validation': 50000, 'test': 100000}
                     }

    remove_dir(TEMP_DATA_DIR)
    # create all directories
    _dirs_to_create = [DATA_DIR, MODEL_PATH, LPIPS_MODEL_PATH, LOG_DIR, DATASETS_DIR, TF_DATASETS_DIR,
                       DATASETS_CACHE_DIR, BAPPS_DIR,
                       TENSORBOARD_LOG_DIR, TEMP_DATA_DIR]
    _dirs_to_create += list(DATASET_MODELS.values())
    for dir_name in _dirs_to_create:
        makedirs(dir_name)
