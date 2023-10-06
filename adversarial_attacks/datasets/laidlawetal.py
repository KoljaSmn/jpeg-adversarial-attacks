from adversarial_attacks.config.config import Config
import os
import wget
from adversarial_attacks.utils.general import makedirs
from adversarial_attacks.utils.logging import info
import zipfile
import pandas as pd
import tensorflow as tf
import math

import json

_ds_dir = Config.LAIDLAW_DIR
_downloaded_file = os.path.join(_ds_dir, 'downloaded')
_human_annotations_csv = os.path.join(_ds_dir, 'human_annotations.csv')
_pairs_json = os.path.join(_ds_dir, 'images', 'pairs.json')
_pair_annotations_json = os.path.join(_ds_dir, 'pair_annotations.json')


def get_type_from_path(p):
    path = os.path.normpath(p)
    return path.split(os.sep)[-1].split('.')[0]


def str_for_types(type1, type2):
    if type1 == 'original' and type2 == 'original':
        return '00'
    if type1 == 'original' and type2 == 'adversarial':
        return '01'
    if type1 == 'adversarial' and type2 == 'adversarial':
        return '11'
    if type1 == 'adversarial' and type2 == 'original':
        return '10'


def types_for_str(st):
    if st == '00':
        return 'original', 'original'
    if st == '01':
        return 'original', 'adversarial'
    if st == '11':
        return 'adversarial', 'adversarial'
    if st == '10':
        return 'adversarial', 'original'


def key_for_id_and_types(_id, type1, type2):
    return _id + '_' + str_for_types(type1, type2)


def id_and_types_for_key(_key):
    _id, types_string = _key.split('_')
    types = types_for_str(types_string)
    return _id, types[0], types[1]


def download_and_prepare_dataset():
    makedirs(_ds_dir)

    # download jnd dataset
    info(
        'Downloading Datasets Laidlaw et al.: Perceptual Adversarial Robustness: Defense Against Unseen Threat Models dataset')
    wget.download("https://perceptual-advex.s3.us-east-2.amazonaws.com/perceptual-advex-perceptual-study-data.zip",
                  _ds_dir)

    info('Extracting dataset')
    _zip_file = os.path.join(_ds_dir, 'perceptual-advex-perceptual-study-data.zip')
    # extract zip file
    with zipfile.ZipFile(_zip_file, 'r') as zip_ref:
        zip_ref.extractall(_ds_dir)

    info('Extracted dataset')
    os.remove(_zip_file)

    info('Preparing annotations')
    with open(_pairs_json, 'r') as f:
        pairs_data = json.load(f)

    _pairs_data_dict = {}
    for entry in pairs_data:
        _id = entry['id']
        _pairs_data_dict[_id] = entry

    annotations_count = {}  # id_type_str: (n_all, n_same)  # type_str -> str_for_types

    df = pd.read_csv(_human_annotations_csv)
    for index, row in df.iterrows():
        _id, _pair_type, _choice = row['pair_id'], row['pair_type'], row['choice']
        _img1_type, _img2_type = get_type_from_path(row['original']), get_type_from_path(row['adversarial'])
        _choice = 1. if _choice == 'same' else 0.

        _key = key_for_id_and_types(_id, _img1_type, _img2_type)

        if _key not in annotations_count:
            annotations_count[_key] = (1, _choice)
        else:
            n_old, same_old = annotations_count[_key]
            annotations_count[_key] = (n_old + 1, same_old + _choice)

    # {_id = {'same': value in [0, 1], 'data': pairs_data}}
    pairs_and_annotation_data = {_key: {'same': annotations_count[_key][1] / annotations_count[_key][0],
                                        'n_users': annotations_count[_key][0],
                                        'sum_annotations': annotations_count[_key][1]}
                                 for _key in annotations_count}

    for _key in pairs_and_annotation_data:
        _id, _, _ = id_and_types_for_key(_key)
        pairs_and_annotation_data[_key]['pair_data'] = _pairs_data_dict[_id]

    with open(_pair_annotations_json, 'w') as f:
        json.dump(pairs_and_annotation_data, f, indent=4)

    open(_downloaded_file, 'a').close()


def check_and_download_and_extract_dataset():
    if not os.path.exists(_downloaded_file):
        download_and_prepare_dataset()


class LaidlawEtAlDataset:
    """
    Loads the dataset from Laidlaw et al.:
    Perceptual Adversarial Robustness: Defense Against Unseen Threat Models, ICLR 2021
    as a tf.data.Dataset.

    Every entry contains an image pair,the human evaluation (same value in [0, 1]) -> 1 image are perceived as the same
    and the truth value in {0, 1} -> is the image the same or not? 1->same

    """

    def __init__(self, perturbation_sizes=['small', 'medium', 'large'], include_same=True):
        check_and_download_and_extract_dataset()
        # Opening JSON file
        with open(_pair_annotations_json) as f:
            # returns JSON object as
            # a dictionary
            self._data = json.load(f)

        _img_w, _img_h = 224, 224
        _output_signature = (tf.TensorSpec(shape=(_img_w, _img_h, 3), dtype=tf.float32),
                             tf.TensorSpec(shape=(_img_w, _img_h, 3), dtype=tf.float32),
                             tf.TensorSpec(shape=(), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32))
    
        self._perturbation_sizes = perturbation_sizes
        self._include_same = include_same
        self._n_images = self._count_n_images()
        self.ds = tf.data.Dataset.from_generator(self._gen, output_signature=_output_signature)

    def _count_n_images(self):
        i = 0
        for _key, entry in self._data.items():
            _id, type1, type2 = id_and_types_for_key(_key)

            if type1 == type2:
                same_truth = 1.
            else:
                same_truth = 0.

            if entry['pair_data']['size'] not in self._perturbation_sizes:
                continue

            if not self._include_same and same_truth == 1.:
                continue

            i += 1
        return i

    def _gen(self):
        def load_image_from_path(p):
            return tf.io.decode_png(tf.io.read_file(p), channels=3)

        for _key, entry in self._data.items():
            _id, type1, type2 = id_and_types_for_key(_key)
            
            if type1 == type2:
                same_truth = 1.
            else:
                same_truth = 0.

            if not self._include_same and same_truth == 1.:
                continue

            same_human = entry['same']

            if entry['pair_data']['size'] not in self._perturbation_sizes:
                continue

            original_path, adversarial_path = os.path.join(_ds_dir, 'images',
                                                           entry['pair_data']['original']), os.path.join(_ds_dir,
                                                                                                         'images',
                                                                                                         entry[
                                                                                                             'pair_data'][
                                                                                                             'adversarial'])

            img1 = load_image_from_path(original_path if type1 == 'original' else adversarial_path)
            img2 = load_image_from_path(original_path if type2 == 'original' else adversarial_path)

            yield img1, img2, same_human, same_truth

        return self._gen()

    def n_images(self):
        return self._n_images

    def n_batches(self, bs):
        return math.ceil(self.n_images() / bs)
