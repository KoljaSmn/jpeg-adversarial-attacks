import codecs
import json
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm.auto import tqdm

import adversarial_attacks.utils.general
from adversarial_attacks.config.config import Config
from adversarial_attacks.utils import logging as logging
from adversarial_attacks.utils.general import round_or_approx, shin_rounding_approximation

# indices to use a convolution to (un-)zig-zag and flatten/unflatten the coefficients between shape 8x8 and 64
zigzag_indices = np.array([
    0, 1, 5, 6, 14, 15, 27, 28,
    2, 4, 7, 13, 16, 26, 29, 42,
    3, 8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
])

# initialized the convolutional filters to flatten or unflatten (inverse) coefficients
unzigzag_filter, zigzag_filter = np.zeros((1, 1, 64, 64), np.float32), np.zeros((1, 1, 64, 64), np.float32)
for i in range(tf.shape(zigzag_indices)[0]):
    unzigzag_filter[0][0][zigzag_indices[i]][i] = 1.0
    zigzag_filter[0][0][i][zigzag_indices[i]] = 1.0
unzigzag_filter, zigzag_filter = tf.constant(unzigzag_filter), tf.constant(zigzag_filter)


def block_dct(x):
    """
    Converts tensorflow's 1D dct to 2D dct.
    See https://stackoverflow.com/questions/60151359/create-keras-tensorflow-layer-that-computes-2d-dct
    """
    norm = 'ortho'
    X1 = tf.signal.dct(x, type=2, norm=norm)
    X1_t = tf.transpose(X1, perm=[0, 1, 2, 4, 3])
    X2 = tf.signal.dct(X1_t, type=2, norm=norm)
    X2_t = tf.transpose(X2, perm=[0, 1, 2, 4, 3])
    return X2_t


def block_idct(x):
    """
    Converts tensorflow's 1D idct to 2D idct.
    """
    norm = 'ortho'
    X2_t = tf.transpose(x, perm=[0, 1, 2, 4, 3])
    X2 = tf.signal.idct(X2_t, type=2, norm=norm)
    X1_t = tf.transpose(X2, perm=[0, 1, 2, 4, 3])
    X1 = tf.signal.idct(X1_t, type=2, norm=norm)
    return X1


def unzigzag(coefficients):
    """
    :param coefficients: shape (n, w, h, 64)

    returns:
            shape (n, w, h, 64)
    """
    return tf.nn.conv2d(coefficients, unzigzag_filter, 1, padding='VALID')


def zigzag(coefficients):
    """
    :param coefficients: shape (n, w, h, 64)

    returns:
            shape (n, w, h, 64)
    """
    return tf.nn.conv2d(coefficients, zigzag_filter, 1, padding='VALID')


def tf_batch_unzigzag_and_unflatten(coefficients):
    """
    :param coefficients: shape (n, w, h, 64)

    returns:
            shape (n, w, h, 8, 8)
    based on https://queuecumber.gitlab.io/torchjpeg/_modules/torchjpeg/dct.html, function zigzag
    """

    shape = [tf.shape(coefficients)[k] for k in range(3)]
    coefficients = unzigzag(coefficients)
    coefficients = tf.reshape(coefficients, [shape[0], shape[1], shape[2], 8, 8])
    return coefficients


def tf_batch_zigzag_and_flatten(coefficients):
    """
    :param coefficients: shape (n, w, h, 8, 8)
    returns:
            shape (n, w, h, 64)
    based on https://queuecumber.gitlab.io/torchjpeg/_modules/torchjpeg/dct.html, function zigzag
    """
    shape = [tf.shape(coefficients)[k] for k in range(3)]
    coefficients = tf.reshape(coefficients, [shape[0], shape[1], shape[2], 64])
    coefficients = zigzag(coefficients)

    return coefficients


def __tf_flatten_unflatten_helper(coefficients, func):
    """
    Wraps the unflatten or flatten function.

    :param func: one of tf_batch_flatten, tf_batch_unflatten
    :param coefficients: batched (n, w, h, 64) or (n, w, h, 8, 8)
    :return:
    """
    coefficients = tf.cast(coefficients, tf.float32)
    coefficients = func(coefficients)
    return coefficients


def tf_zigzag_and_flatten_coefficients(coefficients, batched):
    """
    :param coefficients: unbatched (w, h, 8, 8) or batched (bs, w, h, 8, 8)
    :param batched: whether the coefficients are batches or not.
    :return:
    """
    if not batched:
        coefficients = tf.expand_dims(coefficients, axis=0)
    out = __tf_flatten_unflatten_helper(coefficients, tf_batch_zigzag_and_flatten)
    return out if batched else out[0]


def tf_unzigzag_unflatten_coefficients(coefficients, batched):
    """

    :param coefficients: unbatched (w, h, 64)
    :param batched: whether the coefficients are batches or not.
    :return:
    """
    if not batched:
        coefficients = tf.expand_dims(coefficients, axis=0)

    out = __tf_flatten_unflatten_helper(coefficients, tf_batch_unzigzag_and_unflatten)
    return out if batched else out[0]


def quantize(unquantized_Y, unquantized_Cb, unquantized_Cr, jpeg_quality: int, round: str = 'round'):
    """
    Quantizes the coefficients with the given jpeg quality.

    The input and output coefficients are expected to be **not** zigzagged.

    :param unquantized_Y: tensor of shape (w/8, h/8, 8, 8) or (bs, w/8, h/8, 8, 8)
    :param unquantized_Cb: tensor of shape (w/16, h/16, 8, 8) or (bs, w/16, h/16, 8, 8)
    :param unquantized_Cr: tensor of shape (w/16, h/16, 8, 8) or (bs, w/16, h/16, 8, 8)
    :param jpeg_quality: the jpeg quality to quantize the data.
    :return:
    """
    quant = quantization_matrix(jpeg_quality)
    Y = unquantized_Y / quant[0]
    Cb = unquantized_Cb / quant[1]
    Cr = unquantized_Cr / quant[2]

    Y = round_or_approx(Y, round)
    Cb = round_or_approx(Cb, round)
    Cr = round_or_approx(Cr, round)

    return Y, Cb, Cr


def dequantize(unquantized_Y, unquantized_Cb, unquantized_Cr, jpeg_quality: int, round: str = 'round'):
    """
    Dequantizes the coefficients with the given jpeg quality.

    The input and output coefficients are expected to be **not** zigzagged.

    :param unquantized_Y: tensor of shape (w/8, h/8, 8, 8) or (bs, w/8, h/8, 8, 8)
    :param unquantized_Cb: tensor of shape (w/16, h/16, 8, 8) or (bs, w/16, h/16, 8, 8)
    :param unquantized_Cr: tensor of shape (w/16, h/16, 8, 8) or (bs, w/16, h/16, 8, 8)
    :param jpeg_quality: the jpeg quality to quantize the data.
    :return:
    """
    quant = quantization_matrix(jpeg_quality)
    Y = unquantized_Y * quant[0]
    Cb = unquantized_Cb * quant[1]
    Cr = unquantized_Cr * quant[2]

    Y = round_or_approx(Y, round)
    Cb = round_or_approx(Cb, round)
    Cr = round_or_approx(Cr, round)

    return Y, Cb, Cr


def quantize_zigzagged_coefficients(unquantized_Y, unquantized_Cb, unquantized_Cr, jpeg_quality: int,
                                    batched: bool = False, round: str = 'round'):
    """
    Quantizes the coefficients with the given jpeg quality.

    The input and output coefficients are expected to be zigzagged.

    :param unquantized_Y: tensor of shape (w/8, h/8, 64) or (bs, w/8, h/8, 64)
    :param unquantized_Cb: tensor of shape (w/16, h/16, 64) or (bs, w/16, h/16, 64)
    :param unquantized_Cr: tensor of shape (w/16, h/16, 64) or (bs, w/16, h/16, 64)
    :param jpeg_quality: the jpeg quality to quantize the data.
    :param batched: whether the input coefficients are batched or not.
    :return:
    """
    qm_zigzagged_Y, qm_zigzagged_Cb, qm_zigzagged_Cr = zigzagged_quantization_matrix(jpeg_quality)

    Y, Cb, Cr = unquantized_Y / qm_zigzagged_Y, unquantized_Cb / qm_zigzagged_Cb, unquantized_Cr / qm_zigzagged_Cr

    Y = round_or_approx(Y, round)
    Cb = round_or_approx(Cb, round)
    Cr = round_or_approx(Cr, round)

    return Y, Cb, Cr


def quantize_zigzagged_coeff_tuple(unquantized, jpeg_quality: int, batched: bool = False, round: str = 'round'):
    return quantize_zigzagged_coefficients(unquantized[0], unquantized[1], unquantized[2], jpeg_quality,
                                           batched=batched, round=round)


def dequantize_zigzagged_coefficients(quantized_Y, quantized_Cb, quantized_Cr, jpeg_quality, batched=False,
                                      round: str = 'round'):
    """
    Dequantizes the coefficients with the given jpeg quality.

    The input and output coefficients are expected to be zigzagged.

    :param quantized_Y: tensor of shape (w/8, h/8, 64) or (bs, w/8, h/8, 64)
    :param quantized_Cb: tensor of shape (w/16, h/16, 64) or (bs, w/16, h/16, 64)
    :param quantized_Cr: tensor of shape (w/16, h/16, 64) or (bs, w/16, h/16, 64)
    :param jpeg_quality: the jpeg quality to quantize the data.
    :param batched: whether the input coefficients are batched or not.
    :return:
    """

    qm_zigzagged_Y, qm_zigzagged_Cb, qm_zigzagged_Cr = zigzagged_quantization_matrix(jpeg_quality)

    Y, Cb, Cr = quantized_Y * qm_zigzagged_Y, quantized_Cb * qm_zigzagged_Cb, quantized_Cr * qm_zigzagged_Cr

    Y = round_or_approx(Y, round)
    Cb = round_or_approx(Cb, round)
    Cr = round_or_approx(Cr, round)

    return Y, Cb, Cr


def dequantize_zigzagged_coeff_tuple(quantized, jpeg_quality: int, batched: bool = False, round: str = 'round'):
    return dequantize_zigzagged_coefficients(quantized[0], quantized[1], quantized[2], jpeg_quality,
                                             batched=batched, round=round)


_quantization_cache = {}


def quantization_matrix(jpeg_quality: int):
    """
    Returns the quantization matrices (Y, Cb, Cr) for a given quality.
    :param jpeg_quality:
    :return:
    """
    return QuantizationJson.quantization_for_quality(jpeg_quality)


def dimensions(ds: str):
    """
    Returns the dimensions matrices (Y, Cb, Cr) for a given dataset.
    :return:
    """
    return DimensionJson.dimension_for_dataset(ds)


def zigzagged_quantization_matrix(jpeg_quality: int):
    """
    Zigzagges quantization matrix.
    :param jpeg_quality:
    :return:
    """
    matrix = quantization_matrix(jpeg_quality)
    expanded_matrix = tf.expand_dims(matrix, 0)
    out = tf_zigzag_and_flatten_coefficients(expanded_matrix, batched=False)
    return out[0][0], out[0][1], out[0][2]


def zigzagged_quantization_matrix_max_division(jpeg_quality: int):
    """
    Let Q be the Zigzagges quantization matrix for the given jpeg_quality.
    Then, returned is Q/max(Q).
    :param jpeg_quality:
    :return:
    """
    zigzagged_matrices = zigzagged_quantization_matrix(jpeg_quality)
    return zigzagged_matrices[0] / tf.reduce_max(zigzagged_matrices[0]), \
           zigzagged_matrices[1] / tf.reduce_max(zigzagged_matrices[1]), \
           zigzagged_matrices[2] / tf.reduce_max(zigzagged_matrices[2])


def zigzagged_quantization_matrix_max_division_negated(jpeg_quality: int):
    """
    Let Q be the Zigzagges quantization matrix for the given jpeg_quality.
    Then, returned is (1 + min(Q) - Q)/(max(1 + min(Q) - Q)).
    :param jpeg_quality:
    :return:
    """
    zigzagged_matrices = zigzagged_quantization_matrix_max_division(jpeg_quality)
    zigzagged_matrices = 1. + tf.reduce_min(zigzagged_matrices[0]) - zigzagged_matrices[0], \
                         1. + tf.reduce_min(zigzagged_matrices[1]) - zigzagged_matrices[1], \
                         1. + tf.reduce_min(zigzagged_matrices[2]) - zigzagged_matrices[2]
    return zigzagged_matrices[0] / tf.reduce_max(zigzagged_matrices[0]), \
           zigzagged_matrices[1] / tf.reduce_max(zigzagged_matrices[1]), \
           zigzagged_matrices[2] / tf.reduce_max(zigzagged_matrices[2])


def integer_differentiable_shin_rounding_approximation(x):
    """
    Rounding approximation that is only used when x is not an integer already.
    Otherwise, x will be returned to offer gradients different from zero.
    """
    rounding_approx = shin_rounding_approximation(x)
    return tf.where(tf.math.equal(x, rounding_approx), x, rounding_approx)


class QuantizationJson:
    """
    Helper class to save quantization matrices for different jpeg qualities.

    We use the quantization matrices used by Tensorflow.
    We save an image for different JPEG qualities, load the quantization matrices using torchjpeg and then
    save them to json-files.
    """
    QUANTIZATION_MATRIX_CACHE = {}
    _loaded_quantization_matrices = False

    @staticmethod
    def __load_json():
        try:
            file = codecs.open(Config.QUANTIZATION_JSON, 'r', encoding='utf-8').read()
            logging.info('Loading quantization json file.')
            return json.loads(file)
        except FileNotFoundError:
            return {}

    @staticmethod
    def __save_to_json_file(filename):
        data = QuantizationJson.QUANTIZATION_MATRIX_CACHE
        processed_data = {}
        for key in list(data.keys()):
            processed_data[str(key)] = data[key].numpy().astype(np.float32).tolist()
        json.dump(processed_data, codecs.open(filename,
                                    'w',
                                    encoding='utf-8'),
                  separators=(',', ':'),
                  sort_keys=True,
                  indent=4)

    @staticmethod
    def __save_quantization_matrices():
        QuantizationJson.__save_to_json_file(Config.QUANTIZATION_JSON)

    @staticmethod
    def _load_quantization_matrices():
        if QuantizationJson._loaded_quantization_matrices:
            return

        data = QuantizationJson.__load_json()
        is_complete = True
        for j in range(101):
            if str(j) not in data:
                is_complete = False
                break

        if is_complete:
            QuantizationJson.QUANTIZATION_MATRIX_CACHE = {int(key): tf.convert_to_tensor(data[key]) for key in data}
            QuantizationJson._loaded_quantization_matrices = True
            return

        ds_train = tfds.load('cifar10',
                             data_dir=Config.TF_DATASETS_DIR,
                             split=['train'],
                             as_supervised=True,
                             shuffle_files=False)[0]

        if adversarial_attacks.utils.general.check_linux():
            import torchjpeg.codec
        else:
            raise ValueError('This function is not available since torchjpeg is only available on linux.')

        image, _ = next(ds_train.take(1).__iter__())
        for jpeg_quality in tqdm(range(101), 'Extracting Quantization Matrices.'):
            image_encode = tf.image.encode_jpeg(image, quality=jpeg_quality)
            dirname = os.path.join(Config.TEMP_DATA_DIR, 'quantization')
            adversarial_attacks.utils.general.makedirs(dirname)
            filename = os.path.join(dirname, '{}.jpg'.format(jpeg_quality))
            tf.io.write_file(filename, image_encode)
            _, quantization_matrix, _, _ = torchjpeg.codec.read_coefficients(filename)
            adversarial_attacks.utils.general.remove_dir(dirname)

            quantization_matrix = quantization_matrix.numpy()
            quantization_matrix = tf.convert_to_tensor(quantization_matrix, dtype=tf.float32)
            QuantizationJson.QUANTIZATION_MATRIX_CACHE[jpeg_quality] = quantization_matrix

        QuantizationJson.__save_quantization_matrices()

    @staticmethod
    @tf.autograph.experimental.do_not_convert
    def quantization_for_quality(jpeg_quality):
        QuantizationJson._load_quantization_matrices()
        return QuantizationJson.QUANTIZATION_MATRIX_CACHE[jpeg_quality]


# cache all jpeg qualities
QuantizationJson.quantization_for_quality(50)


class DimensionJson:
    """
    Helper class to save dimension matrices for different for datasets.
    """
    _loaded_json = False
    DIMENSION_CACHE = {}

    @staticmethod
    def __load_json():
        try:
            file = codecs.open(Config.DIMENSION_JSON, 'r', encoding='utf-8').read()
            logging.info('Loading dimension json file.')
            return json.loads(file)
        except FileNotFoundError:
            return {}

    @staticmethod
    def _load_dimensions_from_json():
        if not DimensionJson._loaded_json:
            data = DimensionJson.__load_json()
            DimensionJson.DIMENSION_CACHE = {key: tf.convert_to_tensor(data[key]) for key in data}

    @staticmethod
    def __save_to_json_file(filename):
        data = DimensionJson.DIMENSION_CACHE
        for key in data:
            data[key] = data[key].numpy().astype(np.float32).tolist()
        json.dump(data, codecs.open(filename,
                                    'w',
                                    encoding='utf-8'),
                  separators=(',', ':'),
                  sort_keys=True,
                  indent=4)

    @staticmethod
    def dimension_for_dataset(dataset):
        DimensionJson._load_dimensions_from_json()
        if dataset in DimensionJson.DIMENSION_CACHE:
            return DimensionJson.DIMENSION_CACHE[dataset]

        ds_train = tfds.load(Config.DATASET_LOADING_NAME[dataset],
                             data_dir=Config.DATASETS_DIR,
                             split=['train'],
                             as_supervised=True,
                             shuffle_files=False)[0]

        if adversarial_attacks.utils.general.check_linux():
            import torchjpeg.codec
        else:
            raise ValueError('This function is not available since torchjpeg is only available on linux.')

        image, _ = next(ds_train.take(1).__iter__())

        target_shape = Config.INPUT_SHAPE[dataset]['rgb']
        image = tf.image.resize(image, [target_shape[0], target_shape[1]])
        image = tf.cast(image, tf.uint8)
        image_encode = tf.image.encode_jpeg(image, quality=100)
        dirname = os.path.join(Config.TEMP_DATA_DIR, 'dimensions')
        adversarial_attacks.utils.general.makedirs(dirname)
        filename = os.path.join(dirname, '{}.jpg'.format(dataset))
        tf.io.write_file(filename, image_encode)
        dimension_matrix, _, _, _ = torchjpeg.codec.read_coefficients(filename)
        adversarial_attacks.utils.general.remove_dir(dirname)

        dimension_matrix = dimension_matrix.numpy()
        dimension_matrix = tf.convert_to_tensor(dimension_matrix, dtype=tf.float32)
        DimensionJson.DIMENSION_CACHE[dataset] = dimension_matrix
        DimensionJson.__save_to_json_file(Config.DIMENSION_JSON)
        return DimensionJson.DIMENSION_CACHE[dataset]
