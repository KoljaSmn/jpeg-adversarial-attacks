import tensorflow as tf

import adversarial_attacks.utils.jpeg

lambda_Y_lower_frequencies_higher_weights = \
adversarial_attacks.utils.jpeg.zigzagged_quantization_matrix_max_division_negated(50)[0]
lambda_Y_lower_frequencies_lower_weights = \
adversarial_attacks.utils.jpeg.zigzagged_quantization_matrix_max_division(50)[0]

lambda_Cb_lower_frequencies_higher_weights = \
adversarial_attacks.utils.jpeg.zigzagged_quantization_matrix_max_division_negated(50)[1]
lambda_Cb_lower_frequencies_lower_weights = \
adversarial_attacks.utils.jpeg.zigzagged_quantization_matrix_max_division(50)[1]

lambda_Cr_lower_frequencies_higher_weights = \
adversarial_attacks.utils.jpeg.zigzagged_quantization_matrix_max_division_negated(50)[2]
lambda_Cr_lower_frequencies_lower_weights = \
adversarial_attacks.utils.jpeg.zigzagged_quantization_matrix_max_division(50)[2]

lambda_lower_frequencies_higher_weights_linear = tf.convert_to_tensor([1. - i / 64 for i in range(64)])
lambda_lower_frequencies_lower_weights_linear = tf.convert_to_tensor([(i + 1) / 64 for i in range(64)])

lambda_lower_frequencies_higher_weights_linear = (
lambda_lower_frequencies_higher_weights_linear, lambda_lower_frequencies_higher_weights_linear,
lambda_lower_frequencies_higher_weights_linear)
lambda_lower_frequencies_lower_weights_linear = (
lambda_lower_frequencies_lower_weights_linear, lambda_lower_frequencies_lower_weights_linear,
lambda_lower_frequencies_lower_weights_linear)

unmasked = tf.convert_to_tensor([1. for i in range(64)])
lambdas_unmasked = (unmasked, unmasked, unmasked)

lambda_lower_frequencies_higher_weights = (
lambda_Y_lower_frequencies_higher_weights, lambda_Cb_lower_frequencies_higher_weights,
lambda_Cr_lower_frequencies_higher_weights)
lambda_lower_frequencies_lower_weights = (
lambda_Y_lower_frequencies_lower_weights, lambda_Cb_lower_frequencies_lower_weights,
lambda_Cr_lower_frequencies_lower_weights)

ascending_part_n = 15
ascending_part = (tf.math.log([float(i) for i in range(1, ascending_part_n + 1)]) / tf.math.log(
    float(ascending_part_n + 1))).numpy().tolist()

medium = [0.09193521, 0.381128, 0.30488983, 0.5218679, 0.524519,
          0.57720166, 0.7212526, 0.6533365, 0.62271976, 0.6088673,
          0.7362248, 0.7318446, 0.709922, 0.7330684, 0.78126806,
          0.8735231, 0.8152993, 0.7728358, 0.7505039, 0.73824614,
          0.7451628, 0.8902607, 0.8233613, 0.7712334, 0.766868,
          0.7700578, 0.7853828, 0.83509225, 1., 0.85733694,
          0.77401024, 0.7480685, 0.70956355, 0.7199859, 0.76203024,
          0.8522837, 0.88684005, 0.7572527, 0.6850435, 0.6561498,
          0.6532453, 0.666156, 0.7150206, 0.7206875, 0.64801234,
          0.59303594, 0.57625175, 0.56492406, 0.60066897, 0.60483736,
          0.53106666, 0.4833844, 0.4564871, 0.45350286, 0.44723934,
          0.39181444, 0.35314935, 0.3384714, 0.32057893, 0.25833154,
          0.21121486, 0.18175012, 0.12854235, 0.08305813]

overall_medium_frequencies = tf.convert_to_tensor(medium)
lambdas_medium_frequencies = (overall_medium_frequencies, overall_medium_frequencies, overall_medium_frequencies)


def _mask_dc(_lambdas):
    def mask_dc(_lambda):
        a = tf.convert_to_tensor([0.] + [1. for i in range(63)])
        return _lambda * a

    return mask_dc(_lambdas[0]), mask_dc(_lambdas[1]), mask_dc(_lambdas[2])


lambdas = {'qm_descent': lambda_lower_frequencies_higher_weights,
           'qm_ascent': lambda_lower_frequencies_lower_weights,
           'linear_descent': lambda_lower_frequencies_higher_weights_linear,
           'linear_ascent': lambda_lower_frequencies_lower_weights_linear,
           'unmasked': lambdas_unmasked, 'medium': lambdas_medium_frequencies}

_lambda_keys = list(lambdas.keys())

for lambda_key in _lambda_keys:
    lambdas[f'{lambda_key}_mask_dc'] = _mask_dc(lambdas[lambda_key])
