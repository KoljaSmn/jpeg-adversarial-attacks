from adversarial_attacks.config.config import Config
from adversarial_attacks.models.models import load_model_from_path, load_model
import tensorflow as tf

from adversarial_attacks.models.models import get_input_shapes_for_ds
from adversarial_attacks.utils.general import makedirs
import os
from adversarial_attacks.utils.logging import info
from adversarial_attacks.datasets.bapps import AFC2
from adversarial_attacks.config.tf_distribution import tf_distribution_strategy


class _LinearLayer(tf.keras.layers.Layer):
    """
    Adapted from https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/pretrained_networks.py
    """

    def __init__(self, use_dropout=True):
        super().__init__()

        self._dropout = tf.keras.layers.Dropout(0.5) if use_dropout else lambda x: x
        self._conv = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer='ones',
                                           kernel_constraint=tf.keras.constraints.NonNeg())

    def call(self, x, training=None):
        if training:
            x = self._dropout(x)
        return self._conv(x)


def normalize_tensor(in_feat, eps=1e-10):
    # See https://github.com/richzhang/PerceptualSimilarity/blob/31bc1271ae6f13b7e281b9959ac24a5e8f2ed522/lpips/__init__.py#L13
    norm_factor = tf.math.sqrt(
        tf.math.reduce_sum(in_feat, axis=3, keepdims=True))  # channel axis is 3 (, 1 in the original pytorch version)
    return in_feat / (norm_factor + eps)


def spatial_average(in_tens, keepdim=True):
    # See https://github.com/richzhang/PerceptualSimilarity/blob/31bc1271ae6f13b7e281b9959ac24a5e8f2ed522/lpips/lpips.py#L14
    return tf.math.reduce_mean(in_tens, axis=(1, 2),
                               keepdims=keepdim)  # w, h are located in axes 1, 2. (2, 3 in the original pytorch implementation.)


class LossNetwork:
    """
    Tensorflow implementation of LPIPS model from
    Zhang et al., The Unreasonable Effectiveness of Deep Features as a Perceptual Metric, CVPR 2018

    In our experiments, we use the vgg16 net as it is available pretrained in Tensorflow.
    We use the same layers as in Zhang et al.: The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
    See https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/pretrained_networks.py
    the convolutional layers already include the relu activation.
    See Tensorflow implementation at https://github.com/keras-team/keras/blob/v2.11.0/keras/applications/vgg16.py#L48-L252
    """

    LOSS_LAYER_NAMES = {'vgg16': ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']}
    _nets_including_dataset = {}

    def __init__(self, dataset_name, net='vgg16',
                 imagenet_load_model_name=None, fix_loss_model_weights=True, lpips=True, loss_layer_names=None,
                 input_w=None, input_h=None, use_cpu=False, name=None):
        """

        :param dataset_name: the name of the dataset. 'cifar10', 'imagenet' or 'bapps' (for training mainly)
        :param net: called F in the original paper. only vgg16 available currently. Set to None, if using a custom model
        :param imagenet_load_model_name: if net is None, use the imagenet_load_model_name to use a custom model.
               When using a custom model, make sure to set the loss_layer_names accordingly
        :param fix_loss_model_weights: whether F's weights should be fixed
        :param lpips: whether to use trained the linear layers (LPIPS)
        :param loss_layer_names: when using a custom model, specify the loss layer names
        :param input_w: instead of specifying a dataset_name, you can also specify the input images' width and height manually
        :param input_h:
        :param use_cpu:
        :param name: to give the model a custom name for saving.
               When not specifying it, an already saved model will be loaded
        """

        if net != 'vgg16' and imagenet_load_model_name is None:
            raise NotImplementedError("{} not implemented as LossNetwork")

        if net is not None and imagenet_load_model_name is not None:
            raise ValueError('One of net and imagenet_load_model_name must be None.')

        self._ds_name = dataset_name
        self._use_cpu = use_cpu
        
        self._name = name

        self._net = net
        self._imagenet_load_model_name = imagenet_load_model_name
        self._fix_loss_model_weights = fix_loss_model_weights
        self._lpips = lpips
        if loss_layer_names is None:
            self._loss_layer_names = self.LOSS_LAYER_NAMES[net]
        else:
            self._loss_layer_names = loss_layer_names

        # generate key for saving the model to the file system
        self._file_key = LossNetwork._get_key(self._net, self._imagenet_load_model_name, self._fix_loss_model_weights,
                                              self._lpips, self._loss_layer_names, self._name)
        self._file_key = self._file_key.replace(' ', '')
        self._file_key = self._file_key.replace("'", "")
        self._file_key = self._file_key.replace("[", "_")
        self._file_key = self._file_key.replace("]", "_")

        # the same loss network (the network the activations are used from) is used for every dataset.
        # Thus, ds_name is not part of the key

        self._filepath = os.path.join(Config.LPIPS_LOSS_MODEL_PATH, self._file_key)
        makedirs(self._filepath)

        self._input_w, self._input_h = input_w, input_h

        if not self._use_cpu:
            with tf_distribution_strategy.scope():
                self._loss_network, self.model = self._build_model()
        else:
            with tf.device('/device:cpu:0'):
                self._loss_network, self.model = self._build_model()
        if self._ds_name is not None:
            LossNetwork._nets_including_dataset[LossNetwork._cache_key(self._ds_name, self._net,
                                                                       self._imagenet_load_model_name,
                                                                       self._fix_loss_model_weights,
                                                                       self._lpips,
                                                                       self._loss_layer_names,
                                                                       self._use_cpu,
                                                                       self._name)] = self

    @staticmethod
    def _get_key(net, imagenet_load_model_name, fix_loss_model_weights, lpips, loss_layer_names, name):
        st = '{}{}{}{}{}'.format(net, imagenet_load_model_name, fix_loss_model_weights, lpips, loss_layer_names)
        if name is not None:
            st += name
        return st

    @staticmethod
    def _cache_key(ds, net, imagenet_load_model_name, fix_loss_model_weights, lpips, loss_layer_names, use_cpu, name):
        return ds, use_cpu, LossNetwork._get_key(net, imagenet_load_model_name, fix_loss_model_weights, lpips,
                                                 loss_layer_names, name)

    @staticmethod
    def get_loss_network(dataset_name, net='vgg16',
                         imagenet_load_model_name=None, fix_loss_model_weights=True, lpips=True, loss_layer_names=None,
                         use_cpu=False, name=None):
        key_including_ds = LossNetwork._cache_key(dataset_name, net, imagenet_load_model_name,
                                                  fix_loss_model_weights, lpips, loss_layer_names, use_cpu, name)
        if key_including_ds in LossNetwork._nets_including_dataset:
            return LossNetwork._nets_including_dataset[key_including_ds]

        ln = LossNetwork(dataset_name, net, imagenet_load_model_name, fix_loss_model_weights, lpips, loss_layer_names,
                         use_cpu=use_cpu)
        LossNetwork._nets_including_dataset[key_including_ds] = ln
        return ln

    def _imagenet_model(self):
        """
        Loads/Creates the imagenet model F.
        :return:
        """
        if self._imagenet_load_model_name is None:
            if self._net == 'vgg16':
                preprocessed_input = tf.keras.applications.vgg16.preprocess_input(
                    tf.keras.Input(Config.INPUT_SHAPE['imagenet']['rgb']))
                return tf.keras.applications.vgg16.VGG16(input_shape=Config.INPUT_SHAPE['imagenet']['rgb'],
                                                         input_tensor=preprocessed_input,
                                                         include_top=True, weights='imagenet')
            else:
                raise ValueError("Net {} unknown.".format(self._net))
        else:
            return load_model('imagenet', self._imagenet_load_model_name)

    def _build_loss_network(self):
        """
        Builds the loss network.
        :return:
        """
        # first, model F is loaded
        _imagenet_classification_model = self._imagenet_model()
        _loss_layer_outputs = [normalize_tensor(_imagenet_classification_model.get_layer(layer_name).output) for
                               layer_name in
                               self._loss_layer_names]
        # a helper model than returns the activations of the selected layers
        _loss_model = tf.keras.Model(inputs=_imagenet_classification_model.input, outputs=_loss_layer_outputs)

        if self._fix_loss_model_weights:
            for layer in _loss_model.layers:
                layer.trainable = False

        # computes differences between the _loss_model's outputs
        img0_in, img1_in = tf.keras.Input(_loss_model.input.shape[1:]), tf.keras.Input(_loss_model.input.shape[1:])
        img0_loss_model_outputs = _loss_model(img0_in)
        img1_loss_model_outputs = _loss_model(img1_in)
        output_diff = [
            tf.math.pow(normalize_tensor(img0_loss_model_outputs[i]) - normalize_tensor(img1_loss_model_outputs[i]), 2)
            for i in range(len(_loss_layer_outputs))]  # compute l2 norm of differences

        # if lpips is used, add linear layers that weight each layer and channel
        if self._lpips:
            lin_layers = [_LinearLayer(use_dropout=True) for i in range(len(output_diff))]
            lin_layer_outputs = [lin_layers[i](output_diff[i]) for i in
                                 range(len(output_diff))]  # pass differences through linear layers
        else:
            lin_layer_outputs = [tf.math.reduce_sum(output_diff[i], axis=-1, keepdims=True) for i in
                                 range(len(output_diff))]

        flatten_layer = tf.keras.layers.Flatten()

        spatial_average_outputs = [flatten_layer(spatial_average(layer_output, keepdim=True)) for layer_output in
                                   lin_layer_outputs]
        out = tf.keras.layers.Average()(spatial_average_outputs)
        return tf.keras.Model(inputs=[img0_in, img1_in], outputs=[out])

    def _build_model(self):
        """
        Builds the full network.

        The model returned from _build_loss_network is expecting Imagenet input (size 224*224),
        here a resizing layer is added on top such that the model is usable for the chosen dataset.

        :return:
        """
        if self._ds_name is not None:
            input_shapes = get_input_shapes_for_ds(self._ds_name)['rgb']
            w, h, c = input_shapes
        else:
            w, h, c = self._input_w, self._input_w, 3

        inp0, inp1 = tf.keras.Input(shape=(w, h, c)), tf.keras.Input(shape=(w, h, c))
        imagenet_h, imagenet_w = Config.INPUT_SHAPE['imagenet']['rgb'][:2]
        rescaled_inp_0, rescaled_inp_1 = tf.keras.layers.Resizing(imagenet_h, imagenet_w)(
            inp0), tf.keras.layers.Resizing(imagenet_h, imagenet_w)(inp1)

        with tf_distribution_strategy.scope():
            loss_network = self._load()

            if loss_network is None:
                loss_network = self._build_loss_network()

                info('Built loss network {}.'.format(self._file_key))

            rescaling_and_loss_network = tf.keras.Model(inputs=[inp0, inp1],
                                                        outputs=loss_network((rescaled_inp_0, rescaled_inp_1)))

        return loss_network, rescaling_and_loss_network

    def __call__(self, img0, img1, training=False):
        """
        Returns the LPIPS distances for 2 batches (original, adversarial) of images.
        :param img0:
        :param img1:
        :param training:
        :return:
        """
        if self._use_cpu:
            with tf.device('/device:cpu:0'):
                img0, img1 = img0, img1
                out = self.model((img0, img1), training=training)
                return out

        return self.model((img0, img1), training=training)

    def save(self):
        tf.keras.models.save_model(self._loss_network, self._filepath)

    def _load(self):
        return load_model_from_path(self._filepath)

    def train_on_2afc(self, chn_mid=32, use_sigmoid=True, save=True, bs=50, epochs=5, initial_lr=1e-4, choices=None,
                      unique_trainer_key=None):
        """
        Trains the model on the 2AFC dataset from
        Zhang et al., The Unreasonable Effectiveness of Deep Features as a Perceptual Metric, CVPR 2018.

        See adversarial_attacks.datasets.bapps for details on the dataset

        Builds the trainer model G and trains the model.

        :param chn_mid: for the trainer model G, as in the original work.
        :param use_sigmoid:
        :param save: whether the lpips and trainer model should be saved after training
        :param bs: batch size
        :param epochs: number of epochs
        :param initial_lr: initial learning rate
        :param choices: allows to select choices for the 2AFC dataset.
        :param unique_trainer_key: For being able to use a new trainer (G) when training a new model.
        :return:
        """
        training_model = _TrainingModel(self.model, chn_mid=chn_mid, use_sigmoid=use_sigmoid,
                                        unique_trainer_key=unique_trainer_key)

        if choices is None:
            choices = ['traditional', 'cnn']

        ds_train = AFC2('train', choices)
        ds_test = AFC2('val', choices)

        n_train_steps, n_test_steps = ds_train.n_batches(bs), ds_test.n_batches(bs)

        def map_in_label(ref, p0, p1, judge):
            return (ref, p0, p1), judge

        ds_train = ds_train.ds.map(map_in_label).batch(bs).take(n_train_steps).prefetch(tf.data.AUTOTUNE).repeat()
        ds_test = ds_test.ds.map(map_in_label).batch(bs).take(n_test_steps).prefetch(tf.data.AUTOTUNE).repeat()

        nepoch_decay = epochs

        # implements linear lr decay
        def lr(epoch, lr):
            if epoch > 0:
                lrd = initial_lr / nepoch_decay
                return lr - lrd
            return initial_lr

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr)

        # training distribution
        with tf_distribution_strategy.scope():
            training_model.model = training_model.model
            training_model.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
                                         loss=training_model.loss)

        # training
        training_model.model.fit(ds_train, validation_data=ds_test, steps_per_epoch=n_train_steps,
                                 validation_steps=n_test_steps, epochs=epochs, callbacks=[lr_scheduler])

        if save:
            training_model.save()
            self.save()


_imagenet_w, _imagenet_h = Config.INPUT_SHAPE['imagenet']['rgb'][:2]


def _preprocess_and_resize(inp):
    preprocess_input = tf.keras.applications.vgg16.preprocess_input(inp)
    resized_input = tf.keras.layers.Resizing(_imagenet_w, _imagenet_h)(preprocess_input)
    return resized_input


class _TrainingModel:
    """
    takes 3 images (ref, p0, p1), computes 2 distances (ref - p0, ref - p1) using the distance model, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True)
    https://github.com/richzhang/PerceptualSimilarity/blob/31bc1271ae6f13b7e281b9959ac24a5e8f2ed522/lpips/lpips.py#L169
    """
    _model_dir = Config.LPIPS_TRAINING_MODEL_PATH
    makedirs(_model_dir)

    def __init__(self, distance_model, chn_mid=32, use_sigmoid=True, eps=0.1, unique_trainer_key=None):
        self.distance_model = distance_model

        img_input_shape = distance_model.input[0].shape[1:]
        ref, p0, p1 = tf.keras.Input(img_input_shape), tf.keras.Input(img_input_shape), tf.keras.Input(img_input_shape)

        d0, d1 = self.distance_model((ref, p0)), self.distance_model((ref, p1))

        key = 'chnmid{}use_sigmoid{}eps{}'.format(chn_mid, use_sigmoid, eps)

        if unique_trainer_key is not None:
            key += unique_trainer_key

        self._model_path = os.path.join(self._model_dir, key)

        with tf_distribution_strategy.scope():
            self._G = load_model_from_path(self._model_path)

            if self._G is None:
                x = tf.concat([d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)], axis=1)
                x = tf.keras.layers.Dense(chn_mid, use_bias=True)(x)
                x = tf.keras.layers.LeakyReLU(0.2)(x)
                x = tf.keras.layers.Dense(chn_mid / 2, use_bias=True)(x)
                x = tf.keras.layers.LeakyReLU(0.2)(x)
                x = tf.keras.layers.Dense(1, use_bias=True)(x)
                x = tf.keras.layers.Activation('sigmoid')(x)
                self._G = tf.keras.Model(inputs=[d0, d1], outputs=x)

            self.model = tf.keras.Model(inputs=[ref, p0, p1], outputs=self._G((d0, d1)))
        self._loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    def loss(self, judge, logit):
        # per = (judge + 1.) / 2.
        # per was used instead of judge in the original implementation
        return self._loss(judge, logit)

    def save(self):
        tf.keras.models.save_model(self._G, self._model_path)
