import os
from datetime import datetime
import math

import numpy as np
import tensorflow as tf

import adversarial_attacks.utils.general as utils
import adversarial_attacks.utils.logging as logging
from adversarial_attacks.config.config import Config
from adversarial_attacks.config.tf_distribution import tf_distribution_strategy


class AdversarialLoss(tf.keras.losses.CategoricalCrossentropy):
    def __init__(self, from_logits=True, label_smoothing=0.0,
                 axis=-1,
                 reduction=tf.keras.losses.Reduction.NONE,
                 name='categorical_crossentropy'):
        super().__init__(from_logits=from_logits, axis=axis, label_smoothing=label_smoothing,
                         reduction=tf.keras.losses.Reduction.NONE, name='weighted_adversarial_crossentropy')

    def call(self, y_true, y_pred):
        """
           Assumes that the first half of the batch is original and the second is adversarial
        """
        lambda_adv = 0.3

        per_image_loss = super().call(y_true, y_pred)
        m = tf.shape(per_image_loss)[0]
        k = tf.cast(m / 2, tf.int32)
        # is_original_mask = tf.concat([tf.ones([bs/2]), tf.zeros([bs/2])], axis=0)
        multiplier = tf.concat([tf.ones([m - k]), tf.zeros([k]) + lambda_adv], axis=0)
        m, k = tf.cast(m, tf.float32), tf.cast(k, tf.float32)
        adv_loss = (1 / (m - k + lambda_adv * k)) * tf.math.reduce_sum(multiplier * per_image_loss)
        return adv_loss


def load_model_from_path(model_path, cache=True):
    """
    Returns None if the model was not found.
    """

    if model_path in Model.loaded_models and cache:
        logging.debug('Loaded Model {} from memory.'.format(model_path))
        return Model.loaded_models[model_path]
    try:
        logging.info('Trying to load Model from path {}.'.format(model_path))
        model = tf.keras.models.load_model(model_path, custom_objects={'AdversarialLoss': AdversarialLoss})
        logging.info('Loaded Model {} from file system.'.format(model_path))
    except OSError as e:
        logging.info('Model {} not found: {}.'.format(model_path, str(e)))
        return None

    if cache:
        Model.loaded_models[model_path] = model
        logging.debug('Cached model {}.'.format(model_path))
    return model


def load_model(dataset, model_name, cache=True):
    return load_model_from_path(os.path.join(Config.DATASET_MODELS[dataset], model_name), cache=cache)


def remove_model_from_memory_cache(dataset, model_name):
    if (dataset, model_name) in Model.loaded_models:
        del Model.loaded_models[dataset, model_name]
        logging.info('Removed model {} for dataset {} from memory cache'.format(model_name, dataset))


def get_input_shapes_for_ds(dataset_name):
    rgb_input_shape = Config.INPUT_SHAPE[dataset_name]['rgb']
    ycbcr_channel_input_shapes = (rgb_input_shape[0], rgb_input_shape[1])
    not_subsampled_input_shape = Config.INPUT_SHAPE[dataset_name]['Y']
    subsampled_input_shape = Config.INPUT_SHAPE[dataset_name]['C']
    return {'rgb': rgb_input_shape, 'ycbcr_channels': ycbcr_channel_input_shapes,
            'not_subsampled_jpeg': not_subsampled_input_shape, 'subsampled_jpeg': subsampled_input_shape}


class _AdversarialTrainingModel(tf.keras.Model):
    """
    Helper Model for adversarial training
    """

    def call(self, inputs, training=None, mask=None):
        return super(_AdversarialTrainingModel, self).call(inputs, training=training, mask=mask)

    def __init__(self, model, adversarial_train_ds, adversarial_test_ds):
        """
        :param model: tf.keras.Model
        :param adversarial_train_ds:
                adversarial_attacks.datasets.adversarial_training_datasets.AdversarialTrainingDataset
        :param adversarial_test_ds:
                adversarial_attacks.datasets.adversarial_training_datasets.AdversarialTrainingDataset
        """
        super().__init__(inputs=model.inputs, outputs=model.outputs)
        self._adversarial_train_ds = adversarial_train_ds
        self._adversarial_test_ds = adversarial_test_ds

    def train_step(self, data):
        """
        train step that includes creating the adversarial batch.
        This is defined in the adversarial datasets.

        Otherwise, the train step does the same as in the original tf implementation.

        :param data:
        :return:
        """
        x, y = self._adversarial_train_ds.get_adversarial_batch(data)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred)
        self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, y, y_pred, tf.ones(tf.shape(y)[0]))

    def test_step(self, data):
        """
        test step that includes creating the adversarial batch.
        This is defined in the adversarial datasets.

        Otherwise, the train step does the same as in the original tf implementation.

        :param data:
        :return:
        """
        x, y = self._adversarial_test_ds.get_adversarial_batch(data)
        y_pred = self(x, training=False)
        # Updates stateful loss metrics.
        self.compute_loss(x, y, y_pred)
        return self.compute_metrics(x, y, y_pred, tf.ones(tf.shape(y)[0]))


class Model:
    """
    Model class. Wraps a tf.keras.Model
    """
    # dictionary that save the models loaded in memory
    loaded_models = {}

    def __init__(self, dataset_name, save_model_name='model', load_model_name=None, save_model=False,
                 jpeg_compression_quality=None):
        """
        :param dataset_name: name of the dataset, e.g. cifar10
        :param save_model_name: name to save the model
        :param load_model_name: name to load the model
        :param save_model: whether to save the model or not
        """
        super().__init__()

        self.ds_name = dataset_name
        self.jpeg_compression_quality = jpeg_compression_quality

        # set dataset related variables
        _input_shapes = get_input_shapes_for_ds(self.ds_name)
        self.rgb_input_shape = _input_shapes['rgb']
        self.ycbcr_channel_input_shapes = _input_shapes['ycbcr_channels']
        self.not_subsampled_input_shape = _input_shapes['not_subsampled_jpeg']
        self.subsampled_input_shape = _input_shapes['subsampled_jpeg']

        self.num_classes = Config.DATASET_NUM_CLASSES[self.ds_name]

        self.model_name = save_model_name
        self.load_model_name = load_model_name
        self.trainable = True
        self.model_loaded = False
        self._distribute_strategy = tf_distribution_strategy

        load_saved_model = self.load_model_name is not None
        if load_saved_model:
            cache = self.model_name is None or (self.load_model_name == self.model_name)
            # try to load the model
            with self._distribute_strategy.scope():
                self.model = load_model(self.ds_name, self.load_model_name, cache=cache)
            self.model_loaded = self.model is not None
        if not load_saved_model or self.model is None:
            # if the model was not loaded, build it using self.build_model()
            with self._distribute_strategy.scope():
                self.model = self.build_model()
            logging.info('Build model {} of type {}.'.format(self.model_name, self.__class__.__name__))

        if save_model and not self.model_loaded and self.model_name is not None:
            self.save_model()

    @tf.function
    def __call__(self, inp, logits_or_softmax='softmax', training=False):
        """
        logits_or_softmax in ['softmax', 'logits', 'both']
        """
        logits = self.model(inp, training=training)

        if logits_or_softmax == 'logits':
            return logits

        if logits_or_softmax == 'softmax':
            return tf.nn.softmax(logits)

        if logits_or_softmax == 'both':
            return logits, tf.nn.softmax(logits)

        raise ValueError('logits_or_softmax parameter must be in [softmax, logits, both]. Received {}.'.
                         format(logits_or_softmax))

    def build_model(self):
        """
        Function to build the tf.keras.Model
        :return:
        """
        return None

    def metrics_results(self, metrics):
        """
        Returns the results of given metrics as a dictionary.
        """
        return {metric_name: metrics[metric_name].result().numpy().astype(np.float64) for metric_name in metrics}

    def _get_loss(self, adv_loss=False):
        if not adv_loss:
            return tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        return AdversarialLoss(from_logits=True)

    def _eval_batched_tf_dataset(self, ds, n_batches, verbose=1, adversarial_loss=False):
        """
        Evaluation function for batched tf.data.Datset
        :return:
        """
        ds = ds.take(n_batches)

        with self._distribute_strategy.scope():
            self.model = self.model
            self.model.compile(loss=self._get_loss(adv_loss=adversarial_loss),
                               metrics=self.get_metrics())
        loss, acc, categorical_crossentropy, top5acc = self.model.evaluate(ds, verbose=verbose, steps=n_batches)
        return {'Loss': loss, 'Acc': acc, 'CategoricalCrossentropy': categorical_crossentropy, 'Top5Acc': top5acc}

    def eval_adversarial_ds(self, ds, verbose=1, use_adversarial_loss=True):
        """
        Evaluation for a adversarial_attacks.datasets.adversarial_training_datasets.AdversarialTrainingDataset

        Returns Accuracy, Loss, Top5-Accuracy
        :param ds:
        :param verbose:
        :param use_adversarial_loss:
        :return:
        """
        return self._eval_batched_tf_dataset(ds.ds, n_batches=ds.get_n_batches(), verbose=verbose,
                                             adversarial_loss=use_adversarial_loss)

    def eval_original_ds(self, ds, batch_size, verbose=1):
        """
        Evaluation for a adversarial_attacks.datasets.original.RGBDataset

        Returns Accuracy, Loss, Top5-Accuracy
        :param ds:
        :param batch_size:
        :param verbose:
        :return:
        """
        n_batches = ds.get_n_batches(batch_size)
        return self._eval_batched_tf_dataset(ds.ds.batch(batch_size), n_batches=n_batches, verbose=verbose,
                                             adversarial_loss=False)

    def get_metrics(self):
        """
        Returns evaluation metrics: acc, crossentropy loss, top5-acc
        :return:
        """

        return tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.CategoricalCrossentropy(
            from_logits=True), tf.keras.metrics.TopKCategoricalAccuracy(k=5)

    def save_model(self):
        """
        Saves the model on disk.
        """
        logging.info('Saving model {} of type {} to file system...'.format(self.model_name, self.__class__.__name__))
        self.model.save(os.path.join(Config.DATASET_MODELS[self.ds_name], self.model_name))
        logging.info('Saved model to file system')
        remove_model_from_memory_cache(self.ds_name, self.model_name)

    @staticmethod
    def _lr_schedule(epoch, num_epochs, initial_lr):
        """Learning Rate Schedule
        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.
        # Arguments
            epoch (int): The number of epochs
        # Returns
            lr (float32): learning rate
        """
        lr = initial_lr
        if epoch > math.floor(0.9 * num_epochs):
            lr *= 0.5e-3
        elif epoch > math.floor(0.8 * num_epochs):
            lr *= 1e-3
        elif epoch > math.floor(0.6 * num_epochs):
            lr *= 1e-2
        elif epoch > math.floor(0.4 * num_epochs):
            lr *= 1e-1
        return lr

    def _prepare_training(self, epochs, initial_lr, early_stopping, lr_scheduler):
        """
        Prepares the model for training.
        Initializes callbacks etc.

        :param epochs:
        :param initial_lr:
        :param early_stopping:
        :param lr_scheduler:
        :return:
        """

        if not self.trainable:
            return

        # create dir to save the model
        utils.makedirs(Config.DATASET_MODELS[self.ds_name])

        # model checkpoints. will be saved for optimal val_loss
        if self.model_name is not None:
            filepath = os.path.join(Config.DATASET_MODELS[self.ds_name], self.model_name)
            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, verbose=1, monitor='val_loss',
                                                            save_best_only=True)
        else:
            checkpoint = None

        # lr scheduler
        if lr_scheduler == 'auto':
            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: self._lr_schedule(epoch,
                                                                                                        num_epochs=epochs,
                                                                                                        initial_lr=initial_lr))

        # lr reducer on plateau
        lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                                                          cooldown=0,
                                                          patience=5,
                                                          min_lr=0.5e-6)

        # tensorboard
        log_dir = os.path.join(Config.TENSORBOARD_LOG_DIR,
                               "{}-{}".format(self.model_name, datetime.now().strftime("%Y%m%d-%H%M%S")),
                               'logs/')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=False)
        callbacks = [lr_reducer, tensorboard_callback]

        if early_stopping is not None:
            callbacks.append(early_stopping)

        if lr_scheduler is not None:
            callbacks.append(lr_scheduler)

        if self.model_name is not None:
            callbacks.append(checkpoint)

        return callbacks

    def train_original_ds(self, ds_train, ds_test, batch_size, epochs,
                          optimizer_lambda=lambda: tf.keras.optimizers.legacy.SGD(0.1, momentum=0.9, decay=0.0001),
                          early_stopping=None, lr_schedule='auto'):
        """
        Training on the original dataset.

        :param ds_train: adversarial_attacks.datasets.original.RGBDataset
        :param ds_test: adversarial_attacks.datasets.original.RGBDataset
        :param batch_size:
        :param epochs:
        :param optimizer_lambda:
        :param early_stopping:
        :param lr_schedule:
        :return:
        """

        train_batches = math.ceil(ds_train.number_of_images / batch_size)
        test_batches = math.ceil(ds_test.number_of_images / batch_size)

        def data_gen(ds):
            for images, labels in ds.ds.batch(batch_size,
                                              num_parallel_calls=tf.data.AUTOTUNE):
                yield images, labels

        def data_gen_train():
            return data_gen(ds_train)

        def data_gen_test():
            return data_gen(ds_test)

        train_data = tf.data.Dataset.from_generator(data_gen_train,
                                                    output_signature=ds_train.batched_output_signature) \
            .prefetch(tf.data.AUTOTUNE).repeat()

        test_data = tf.data.Dataset.from_generator(data_gen_test,
                                                   output_signature=ds_train.batched_output_signature) \
            .prefetch(tf.data.AUTOTUNE).repeat()

        self._fit(train_data, test_data, epochs, train_batches, test_batches, batch_size,
                  optimizer_lambda, early_stopping, lr_schedule, adversarial_loss=False)

    def train_adversarial_ds(self, cascade_ds_train, cascade_ds_test, epochs,
                             optimizer_lambda=lambda: tf.keras.optimizers.legacy.SGD(0.1, momentum=0.9, decay=0.0001),
                             early_stopping=None, lr_schedule='auto', use_adversarial_loss=True):
        """
        Adversarial Training on a adversarial_attacks.datasets.adversarial_training_datasets.AdversarialTrainingDataset

        """
        # defines helper model for adversarial training.
        # The _AdversarialTrainingModel includes the creation of adversarial batches in the train (and val) steps
        with self._distribute_strategy.scope():
            self.model = _AdversarialTrainingModel(self.model, cascade_ds_train, cascade_ds_test)

        train_batches = cascade_ds_train.get_n_batches()
        test_batches = cascade_ds_test.get_n_batches()

        if cascade_ds_train.get_batch_size() != cascade_ds_test.get_batch_size():
            raise ValueError(
                'The test and train datasets batch sizes must be equal for the loss to work with the mirrored strategy.')

        self._fit(cascade_ds_train.original_ds, cascade_ds_test.original_ds, epochs, train_batches, test_batches,
                  cascade_ds_train.get_batch_size(),
                  optimizer_lambda, early_stopping, lr_schedule, adversarial_loss=use_adversarial_loss)

    def _fit(self, train_data, test_data, epochs, number_of_train_batches, number_of_test_batches, batch_size,
             optimizer_lambda, early_stopping, lr_schedule, adversarial_loss):
        """
        Internal Training function.

        :param train_data:
        :param test_data:
        :param epochs:
        :param number_of_train_batches:
        :param number_of_test_batches:
        :param batch_size:
        :param optimizer_lambda:
        :param early_stopping:
        :param lr_schedule:
        :param adversarial_loss:
        :return:
        """

        train_data = train_data.unbatch().batch(batch_size * self._distribute_strategy.num_replicas_in_sync)
        test_data = test_data.unbatch().batch(batch_size * self._distribute_strategy.num_replicas_in_sync)

        # Disable File AutoShard.
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        train_data = train_data.with_options(options)
        test_data = test_data.with_options(options)

        number_of_train_batches = math.ceil(number_of_train_batches / self._distribute_strategy.num_replicas_in_sync)
        number_of_test_batches = math.ceil(number_of_test_batches / self._distribute_strategy.num_replicas_in_sync)

        callbacks = self._prepare_training(epochs, optimizer_lambda().lr.numpy(), early_stopping, lr_schedule)

        with self._distribute_strategy.scope():
            self.model = self.model
            self.model.compile(loss=self._get_loss(adv_loss=adversarial_loss),
                               optimizer=optimizer_lambda(), metrics=self.get_metrics())

        self.model.fit(train_data,
                       validation_data=test_data,
                       epochs=epochs,
                       steps_per_epoch=number_of_train_batches,
                       validation_steps=number_of_test_batches,
                       callbacks=callbacks,
                       verbose=1)

        self.model = tf.keras.Model(inputs=self.model.inputs, outputs=self.model.outputs)
