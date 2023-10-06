import tensorflow as tf

import adversarial_attacks.models.models
import adversarial_attacks.utils.transformation


def jpeg_compression_for_rgb_model(ds_name, input, jpeg_compression_quality):
    """
    Compression for RGB input. Not differentiable.
    :param ds_name:
    :param input:
    :param jpeg_compression_quality:
    :return:
    """
    compressed = adversarial_attacks.utils.transformation.get_rgb_to_jpeg_model(ds_name,
                                                                                jpeg_compression_quality,
                                                                                round='round')(
        input)
    compressed_rgb = adversarial_attacks.utils.transformation.get_jpeg_to_rgb_model(ds_name,
                                                                                    jpeg_compression_quality,
                                                                                    round=False)(
        compressed)
    return compressed_rgb


class ImagenetResNet(adversarial_attacks.models.models.Model):
    def __init__(self, save_model_name=None, load_model_name=None, save_model=True, jpeg_compression_quality=None):
        super().__init__(dataset_name='imagenet', save_model_name=save_model_name, load_model_name=load_model_name,
                         save_model=save_model, jpeg_compression_quality=jpeg_compression_quality)

    def build_model(self):
        inputs = tf.keras.Input(shape=(224, 224, 3))
        preprocessed_input = inputs
        if self.jpeg_compression_quality is not None:
            preprocessed_input = jpeg_compression_for_rgb_model(self.ds_name, preprocessed_input,
                                                                self.jpeg_compression_quality)
        preprocessed_input = tf.keras.applications.resnet_v2.preprocess_input(preprocessed_input)
        tf_resnet = tf.keras.applications.resnet_v2.ResNet152V2(include_top=True, weights='imagenet',
                                                                input_shape=self.rgb_input_shape,
                                                                classifier_activation=None,
                                                                input_tensor=preprocessed_input)
        return tf_resnet


class ImagenetDensenet(adversarial_attacks.models.models.Model):
    def __init__(self, save_model_name=None, load_model_name=None, save_model=True, jpeg_compression_quality=None):
        super().__init__(dataset_name='imagenet', save_model_name=save_model_name, load_model_name=load_model_name,
                         save_model=save_model, jpeg_compression_quality=jpeg_compression_quality)

    def build_model(self):
        inputs = tf.keras.Input(shape=(224, 224, 3))
        preprocessed_input = inputs
        if self.jpeg_compression_quality is not None:
            preprocessed_input = jpeg_compression_for_rgb_model(self.ds_name, preprocessed_input,
                                                                self.jpeg_compression_quality)
        preprocessed_input = tf.keras.applications.densenet.preprocess_input(preprocessed_input)
        tf_densenet = tf.keras.applications.densenet.DenseNet201(include_top=True, weights='imagenet',
                                                                 input_shape=(224, 224, 3),
                                                                 classifier_activation=None,
                                                                 input_tensor=preprocessed_input)
        return tf_densenet


class ImagenetVGG16(adversarial_attacks.models.models.Model):
    def __init__(self, save_model_name=None, load_model_name=None, save_model=True, jpeg_compression_quality=None):
        super().__init__(dataset_name='imagenet', save_model_name=save_model_name, load_model_name=load_model_name,
                         save_model=save_model, jpeg_compression_quality=jpeg_compression_quality)

    def build_model(self):
        inputs = tf.keras.Input(shape=(224, 224, 3))
        preprocessed_input = inputs
        if self.jpeg_compression_quality is not None:
            preprocessed_input = jpeg_compression_for_rgb_model(self.ds_name, preprocessed_input,
                                                                self.jpeg_compression_quality)
        preprocessed_input = tf.keras.applications.vgg16.preprocess_input(preprocessed_input)
        tf_vgg16 = tf.keras.applications.vgg16.VGG16(include_top=True, weights='imagenet',
                                                     input_shape=(224, 224, 3),
                                                     classifier_activation=None, input_tensor=preprocessed_input)
        return tf_vgg16


class CifarResNet(adversarial_attacks.models.models.Model):
    def __init__(self, save_model_name=None, load_model_name=None, save_model=True, jpeg_compression_quality=None):
        super().__init__(dataset_name='cifar10', save_model_name=save_model_name, load_model_name=load_model_name,
                         save_model=save_model, jpeg_compression_quality=jpeg_compression_quality)

    def build_model(self):
        resnet50 = tf.keras.applications.resnet50.ResNet50(input_shape=self.rgb_input_shape,
                                                           include_top=False, weights='imagenet')

        inputs = tf.keras.Input(shape=(32, 32, 3))
        preprocessed_input = inputs
        if self.jpeg_compression_quality is not None:
            preprocessed_input = jpeg_compression_for_rgb_model(self.ds_name, preprocessed_input,
                                                                self.jpeg_compression_quality)
        preprocessed_input = tf.keras.applications.resnet50.preprocess_input(preprocessed_input)
        output = resnet50(preprocessed_input)
        x = tf.keras.layers.BatchNormalization()(output)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        logits = tf.keras.layers.Dense(self.num_classes,
                                       activation=None)(x)
        classification_model = tf.keras.Model(inputs=inputs, outputs=logits)
        return classification_model


class CifarDensenet(adversarial_attacks.models.models.Model):
    def __init__(self, save_model_name=None, load_model_name=None, save_model=True, jpeg_compression_quality=None):
        super().__init__(dataset_name='cifar10', save_model_name=save_model_name, load_model_name=load_model_name,
                         save_model=save_model, jpeg_compression_quality=jpeg_compression_quality)

    def build_model(self):
        densenet = tf.keras.applications.densenet.DenseNet121(input_shape=self.rgb_input_shape,
                                                              include_top=False, weights='imagenet')

        inputs = tf.keras.Input(shape=(32, 32, 3))
        preprocessed_input = inputs
        if self.jpeg_compression_quality is not None:
            preprocessed_input = jpeg_compression_for_rgb_model(self.ds_name, preprocessed_input,
                                                                self.jpeg_compression_quality)
        preprocessed_input = tf.keras.applications.densenet.preprocess_input(preprocessed_input)
        output = densenet(preprocessed_input)
        x = tf.keras.layers.GlobalAveragePooling2D()(output)
        y = tf.keras.layers.Flatten(name='flatten')(x)
        logits = tf.keras.layers.Dense(self.num_classes,
                                       activation=None,
                                       name='logits')(y)
        classification_model = tf.keras.Model(inputs=inputs, outputs=logits)
        return classification_model
