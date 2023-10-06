import tensorflow as tf

# For experiments, the default strategy (seems more effective) as the data does not have to be loaded to all devices,
# at least for cifar10
# For training, we use the MirroredStrategy by default
# the init function is called by the main.init()

tf_distribution_strategy = tf.distribute.get_strategy()


def init_strategy(strategy='default'):
    """
    strategy in ['default', 'mirrored']
    """
    
    global tf_distribution_strategy
    if strategy == 'default':
        tf_distribution_strategy = tf.distribute.get_strategy()
    elif strategy == 'one_device':
        tf_distribution_strategy = tf.distribute.OneDeviceStrategy()
    elif strategy == 'mirrored':
        tf_distribution_strategy = tf.distribute.MirroredStrategy(cross_device_ops=
                                                                  tf.distribute.HierarchicalCopyAllReduce())
