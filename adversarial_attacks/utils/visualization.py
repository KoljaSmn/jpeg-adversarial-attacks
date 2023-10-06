import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def show_multiple_images(images_list, labels_list=None, figsize=None):
    """
    Plots multiple images using matplotlib.
    """
    if figsize is None:
        figsize = (10, 10)
    fig = plt.figure(figsize=figsize)

    images = np.asarray(images_list)

    if labels_list is None:
        labels_list = ['' for i in range(len(images))]

    chunk_size = 8

    images_chunks = [images[x:x + chunk_size] for x in range(0, len(images), chunk_size)]
    labels_chunks = [labels_list[x:x + chunk_size] for x in range(0, len(labels_list), chunk_size)]
    columns = len(images_chunks[0])
    rows = len(images_chunks)

    img_idx = 1

    for chunk_idx in range(len(images_chunks)):
        images_chunk, labels_chunk = images_chunks[chunk_idx], labels_chunks[chunk_idx]
        n_images = len(images_chunk)

        for i in range(n_images):
            fig.add_subplot(rows, columns, img_idx)
            plt.imshow(tf.cast(images_chunk[i], tf.uint8), cmap='gray')
            plt.title(labels_chunk[i])
            img_idx += 1

    plt.show()


def show_rgb(image):
    plt.imshow(image)
    plt.show()
