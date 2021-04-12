# Import frameworks for ML calculation.
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tensorflow_hub as hub

# Import other libraries.
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import time

""" The style transfer test with Arbitrary Image Stylization. """


# export CUDA_VISIBLE_DEVICES=-1


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Print information about TensorFlow and make sure it's running with GPU support.


class Unit:
    def __init__(self, content_image, style_image, output_image) -> None:

        self.content_image = plt.imread(content_image)
        self.style_image = plt.imread(style_image)
        self.hub_module = (
            "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
        )
        self.output_image = output_image

    def run(self):

        time_start = time.time()

        # Convert to float32 numpy array, add batch dimension, and normalize to range [0, 1]. Example using numpy:
        content_image = self.content_image.astype(np.float32)[np.newaxis, ...] / 255.0
        style_image = self.style_image.astype(np.float32)[np.newaxis, ...] / 255.0

        style_image = tf.image.resize(style_image, (256, 256))

        hub_module = hub.load(self.hub_module)
        outputs = hub_module(tf.constant(content_image), tf.constant(style_image))

        output_image = outputs[0]
        time_end = time.time()

        print("Elapsed time:", time_end - time_start, "sec")

        # Adjust figure size for the plot.
        mpl.rcParams["figure.figsize"] = (12, 12)
        mpl.rcParams["axes.grid"] = False

        # Show results.
        plt.subplot(1, 3, 1)
        plt.imshow(tf.squeeze(content_image, axis=0))
        plt.title("Content Image")

        plt.subplot(1, 3, 2)
        plt.imshow(tf.squeeze(style_image, axis=0))
        plt.title("Style Image")

        plt.subplot(1, 3, 3)
        plt.imshow(tf.squeeze(output_image, axis=0))
        plt.title("Output Image")

        plt.savefig(self.output_image)
