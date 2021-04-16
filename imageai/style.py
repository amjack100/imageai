# Import frameworks for ML calculation.

import tensorflow as tf
import tensorflow_hub as hub

# Import other libraries.
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import time
from PIL import Image


class Style:
    """
    The image stylization module
    """

    def __init__(self, content_image, style_image, output_image) -> None:

        self.analysis_mode = False
        self.images = (plt.imread(content_image), plt.imread(style_image))
        self.hub_module = (
            "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
        )
        self.output_image = output_image

    def _make_img_adjustments(self):
        """
        Make preprocessing adjustments based on image input file type and number
        of channels
        """

        self.images = [img.astype(np.float32)[np.newaxis, ...] for img in self.images]
        self.images = [img / 255.0 if np.max(img) > 1 else img for img in self.images]
        self.images = [
            img[:, :, :, 0:3] if img.shape[-1] == 4 else img for img in self.images
        ]

    def _save_normal(self, output):
        """
        Save the output file as expected with no extra annotations
        """
        img = Image.fromarray(tf.squeeze(output * 255, axis=0).numpy().astype(np.uint8))
        img.save(self.output_image)

    def _save_analytic(self, content, style, output):
        """
        Save a file showing the content, style, and output side by side
        Not yet implemented into the cli
        """

        mpl.rcParams["figure.figsize"] = (12, 12)
        mpl.rcParams["axes.grid"] = False

        plt.subplot(1, 3, 1)
        plt.imshow(tf.squeeze(content, axis=0))
        plt.title("Content Image")

        plt.subplot(1, 3, 2)
        plt.imshow(tf.squeeze(style, axis=0))
        plt.title("Style Image")

        plt.subplot(1, 3, 3)
        plt.imshow(tf.squeeze(output, axis=0))
        plt.title("Output Image")

        plt.savefig(self.output_image)

    def run(self):
        """
        Main calling method of the class which executes pre-processing,
        processing, and saving steps sequentially
        """
        time_start = time.time()
        self._make_img_adjustments()

        (content_image, style_image) = self.images

        style_image = tf.image.resize(style_image, (256, 256))

        hub_module = hub.load(self.hub_module)
        outputs = hub_module(tf.constant(content_image), tf.constant(style_image))

        output_image = outputs[0]
        time_end = time.time()

        print("Elapsed time:", time_end - time_start, "sec")

        if self.analysis_mode:
            self._save_analytic(content_image, style_image, output_image)
        else:
            self._save_normal(output_image)
