import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt


from imageai.base import ImgModule


class Enhance(ImgModule):

    hub_module = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

    def __init__(self, input: str, output: str) -> None:

        self.input = input
        self.output = output

    def run(self):
        model = hub.load(self.hub_module)
        input = self._preprocess()
        sr_img = model(input)
        sr_img = tf.squeeze(sr_img)
        self._save_results(sr_img)

    def _save_results(self, img):
        img = tf.squeeze(img)
        if not isinstance(img, Image.Image):
            img = tf.clip_by_value(img, 0, 255)
            img = Image.fromarray(tf.cast(img, tf.uint8).numpy())
        img.save(self.output)

    def _preprocess(self):

        hr_image = tf.image.decode_image(tf.io.read_file(self.input))

        if hr_image.shape[-1] == 4:
            hr_image = hr_image[..., :-1]

        hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4

        hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
        hr_image = tf.cast(hr_image, tf.float32)
        return tf.expand_dims(hr_image, 0)
