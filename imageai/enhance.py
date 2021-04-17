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
        sr_img = model(self.input)
        sr_img = tf.squeeze(sr_img)

    def _save_results(self, img):

        if not isinstance(img, Image.Image):
            image = tf.clip_by_value(img, 0, 255)

    def _preprocess(self):
        pass
