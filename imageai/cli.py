import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf
import fire

from imageai.style import Unit


def cli_main():

    print("TensorFlow version: {version}".format(version=tf.__version__))
    print("Eager mode enabled: {mode}".format(mode=tf.executing_eagerly()))
    print(
        "GPU available: {gpu_available}".format(
            gpu_available=tf.config.list_physical_devices("GPU")
        )
    )

    fire.Fire(Unit)
    # print("Poo!")