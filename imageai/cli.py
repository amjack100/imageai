import tensorflow as tf
import fire
from tensorflow.python.eager.context import device

from imageai.style import Unit
from tensorflow.python.client import device_lib


def gpus_available():
    devices = device_lib.list_local_devices()

    if len(devices) > 1:
        print("Using GPU")
    else:
        print("GPU not available, using CPU")


def cli_main():

    gpus_available()

    print("TensorFlow version: {version}".format(version=tf.__version__))
    print("Eager mode enabled: {mode}".format(mode=tf.executing_eagerly()))
    print(
        "GPU available: {gpu_available}".format(
            gpu_available=tf.config.list_physical_devices("GPU")
        )
    )

    fire.Fire(Unit)
    # print("Poo!")