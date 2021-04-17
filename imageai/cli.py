# import tensorflow as tf
from tensorflow.python.eager.context import device
import tensorflow as tf
from tensorflow.python.client import device_lib
import click

from imageai.style import Style
from imageai.enhance import Enhance


def gpus_available():
    devices = device_lib.list_local_devices()

    if len(devices) > 1:
        print("Using GPU")
    else:
        print("GPU not available, using CPU")


@click.group()
def cli():

    gpus_available()

    print("TensorFlow version: {version}".format(version=tf.__version__))
    print("Eager mode enabled: {mode}".format(mode=tf.executing_eagerly()))
    print(
        "GPU available: {gpu_available}".format(
            gpu_available=tf.config.list_physical_devices("GPU")
        )
    )


@cli.command()
@click.argument("content")
@click.argument("style")
@click.argument("output")
def style(content, style, output):
    Style(content, style, output).run()


def enhance(input, output):
    Enhance(input)


if __name__ == "__main__":
    cli()