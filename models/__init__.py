import tensorflow as tf

from .detector import ObjectClassifier

__all__ = [
    "ObjectClassifier",
]

gpu_devices = tf.config.get_visible_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
