import tensorflow as tf

from .object_classifier import ObjectClassifier

__all__ = [
    "ObjectClassifier",
]

gpu_devices = tf.config.get_visible_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
    # tf.config.set_logical_device_configuration(
    #     device,
    #     [
    #         tf.config.LogicalDeviceConfiguration(memory_limit=2000),
    #     ],
    # )
