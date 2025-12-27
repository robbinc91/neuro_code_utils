"""Keras utilities.

Contains helpers to load Keras models without requiring the caller to pass
`custom_objects`. When loading fails due to unknown custom classes/functions,
this module provides placeholder implementations so the model can still be
deserialized for inspection or inference (placeholders act as identity
operations and will not reproduce custom behavior).
"""
from typing import Any
import re

import tensorflow as tf


class _PlaceholderLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, inputs, *args, **kwargs):
        return inputs

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class _PlaceholderDict(dict):
    """Dict that creates placeholder objects for any missing key.

    When Keras attempts to deserialize an unknown custom object it will look
    into `custom_objects`. By returning a placeholder class/function for any
    unknown name we avoid requiring the user to supply `custom_objects`.
    """
    def __missing__(self, key: str) -> Any:
        # Create a dynamic subclass with the requested name so reprs look reasonable
        placeholder_cls = type(key, (_PlaceholderLayer,), {})
        # Also provide a simple function fallback (e.g., for activations)
        def _identity(x, *a, **k):
            return x

        # Register both forms: the class and a function
        self[key] = placeholder_cls
        # a lowercase function name fallback
        self[key.lower()] = _identity
        return placeholder_cls


def load_keras_model_no_custom(model_path: str, compile: bool = True, verbose: bool = False):
    """Load a Keras model without requiring `custom_objects`.

    If standard `tf.keras.models.load_model` fails due to missing custom
    classes or functions, this function retries loading while providing
    placeholder implementations for any unknown names. Placeholders are
    identity operations and may change runtime behavior; use only for
    inspection or as a starting point to reimplement missing custom objects.

    Args:
        model_path: Path to SavedModel directory or HDF5 file.
        compile: Whether to compile the model after loading.
        verbose: If True, print fallback info when placeholders are used.

    Returns:
        A `tf.keras.Model` instance (possibly containing placeholder layers).
    """
    try:
        return tf.keras.models.load_model(model_path, compile=compile)
    except Exception as e:
        if verbose:
            print("Initial load failed; attempting to load with placeholders:", str(e))
        placeholders = _PlaceholderDict()
        # Retry with placeholder custom_objects
        model = tf.keras.models.load_model(model_path, custom_objects=placeholders, compile=compile)
        if verbose:
            print("Loaded model with placeholders for unknown custom objects.")
        return model
