"""Feature management for data preprocessing."""

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema


LABEL_COLUMN = 'label'

KEY_COLUMNS = ['series', 'subject']

IMAGE_COLUMN = 'image'

CATEGORICAL_COLUMNS = []
STRING_COLUMNS = []
NUMERIC_COLUMNS = [LABEL_COLUMN]
NUMERIC_LIST_COLUMNS = [IMAGE_COLUMN]
BOOLEAN_COLUMNS = []



def get_raw_feature_spec():
    """Returns TF feature spec for preprocessing"""
    features = dict(
        [(name, tf.FixedLenFeature([], tf.string))
            for name in CATEGORICAL_COLUMNS] +
        [(name, tf.FixedLenFeature([], tf.string))
            for name in STRING_COLUMNS] +
        [(name, tf.FixedLenFeature([], tf.float32))
            for name in NUMERIC_COLUMNS] +
        [(name, tf.FixedLenFeature([], tf.int64))
            for name in BOOLEAN_COLUMNS] +
        [(name, tf.FixedLenFeature([1, 256*256*256], tf.float32))
            for name in NUMERIC_LIST_COLUMNS]
    )
    return features


RAW_FEATURE_SPEC = get_raw_feature_spec()


def get_raw_dataset_metadata():
    return dataset_metadata.DatasetMetadata(
        dataset_schema.from_feature_spec(RAW_FEATURE_SPEC))


def preprocess_fn(inputs):
    """TensorFlow transform preprocessing function.

    Args:
        inputs: Dict of key to Tensor.
    Returns:
        Dict of key to transformed Tensor.
    """
    outputs = inputs.copy()
    for key in CATEGORICAL_COLUMNS:
        tft.vocabulary(inputs[key], vocab_filename=key)
    return outputs
