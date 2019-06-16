"""TensorFlow model definition."""

import tensorflow as tf


TF_NUMERIC_TYPES = [
    tf.float16,
    tf.float32,
    tf.float64,
    tf.int8,
    tf.int16,
    tf.int32,
    tf.int64,
]


def get_feature_columns(tf_transform_output, exclude_columns=[]):
    """Returns list of feature columns for a TensorFlow estimator.
    Args:
        tf_transform_output: tensorflow_transform.TFTransformOutput.
        exclude_columns: `tf_transform_ooutput` column names to be excluded
            from feature columns.

    Returns:
        List of TensorFlow feature columns.
    """
    feature_columns = []
    feature_spec = tf_transform_output.transformed_feature_spec()

    for col in exclude_columns:
        _ = feature_spec.pop(col, None)

    for k, v in feature_spec.items():
        if v.dtype in TF_NUMERIC_TYPES:
            feature_columns.append(tf.feature_column.numeric_column(
                k, dtype=v.dtype))
        elif v.dtype == tf.string:
            vocab_file = tf_transform_output.vocabulary_file_by_name(
                vocab_filename=k)
            feature_column = \
                tf.feature_column.categorical_column_with_vocabulary_file(
                    k,
                    vocab_file)
            feature_columns.append(tf.feature_column.indicator_column(
                feature_column))
    return feature_columns


def cnn_model(features, labels, mode, params):
    """3D CNN model to classify Alzheimer's."""
    #Create model

    #Compute predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)        


def build_estimator(run_config, flags, feature_columns, num_intervals):
    """Returns TensorFlow estimator"""

    estimator = tf.estimator.Estimator(
        model_fn=cnn_model,
        model_dir=flags.job_dir,
        config=run_config,
        params={}
    )
    return estimator