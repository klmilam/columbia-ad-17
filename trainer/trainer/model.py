"""TensorFlow model definition."""

import tensorflow as tf


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