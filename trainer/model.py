"""TensorFlow model definition."""

import tensorflow as tf
import numpy as np


def model_fn(features, labels, mode, params):
    """Constructs 3D CNN model"""

    del params
    image = features

    if isinstance(image, dict):
        image = features["image"]

    with tf.contrib.tpu.bfloat16_scope():

        x = tf.reshape(features, [-1, 128, 128, 16, 1])

        conv1 = tf.layers.conv3d(
            inputs=x,
            filters=16,
            kernel_size=[3,3,3],
            activation=tf.nn.relu
        )

        conv2 = tf.layers.conv3d(
            inputs=conv1,
            filters=32,
            kernel_size=[3,3,3],
            activation=tf.nn.relu
        )

        pool1 = tf.layers.max_pooling3d(
            inputs=conv2,
            pool_size=[2, 2, 2],
            strides=[2, 2, 2]
        )

        conv4 = tf.layers.conv3d(
            inputs=pool1,
            filters=64,
            kernel_size=[3,3,3],
            activation=tf.nn.relu
        )

        conv5 = tf.layers.conv3d(
            inputs=conv4,
            filters=128,
            kernel_size=[3,3,3],
            activation=tf.nn.relu
        )

        pool2 = tf.layers.max_pooling3d(
            inputs=conv5,
            pool_size=[2, 2, 2],
            strides=[2, 2, 2]
        )

        pool2_flat = tf.layers.flatten(pool2)

        dense1 = tf.layers.dense(
            inputs=pool2_flat, units=512, activation=tf.nn.relu)

        dense2 = tf.layers.dense(
            inputs=dense1, units=512, activation=tf.nn.relu)

        batch_norm = tf.layers.batch_normalization(dense2)

        # regularization
        dropout = tf.layers.dropout(batch_norm, rate=0.5)

        logits = tf.layers.dense(
            inputs=dropout, units=6)

        logits = tf.cast(logits, tf.float32) # Casting necessary to use bfloat32
        probabilities = tf.nn.softmax(logits)
        class_ids = tf.argmax(logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'pred_class_ids': class_ids,
            'actual_class_ids': labels,
            'probabilities': probabilities
        }
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions=predictions
        )

    labels_onehot = tf.one_hot(labels, 6)
    counts = tf.reduce_sum(labels_onehot, axis=0)

    if params.weight_type == "global_frequency":
        counts = params.fixed_weights
        class_weights = np.sum(counts)/counts
        class_weights = tf.reshape(class_weights, [6,1])
    elif params.weight_type == "batch_frequency":
        class_weights = np.sum(counts)/counts
        class_weights = tf.reshape(class_weights, [6,1])
    else:
        beta = .999
        class_weights = (1.0 - beta)/(1.0 - tf.math.pow(beta, counts))
        class_weights = tf.reshape(class_weights, [6,1])
        class_weights = tf.reshape(
           class_weights/tf.reduce_sum(class_weights), [6,1])
