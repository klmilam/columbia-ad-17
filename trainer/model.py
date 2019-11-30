"""TensorFlow model definition."""

import tensorflow as tf
import numpy as np

from trainer import metrics


def model_fn(features, labels, mode, params, hidden_units, reg_rate, cnn_filters):
    """Constructs 3D CNN model"""

    image = features
    print(params)

    if isinstance(image, dict):
        image = features["image"]

    with tf.contrib.tpu.bfloat16_scope():

        x = tf.reshape(features, [-1, 128, 128, 16, 1])
        conv1 = tf.layers.conv3d(
            inputs=x,
            filters=cnn_filters[0],
            kernel_size=[3,3,3],
            activation=tf.nn.relu
        )

        conv2 = tf.layers.conv3d(
            inputs=conv1,
            filters=cnn_filters[1],
            kernel_size=[3,3,3],
            activation=tf.nn.relu
        )

        pool1 = tf.layers.max_pooling3d(
            inputs=conv2,
            pool_size=[2, 2, 2],
            strides=[2, 2, 2]
        )

        conv3 = tf.layers.conv3d(
            inputs=pool1,
            filters=cnn_filters[2],
            kernel_size=[3,3,3],
            activation=tf.nn.relu
        )

        conv4 = tf.layers.conv3d(
            inputs=conv3,
            filters=cnn_filters[3],
            kernel_size=[3,3,3],
            activation=tf.nn.relu
        )

        pool2 = tf.layers.max_pooling3d(
            inputs=conv4,
            pool_size=[2, 2, 2],
            strides=[2, 2, 2]
        )

        layer = tf.layers.flatten(pool2)

        for units in hidden_units:
            layer = tf.layers.dense(
                inputs=layer, units=units, activation=tf.nn.relu)

        batch_norm = tf.layers.batch_normalization(layer)

        # regularization
        dropout = tf.layers.dropout(batch_norm, rate=reg_rate)

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

    if params["weight_type"] == "global_frequency":
        counts = np.asarray(params["fixed_weights"])
        class_weights = np.sum(counts)/counts
        class_weights = tf.reshape(class_weights, [6,1])
    elif params["weight_type"] == "batch_frequency":
        class_weights = np.sum(counts)/counts
        class_weights = tf.reshape(class_weights, [6,1])
    else:
        beta = params["beta"]
        class_weights = (1.0 - beta)/(1.0 - tf.math.pow(beta, counts))
        class_weights = tf.reshape(class_weights, [6,1])

    weights = tf.matmul(
        labels_onehot,
        tf.cast(class_weights, tf.float32))

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits,
        weights=weights)
    loss = tf.reduce_mean(loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=params["learning_rate"])
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
       
        predictions = {
            'class_ids': class_ids,
            'probabilities': probabilities
        }
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            predictions=predictions,
            eval_metrics=(metrics.metric_fn, [labels, logits]),
            train_op=optimizer.minimize(loss, tf.train.get_global_step())
    )

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metrics=(metrics.metric_fn, [labels, logits])
        )
