"""TensorFlow model definition."""

import tensorflow as tf
import numpy as np

from trainer import metrics


def model_fn(features, labels, mode, params, hidden_units, reg_rate):
    """Constructs 3D CNN model"""

    image = features
    print(params)

    if isinstance(image, dict):
        image = features["image"]

    with tf.contrib.tpu.bfloat16_scope():

        x = tf.reshape(features, [-1, 128, 128, 16, 1])
        #squeezenet
        conv1 = tf.layers.conv3d(
            inputs=x,
            filters=96,
            stride=2,
            kernel_size=[7,7,7],
            activation=tf.nn.relu
        )
        pool1 = tf.layers.max_pooling3d(
            inputs=conv1,
            pool_size=[3, 3, 3],
            strides=[2, 2, 2]
        )
        squeeze1_1 = tf.layers.conv3d(
            inputs=pool1,
            filters=16,
            kernel_size=[1,1,1],
            activation=tf.nn.relu
        )
        squeeze1_2 = tf.layers.conv3d(
            inputs=squeeze1_1,
            filters=64,
            kernel_size=[1,1,1],
            activation=tf.nn.relu
        )
        squeeze1_3 = tf.layers.conv3d(
            inputs=squeeze1_2,
            filters=64,
            kernel_size=[3,3,3],
            activation=tf.nn.relu
        )
        squeeze2_1 = tf.layers.conv3d(
            inputs=squeeze1_3,
            filters=16,
            kernel_size=[1,1,1],
            activation=tf.nn.relu
        )
        squeeze2_2 = tf.layers.conv3d(
            inputs=squeeze2_1,
            filters=64,
            kernel_size=[1,1,1],
            activation=tf.nn.relu
        )
        squeeze2_3 = tf.layers.conv3d(
            inputs=squeeze2_2,
            filters=64,
            kernel_size=[3,3,3],
            activation=tf.nn.relu
        )
        squeeze3_1 = tf.layers.conv3d(
            inputs=squeeze2_3,
            filters=32,
            kernel_size=[1,1,1],
            activation=tf.nn.relu
        )
        squeeze3_2 = tf.layers.conv3d(
            inputs=squeeze3_1,
            filters=128,
            kernel_size=[1,1,1],
            activation=tf.nn.relu
        )
        squeeze3_3 = tf.layers.conv3d(
            inputs=squeeze3_3,
            filters=128,
            kernel_size=[3,3,3],
            activation=tf.nn.relu
        )
        pool2 = tf.layers.max_pooling3d(
            inputs=squeeze3_3,
            pool_size=[3, 3, 3],
            strides=[2, 2, 2]
        )
        squeeze4_1 = tf.layers.conv3d(
            inputs=pool2,
            filters=32,
            kernel_size=[1,1,1],
            activation=tf.nn.relu
        )
        squeeze4_2 = tf.layers.conv3d(
            inputs=squeeze4_1,
            filters=128,
            kernel_size=[1,1,1],
            activation=tf.nn.relu
        )
        squeeze4_3 = tf.layers.conv3d(
            inputs=squeeze4_2,
            filters=128,
            kernel_size=[3,3,3],
            activation=tf.nn.relu
        )

        squeeze5_1 = tf.layers.conv3d(
            inputs=squeeze4_3,
            filters=48,
            kernel_size=[1,1,1],
            activation=tf.nn.relu
        )
        squeeze5_2 = tf.layers.conv3d(
            inputs=squeeze5_1,
            filters=192,
            kernel_size=[1,1,1],
            activation=tf.nn.relu
        )
        squeeze5_3 = tf.layers.conv3d(
            inputs=squeeze5_2,
            filters=192,
            kernel_size=[3,3,3],
            activation=tf.nn.relu
        )
        squeeze6_1 = tf.layers.conv3d(
            inputs=squeeze5_3,
            filters=48,
            kernel_size=[1,1,1],
            activation=tf.nn.relu
        )
        squeeze6_2 = tf.layers.conv3d(
            inputs=squeeze6_1,
            filters=192,
            kernel_size=[1,1,1],
            activation=tf.nn.relu
        )
        squeeze6_3 = tf.layers.conv3d(
            inputs=squeeze6_2,
            filters=192,
            kernel_size=[3,3,3],
            activation=tf.nn.relu
        )
        squeeze7_1 = tf.layers.conv3d(
            inputs=squeeze6_3,
            filters=64,
            kernel_size=[1,1,1],
            activation=tf.nn.relu
        )
        squeeze7_2 = tf.layers.conv3d(
            inputs=squeeze7_1,
            filters=256,
            kernel_size=[1,1,1],
            activation=tf.nn.relu
        )
        squeeze7_3 = tf.layers.conv3d(
            inputs=squeeze7_2,
            filters=256,
            kernel_size=[3,3,3],
            activation=tf.nn.relu
        )
        pool3 = tf.layers.max_pooling3d(
            inputs=squeeze7_3,
            pool_size=[3, 3, 3],
            strides=[2, 2, 2]
        )
        squeeze8_1 = tf.layers.conv3d(
            inputs=pool3,
            filters=64,
            kernel_size=[1,1,1],
            activation=tf.nn.relu
        )
        squeeze8_2 = tf.layers.conv3d(
            inputs=squeeze8_1,
            filters=256,
            kernel_size=[1,1,1],
            activation=tf.nn.relu
        )
        squeeze8_3 = tf.layers.conv3d(
            inputs=squeeze8_2,
            filters=256,
            kernel_size=[3,3,3],
            activation=tf.nn.relu
        )
        conv2 = tf.layers.conv3d(
            inputs=squeeze8_3,
            filters=1000,
            stride=1,
            kernel_size=[1,1,1],
            activation=tf.nn.relu
        )
        logits = tf.squeeze(
            conv2,
            [6]
        )
        # conv1 = tf.layers.conv3d(
        #     inputs=x,
        #     filters=96,
        #     kernel_size=[3,3,3],
        #     activation=tf.nn.relu
        # )

        # conv2 = tf.layers.conv3d(
        #     inputs=conv1,
        #     filters=32,
        #     kernel_size=[3,3,3],
        #     activation=tf.nn.relu
        # )

        # pool1 = tf.layers.max_pooling3d(
        #     inputs=conv2,
        #     pool_size=[2, 2, 2],
        #     strides=[2, 2, 2]
        # )

        # conv4 = tf.layers.conv3d(
        #     inputs=pool1,
        #     filters=64,
        #     kernel_size=[3,3,3],
        #     activation=tf.nn.relu
        # )

        # conv5 = tf.layers.conv3d(
        #     inputs=conv4,
        #     filters=128,
        #     kernel_size=[3,3,3],
        #     activation=tf.nn.relu
        # )

        # pool2 = tf.layers.max_pooling3d(
        #     inputs=conv5,
        #     pool_size=[2, 2, 2],
        #     strides=[2, 2, 2]
        # )

        layer = tf.layers.flatten(pool2)

        for units in hidden_units:
            layer = tf.layers.dense(
                inputs=layer, units=units, activation=tf.nn.relu)

        batch_norm = tf.layers.batch_normalization(layer)

        # regularization
        dropout = tf.layers.dropout(batch_norm, rate=reg_rate)

        # logits = tf.layers.dense(
        #     inputs=dropout, units=6)

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
