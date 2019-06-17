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


def _cnn_model_fn(features, labels, mode, params):
    """3D CNN model to classify Alzheimer's."""
    #Create model

    #Compute predictions
    input_layer = tf.reshape(features, [-1, 256, 256, 256, 1])
    conv1 = tf.layers.conv3d(
        inputs=input_layer,
        filters=64,
        kernel_size=[7,7,7],
        strides=[2, 2, 2],
        padding='same',
        activation=tf.nn.relu
    )
    pool1 = tf.layers.max_pooling3d(
        inputs=conv1, pool_size=[3, 3, 3], strides=2)
    conv2a = tf.layers.conv3d(
        inputs=pool1,
        filters=256,
        kernel_size=[1,1,1],
        strides=2,
        padding='same',
        activation=tf.nn.relu
    )
    conv2b = tf.layers.conv3d(
        inputs=conv2a,
        filters=64,
        kernel_size=[3,3,3],
        strides=2,
        padding='same',
        activation=tf.nn.relu
    )
    conv2c = tf.layers.conv3d(
        inputs=conv2b,
        filters=64,
        kernel_size=[1,1,1],
        strides=2,
        padding='same',
        activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling3d(
        inputs=conv2c, pool_size=[2,2,2], strides=2)
    pool4_flat = tf.layers.flatten(pool2)
    dense1 = tf.layers.dense(
        inputs=pool4_flat, units=32, activation=tf.nn.relu)
    dense2 = tf.layers.dense(
        inputs=dense1, units=64, activation=tf.nn.relu)
    logits = tf.layers.dense(
        inputs=dense2, units=6, activation=tf.nn.softmax,
        kernel_initializer=tf.random_normal_initializer(stddev=.01))
    one_hot_labels = tf.one_hot(labels, 6)
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels))
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Compute the current epoch and associated learning rate from global_step.
        global_step = tf.train.get_global_step()
        batches_per_epoch = 7943/params['batch_size']
        learning_rate = tf.train.exponential_decay(
            learning_rate=0.01, global_step=global_step,
            decay_steps=100, decay_rate=0.005)
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=0.0005)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss,
                global_step=global_step)
        gs_t = tf.reshape(global_step, [1])
        loss_t = tf.reshape(loss, [1])
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op)
    return tf.estimatr.EstimatorSpec(
        mode,
        loss=loss)

def build_estimator(run_config, flags, feature_columns):
    """Returns TensorFlow estimator"""

    estimator = tf.estimator.Estimator(
        model_fn=_cnn_model_fn,
        model_dir=flags.job_dir,
        config=run_config,
        params={
            'batch_size': flags.train_batch_size
        }
    )
    return estimator