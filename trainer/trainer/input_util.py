"""Input functions."""

import os
import multiprocessing

import tensorflow as tf
import tensorflow_transform as tft



def input_fn(input_dir, mode, batch_size=1, num_epochs=100,
    label_name=None, feature_spec=None):
    """Reads TFRecords and returns the features and labels"""
    def read_and_decode_fn(example):
        """Parses Serialized Example."""
        features = tf.parse_single_example(example, feature_spec)
        image = features['image']
        image = tf.reshape(image, [256, 256, 256])
        label = tf.cast(features[label_name], tf.int32)
        return image, label

    if feature_spec is None:
        tf_transform_output = tft.TFTransformOutput(
            os.path.join(input_dir, 'transformed_metadata'))
        feature_spec = tf_transform_output.transformed_feature_spec()

    prefix = str(mode).lower()
    suffix = '.tfrecord'
    num_cpus = multiprocessing.cpu_count()
    file_pattern = os.path.join(input_dir, 'data', prefix, prefix+'*'+suffix)
    filenames = tf.matching_files(file_pattern)
    dataset = tf.data.TFRecordDataset(filenames=filenames, buffer_size=None,
                                      num_parallel_reads=num_cpus)
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.apply(
            tf.data.experimental.shuffle_and_repeat(buffer_size=100)
        )
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            map_func=read_and_decode_fn,
            batch_size=batch_size,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    )
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    if mode == tf.estimator.ModeKeys.PREDICT:
        return features
    return features, labels


def tfrecord_serving_input_fn(feature_spec, label_name=None):
    """Creates ServingInputReceiver for TFRecord inputs"""
    if label_name:
        _ = feature_spec.pop(label_name)

    serving_input_receiver = (
        tf.estimator.export.build_parsing_serving_input_receiver_fn(
            feature_spec)())

    return tf.estimator.export.ServingInputReceiver(
        serving_input_receiver.features, serving_input_receiver.receiver_tensors)
