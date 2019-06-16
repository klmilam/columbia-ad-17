"""Input functions."""

import os
import multiprocessing

import tensorflow as tf
import tensorflow_transform as tft

def read_and_decode_fn(example):
    fn = lambda examples: tf.parse_example(example, feature_spec)
    return fn

def input_fn(input_dir, mode, batch_size=1, label_name=None, 
    feature_spec=None):
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
            batch_size=BATCH_SIZE,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    )
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    if mode == tf.estimator.ModeKeys.PREDICT:
        return features

    label = features.pop(label_name)
    return features, label
