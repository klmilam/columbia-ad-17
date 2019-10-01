"""Trainer for 3D CNN MRI model."""

import sys
import os
import argparse
import functools
from datetime import datetime
import numpy as np

import tensorflow as tf
import tensorflow_transform as tft

from absl import app as absl_app

import model

SEED = 123


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.path[0]))))

tf.flags.DEFINE_string(
    'tpu',
    default=None,
    help="""The Cloud TPU to use for training. This should be either the name
         used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470
         url.""")
tf.flags.DEFINE_string(
    'tpu_zone',
    default=None,
    help="""[Optional] GCE zone where the Cloud TPU is located in. If not
         specified, we will attempt to automatically detect the GCE project from
         metadata.""")
tf.flags.DEFINE_string(
    'gcp_project',
    default=None,
    help="""[Optional] Project name for the Cloud TPU-enabled project. If not 
         specified, we will attempt to automatically detect the GCE project from 
         metadata.""")

# TPU specific parameters

tf.flags.DEFINE_bool('use_tpu', True, 'Use TPUs rather than plain CPUs')
tf.flags.DEFINE_bool('enable_predict', True, 'Do some predictions at the end')
tf.flags.DEFINE_integer('iterations', 25,
                        'Number of iterations per TPU training loop.')
tf.flags.DEFINE_integer('num_shards', 8, 'Number of shards (TPU chips).')

FLAGS = tf.flags.FLAGS


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weight-type',
        type=str,
        default='',
        help="""Pass 'fixed' to use fixed class weights (based on global frequency)"""
    )
    parser.add_argument(
        '--fixed-weights',
        default=[1,1,1,1,1,1]
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='gs://internal-klm-tpu/mri/128_128_16/20190923020412/'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='gs://internal-klm-tpu/mri/model/'+ datetime.now().strftime('%Y%m%d%H%M%S'),
        help="""Location to write checkpoints and summaries to.
             Must be a GCS URI when using Cloud TPU."""
    )
    parser.add_argument(
        '--train-steps',
        type=int,
        default=1000,
        help='Total number of training steps.'
    )
    parser.add_argument(
        '--eval-steps',
        type=int,
        default=10,
        help='Total number of eval steps. If `0`, evaluation after training is skipped.'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default='0.0000005',
        help='learning rate'
    )
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=1024,
        help='Batch size for training'
    )
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=1024,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--predict-batch-size',
        type=int,
        default=1024,
        help='Batch size for prediction'
    )
    return parser.parse_args(argv)


def input_fn(input_dir, mode, num_epochs=100, label_name=None,
    feature_spec=None, params={}):
    """Reads TFRecords and returns the features and labels for all datasets."""

    def read_and_decode_fn(example):
        """Parses a Serialized Example."""
        features = tf.parse_single_example(example, feature_spec)
        image = features['image']
        image = tf.cast(image, tf.bfloat16) # TPUs don't support float64
        image = tf.reshape(image, shape=(160, 160, 32))
        label = tf.cast(features[label_name], tf.int32)
        return image, label

    def random_flip(image, label) -> tf.Tensor:
        """Randomly flips a training image for data augmentation purposes."""
        image = tf.image.random_flip_up_down(image)
        return image, label

    def random_crop(image, label) -> tf.Tensor:
        """Randomly crops a training image for data augmentation purposes."""
        image = tf.image.random_crop(image, size=[128, 128, 16])
        return image, label

    def crop(image, label) -> tf.Tensor:
        """Deterministically crops a testing or eval image.

        Returns:
            image: the middle 128 x 128 x 16 cube (a Tensor) of the input
                image.
            label: the un-transformed label
        """
        image = tf.slice(image, [16, 16, 8], [128, 128, 16])
        return image, label

    prefix = str(mode).lower()
    prefix = 'predict' if prefix == 'infer' else prefix
    suffix = '.tfrecord'
    file_pattern = os.path.join(
        input_dir, 'data', prefix, prefix + '*' + suffix)
    tf.logging.info(prefix + ' data from ' + file_pattern)
    filenames = tf.matching_files(file_pattern) # list of TFRecords files

    dataset = tf.data.TFRecordDataset(filenames=filenames, buffer_size=None)
    dataset = dataset.map(read_and_decode_fn, num_parallel_calls=100)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Apply data augmentation transformations to training dataset
        dataset = dataset.map(random_crop, num_parallel_calls=100)
        dataset = dataset.map(random_flip, num_parallel_calls=100)
    else:
        dataset = dataset.map(crop, num_parallel_calls=100)

    dataset = dataset.batch(
        params['batch_size'],
        drop_remainder=True) # Must drop remainder when working with TPUs
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    if mode == tf.estimator.ModeKeys.PREDICT:
        # Not necessary to repeat or shuffle prediction dataset
        return dataset

    dataset = dataset.repeat()
    dataset = dataset.shuffle(50)
    return dataset


def train_and_evaluate(params):
    """Runs model training and evaluation using TF Estimator API"""

    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu,
        zone=FLAGS.tpu_zone,
        project=FLAGS.gcp_project
    )

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=params.model_dir,
        save_summary_steps=100,
        session_config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True),
        save_checkpoints_steps=100,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=40,
            num_shards=FLAGS.num_shards,
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        )
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model.model_fn,
        use_tpu=FLAGS.use_tpu,
        train_batch_size=params.train_batch_size,
        eval_batch_size=params.eval_batch_size,
        predict_batch_size=params.predict_batch_size,
        params={
            "data_dir": params.data_dir,
            "class_weights": params.class_weights,
            "fixed_weights": np.asarray(params.fixed_weights)},
        config=run_config)

    tf_transform_output = tft.TFTransformOutput(params.input_dir)
    feature_spec = tf_transform_output.transformed_feature_spec()

    train_input_fn = functools.partial(
        input_fn,
        input_dir,
        tf.estimator.ModeKeys.TRAIN,
        num_epochs=100,
        label_name='label',
        feature_spec=feature_spec)

    eval_input_fn = functools.partial(
        input_fn,
        input_dir,
        tf.estimator.ModeKeys.EVAL,
        num_epochs=1,
        label_name='label',
        feature_spec=feature_spec)

    predict_input_fn = functools.partial(
        input_fn,
        input_dir,
        tf.estimator.ModeKeys.PREDICT,
        num_epochs=1,
        label_name='label',
        feature_spec=feature_spec)


def main(argv):
    # Parse command-line arguments
    params = parse_arguments(argv[1:])
    
    #Run model training and evaluate
    train_and_evaluate(params)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)
    absl_app.run(main)
