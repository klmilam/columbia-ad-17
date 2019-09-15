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

# Model specific parameters


tf.flags.DEFINE_bool('use_tpu', True, 'Use TPUs rather than plain CPUs')
tf.flags.DEFINE_bool('enable_predict', True, 'Do some predictions at the end')
tf.flags.DEFINE_integer('iterations', 30,
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
        '--data-dir',
        type=str,
        default='gs://internal-klm-tpu/mri/128_128_16/20190915162022/'
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


def main(argv):
    # Parse command-line arguments
    params = parse_arguments(argv[1:])
    
    #Run model training and evaluate
    train_and_evaluate(params)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)
    absl_app.run(main)
