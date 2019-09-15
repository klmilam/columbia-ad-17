"""Trainer for 3D CNN MRI model."""

import sys
import os
import argparse
import functools

import tensorflow as tf
import tensorflow_transform as tft

from absl import app as absl_app

from trainer import input_util
from trainer import metadata
from trainer import model

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
        type='str',
        default="",
        help="""Pass 'fixed' to use fixed class weights (based on global frequency)"""
    )
    parser.add_argument(
        '--fixed-weights',
        default=[1,1,1,1,1,1]
    )
    parser.add_argument(
        '--data-dir',
        type='str',
        default='gs://internal-klm-tpu/mri/128_128_16/20190915162022/'
    )
    parser.add_argument(
        '--model-dir',
        type='str',
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
    return parser.parse_args(argv)


def train_and_evaluate(params):
    """Runs model training and evaluation using TF Estimator API"""
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu,
        zone=FLAGS.tpu_zone,
        project=FLAGS.gcp_project
    )


def main():
    # Parse command-line arguments
    params = parse_arguments(sys.argv[1:])

    tf.logging.set_verbosity(flags.verbosity)
    
    #Run model training and evaluate
    train_and_evaluate(params)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)
    absl_app.run(main)
