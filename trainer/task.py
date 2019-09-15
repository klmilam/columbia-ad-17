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
    "tpu", default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")
tf.flags.DEFINE_string(
    "tpu_zone", default=None,
    help="[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
tf.flags.DEFINE_string(
    "gcp_project", default=None,
    help="[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

# Model specific parameters
tf.flags.DEFINE_string("data_dir", "",
                         "Path to directory containing the MNIST dataset")
tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.flags.DEFINE_integer("train_steps", 1000, "Total number of training steps.")
tf.flags.DEFINE_integer("eval_steps", 10,
                        "Total number of evaluation steps. If `0`, evaluation "
                        "after training is skipped.")
tf.flags.DEFINE_float("learning_rate", 0.0000005, "Learning rate.")

tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_bool("enable_predict", True, "Do some predictions at the end")
tf.flags.DEFINE_integer("iterations", 30,
                        "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")

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
    return parser.parse_args(argv)

def train_and_evaluate(flags):
    """Runs model training and evaluation using TF Estimator API"""
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu,
        zone=FLAGS.tpu_zone,
        project=FLAGS.gcp_project
    )


def main():
    #Parse command-line arguments
    flags = parse_arguments(sys.argv[1:])

    #Set python level verbosity
    tf.logging.set_verbosity(flags.verbosity)
    
    #Run model training and evaluate
    train_and_evaluate(flags)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)
    absl_app.run(main)
