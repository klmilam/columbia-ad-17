"""Trainer for 3D CNN MRI model."""

import sys
import argparse
import functools

import tensorflow as tf
import tensorflow_transform as tft

from absl import app as absl_app

from trainer import input_util
from trainer import metadata
from trainer import model

SEED = 123


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
    return 0


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
