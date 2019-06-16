"""Trainer for 3D CNN MRI model."""

import sys
import argparse
import functools

import tensorflow as tf
import tensorflow_transform as tft

from trainer import input_util
from trainer import metadata

SEED = 123

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO'
    )
    parser.add_argument(
        '--train-batch-size',
        help='Batch size for each training step',
        type=int,
        default=120
    )
    parser.add_argument(
        '--eval-batch-size',
        help='Batch size for evaluation steps',
        type=int,
        default=1000
    )
    parser.add_argument(
        '--eval-start-secs',
        help='How long to wait before starting first evaluation',
        default=20,
        type=int
    )
    parser.add_argument(
        '--eval-steps',
        help="""Number of steps to run evaluation for at each checkpoint',
        Set to None to evaluate on the whole evaluation data
        """,
        default=None,
        type=int
    )
    parser.add_argument(
        '--num-epochs',
        help="""Maximum number of training data epochs.
        If both --train-size and --num-epochs are specified, --train-steps will
        be: (train-size/train-batch-size) * num-epochs.""",
        default=None,
        type=int
    )
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--input-dir',
        help='GCS or local directory to TFRecord and metadata files.',
        required=True
    )
    parser.add_argument(
        '--train-steps',
        help="""Steps to run the training job for.
        If --num-epochs and --train-size are specified, then --train-steps will
        be: (train-size/train-batch-size) * num-epochs""",
        default=10000,
        type=int
    )
    parser.add_argument(
        '--train-size',
        help='Size of training set (instance count)',
        type=int,
        default=None
    )
    parser.add_argument(
        '--learning-rate',
        help='Learning rate',
        default=0.021388123321319803,
        type=float
    )
    return parser.parse_args(argv)

def train_and_evaluate(flags):
    """Runs model training and evaluation using TF Estimator API"""
    #Get TF transform metadata generated during preprocessing
    tf_transform_output = tft.TFTransformOutput(flags.input_dir)

    #Define training spec
    feature_spec = tf.tf_transform_output.transformed_feature_spec()
    train_input_fn = functools.partial(
    	input_util.input_fn,
    	flags.input_dir,
    	tf.estimator.ModeKeys.TRAIN,
    	flags.train_batch_size,
    	flags.num_epochs,
    	label_name=metadata.LABEL_COLUMN,
    	feature_spec=feature_spec
    )
    train_spec = tf.estimator.TrainSpec(
        train_input_fn, max_steps=flags.train_steps)

    #Define eval spec
    eval_input_fn = functools.partial(
        input_util.input_fn,
        flags.input_dir,
        tf.estimator.ModeKeys.EVAL,
        flags.eval_batch_size,
        num_epochs=1,
        label_name=metadata.LABEL_COLUMN,
        feature_spec=feature_spec
    )

    

def main():
    #Parse command-line arguments
    flags = parse_arguments(sys.argv[1:])

    #Set python level verbosity
    tf.logging.set_verbosity(flags.verbosity)
    
    #Run model training and evaluate
    train_and_evaluate(flags)

if __name__ == '__main__':
    main()
