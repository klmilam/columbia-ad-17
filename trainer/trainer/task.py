"""Trainer for 3D CNN MRI model."""

import sys
import argparse
import functools

import tensorflow as tf
import tensorflow_transform as tft

from trainer import input_util
from trainer import metadata
from trainer import model

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
        default=8
    )
    parser.add_argument(
        '--eval-batch-size',
        help='Batch size for evaluation steps',
        type=int,
        default=8
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
    parser.add_argument(
        '--tpu',
        default=None,
        help="""The name or GRPC URL of the TPU node.
        Leave it as `None` when training on AI Platform."""
    )
    parser.add_argument(
        '--use-tpu',
        action='store_true',
        help='Include flag if using TPU.'
    )
    return parser.parse_args(argv)

def train_and_evaluate(flags):
    """Runs model training and evaluation using TF Estimator API"""
    #Get TF transform metadata generated during preprocessing
    tf_transform_output = tft.TFTransformOutput(flags.input_dir)

    #Define training spec
    feature_spec = tf_transform_output.transformed_feature_spec()
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
    exporter = tf.estimator.FinalExporter(
        "export", functools.partial(
            input_util.tfrecord_serving_input_fn,
            feature_spec,
            label_name=metadata.LABEL_COLUMN))

    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        steps=flags.eval_steps,
        start_delay_secs=flags.eval_start_secs,
        exporters=[exporter],
        name='MRI-eval'
    )
    steps_per_run_train = 7943 // (flags.train_batch_size * 4)
    steps_per_run_eval = 964 // (flags.eval_batch_size * 4)

    #additional configs required for using TPUs
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        flags.tpu)
    tpu_config = tf.contrib.tpu.TPUConfig(
        num_shards=8, # using Cloud TPU v2-8
        iterations_per_loop=200)
    #Define training config
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=flags.job_dir,
        tpu_config=tpu_config,
        save_checkpoints_steps=200,
        save_summary_steps=100)
    #Build the estimator
    feature_columns = model.get_feature_columns(
        tf_transform_output, exclude_columns=metadata.NON_FEATURE_COLUMNS)

    estimator = model.build_estimator(run_config, flags, feature_columns)

    #Run training and evaluation
    #tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    estimator.train(train_input_fn)

def main():
    #Parse command-line arguments
    flags = parse_arguments(sys.argv[1:])

    #Set python level verbosity
    tf.logging.set_verbosity(flags.verbosity)
    
    #Run model training and evaluate
    train_and_evaluate(flags)

if __name__ == '__main__':
    main()
