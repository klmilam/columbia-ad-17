"""Trainer for 3D CNN MRI model."""

import sys
import os
import argparse
import functools
from datetime import datetime
import time
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_transform as tft

from absl import app as absl_app

import model
import input_util

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
    parser.add_argument(
        '--beta',
        type=float,
        default=.99999,
        help='Beta value for beta class weighting'
    )
    return parser.parse_args(argv)


def load_global_step_from_checkpoint_dir(checkpoint_dir):
  try:
    checkpoint_reader = tf.train.NewCheckpointReader(
        tf.train.latest_checkpoint(checkpoint_dir))
    return checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
  except:  # pylint: disable=bare-except
    return 0


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
        params=params,
        config=run_config)

    tf_transform_output = tft.TFTransformOutput(params.input_dir)
    feature_spec = tf_transform_output.transformed_feature_spec()

    train_input_fn = functools.partial(
        input_util.input_fn,
        params.input_dir,
        tf.estimator.ModeKeys.TRAIN,
        num_epochs=100,
        label_name='label',
        feature_spec=feature_spec)

    eval_input_fn = functools.partial(
        input_util.input_fn,
        params.input_dir,
        tf.estimator.ModeKeys.EVAL,
        num_epochs=1,
        label_name='label',
        feature_spec=feature_spec)

    predict_input_fn = functools.partial(
        input_util.input_fn,
        params.input_dir,
        tf.estimator.ModeKeys.PREDICT,
        num_epochs=1,
        label_name='label',
        feature_spec=feature_spec)

    start_timestamp = time.time()
    current_step = load_global_step_from_checkpoint_dir(params.model_dir)

    while current_step < int(params.train_steps):
        # Workaround to support training and evaluating with TPUs
        # Training stage
        next_checkpoint = min(current_step + 500, int(params.train_steps))
        estimator.train(
            input_fn=train_input_fn,
            max_steps=next_checkpoint)
        current_step = next_checkpoint
        tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                        next_checkpoint, int(time.time() - start_timestamp))

        # Evaluation stage
        tf.logging.info('Starting to evaluate at step %d.', next_checkpoint)
        eval_result = estimator.evaluate(
            input_fn=eval_input_fn,
            steps=params.eval_steps)
        tf.logging.info(
            'Eval results at step %d: %s', next_checkpoint, eval_result)

    elapsed_time = int(time.time() - start_timestamp)
    tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                    params.train_steps, elapsed_time)
    # Predictions stage
    predictions = estimator.predict(
        input_fn=predict_input_fn,
        yield_single_examples=False) # Make predictions a batch at a time
    predict_list = {} # Create a dict of lists to store predictions
    for p in predictions:
        for key in p.keys():
            if key not in predict_list:
                predict_list[key] = []
            if key == 'probabilities':
                predict_list[key].extend(p[key].flatten().reshape(-1, 6))
            else:
                predict_list[key].extend(p[key].flatten())

    predict_list = pd.DataFrame(predict_list).to_dict(
        'list') # Convert to list of dicts
    df = pd.DataFrame(predict_list)
    df.to_csv(os.path.join(
        params.model_dir, str(params.train_steps), "predictions.csv"))


def main(argv):
    # Parse command-line arguments
    params = parse_arguments(argv[1:])
    
    #Run model training and evaluate
    train_and_evaluate(params)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)
    absl_app.run(main)
