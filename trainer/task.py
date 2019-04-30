import argparse

from trainer import model

import os
import tensorflow as tf

#from tensorflow.contrib.learn import Experiment
#from tensorflow.contrib.learn.python.learn import learn_runner
#from tensorflow.contrib.learn.python.learn.utils import (
#    saved_model_export_utils)

BATCH_SIZE=8

def train_and_evaluate(args):
  def read_and_decode(serialized_example):
    features = tf.parse_single_example(
          serialized_example,
          features={
            'image': tf.FixedLenFeature([256*256*256], tf.float32),
            'label': tf.FixedLenFeature([], tf.int64)
          })
    image = features['image']
    image = tf.reshape(image, [256, 256, 256])
    image = tf.where(tf.is_nan(image), tf.zeros_like(image), image)
    label = tf.cast(features['label'], tf.int32)
    return image, label


  def make_input_fn(filenames, batch_size=1):
    dataset = tf.data.TFRecordDataset(filenames)
    #dataset = dataset.map(read_and_decode, num_parallel_calls=8)
    #TODO: increase shuffle buffer when using distributed training
    dataset = dataset.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=16)
      )
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(map_func=read_and_decode,
            batch_size=BATCH_SIZE,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
      )
    #dataset = dataset.shuffle(buffer_size=2)
    #dataset = dataset.batch(BATCH_SIZE)
    #dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

  def train_input():
    train_filenames = tf.gfile.Glob('gs://ieor-dl-group17/input-data/0413191822/train/*')
    return make_input_fn(train_filenames)

  def eval_input():
    val_filenames = tf.gfile.Glob('gs://ieor-dl-group17/input-data/0413191822/val/*')
    return make_input_fn(val_filenames)
  train_spec = tf.estimator.TrainSpec(
    train_input)

  eval_spec = tf.estimator.EvalSpec(
    eval_input)

  #run_config = tf.estimator.RunConfig(train_distribute=tf.contrib.distribute.MirroredStrategy())
  estimator = model.build_estimator(args.job_dir)#, config=run_config)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      help='GCS or local path to training data',
      default='gs://ieor-dl-group17/input-data/0413191822/',
      #required=True
  )
  parser.add_argument(
      '--train_batch_size',
      help='Batch size for training steps',
      type=int,
      default=10
  )
  parser.add_argument(
      '--eval_batch_size',
      help='Batch size for evaluation steps',
      type=int,
      default=10
  )
  parser.add_argument(
      '--train_steps',
      help='Steps to run the training job for.',
      type=int,
      default=10000
  )
  parser.add_argument(
      '--eval_steps',
      help='Number of steps to run evalution for at each checkpoint',
      default=100,
      type=int
  )
  parser.add_argument(
      '--output_dir',
      help='GCS location to write checkpoints and export models',
      #required=True,
      default='gs://ieor-dl-group17/models/0413191822/mirror'
  )
  parser.add_argument(
      '--job-dir',
      help='this model ignores this field, but it is required by gcloud',
      default='gs://ieor-dl-group17/models/0413191823/model'
  )
  parser.add_argument(
      '--eval_delay_secs',
      help='How long to wait before running first evaluation',
      default=10,
      type=int
  )
  parser.add_argument(
      '--min_eval_frequency',
      help='Minimum number of training steps between evaluations',
      default=1,
      type=int
  )

  args = parser.parse_args()
  arguments = args.__dict__

  # unused args provided by service
  #arguments.pop('job_dir', None)
  #arguments.pop('job-dir', None)
  arguments.pop('eval-batch-size', None)
  arguments.pop('eval_batch_size', None)


  output_dir = arguments.pop('output_dir')
  #config = tf.ConfigProto()
  #sess = tf.Session(config=config)
  #sess.run(tf.global_variables_initializer())

  tf.logging.set_verbosity(tf.logging.INFO)
  filenames = tf.gfile.Glob('gs://ieor-dl-group17/input-data/0413191822/train/*')
  model_dir = 'gs://internal-klm/models/0414191822/mirror2'
  train_and_evaluate(args)

  #learn_runner.run(generate_experiment_fn(**arguments), output_dir)
