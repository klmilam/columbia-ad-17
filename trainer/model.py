import tensorflow as tf
#from tensorflow.contrib.learn.python.learn.estimators.model_fn import ModeKeys as Modes
#from tensorflow.contrib.distribute.python import parameter_server_strategy
from tensorflow.python.estimator import run_config


tf.logging.set_verbosity(tf.logging.INFO)

BATCH_SIZE=8

LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]

MODEL_DIR='gs://internal-klm/models/0414191822/mirror2'

def learning_rate_schedule(current_epoch):
  """Handles linear scaling rule, gradual warmup, and LR decay.
  The learning rate starts at 0, then it increases linearly per step.
  After 5 epochs we reach the base learning rate (scaled to account
    for batch size).
  After 30, 60 and 80 epochs the learning rate is divided by 10.
  After 90 epochs training stops and the LR is set to 0. This ensures
    that we train for exactly 90 epochs for reproducibility.
  Args:
    current_epoch: `Tensor` for current epoch.
  Returns:
    A scaled `Tensor` for current learning rate.
  """
  scaled_lr = .0000002 * (7943 / 256.0)

  decay_rate = (scaled_lr * LR_SCHEDULE[0][0] *
                current_epoch / LR_SCHEDULE[0][1])
  for mult, start_epoch in LR_SCHEDULE:
    decay_rate = tf.where(current_epoch < start_epoch,
                          decay_rate, scaled_lr * mult)
  return decay_rate

def _cnn_model_fn(features, labels, mode):
  input_layer = tf.reshape(features, [-1, 256, 256, 256, 1])
  conv1 = tf.layers.conv3d(
    inputs=input_layer,
    filters=12,
    kernel_size=[7,7,7],
    strides=[2, 2, 2],
    padding='same',
    activation=tf.nn.leaky_relu
    )
  pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[2, 2, 2], strides=2)
  conv2 = tf.layers.conv3d(
    inputs=pool1,
    filters=12,
    kernel_size=[3,3,3],
    padding='same',
    activation=tf.nn.leaky_relu
    )
  pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2,2,2], strides=2)
  conv3 = tf.layers.conv3d(
    inputs=pool2,
    filters=64,
    kernel_size=[3,3,3],
    padding='same',
    activation=tf.nn.leaky_relu
    )
  pool1_flat = tf.layers.flatten(conv3)
  dense1 = tf.layers.dense(inputs=pool1_flat, units=32, activation=tf.nn.relu)
  logits = tf.layers.dense(inputs=dense1, units=6, activation=tf.nn.softmax,
    kernel_initializer=tf.random_normal_initializer(stddev=.01))
  #logits = tf.cast(tf.identity(logits, 'final_dense'), tf.float32)
  tf.logging.info(logits)
  #logits = tf.print(logits, [logits], message="logits: ")
  one_hot_labels = tf.one_hot(labels, 6)
  tf.logging.info(tf.shape(one_hot_labels))
  #why is loss so large? may be causing loss -> NaN
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels))
  tf.logging.info(tf.shape(loss))
  host_call=None
  if mode == tf.estimator.ModeKeys.TRAIN:
    # Compute the current epoch and associated learning rate from global_step.
    global_step = tf.train.get_global_step()
    batches_per_epoch = 7943/BATCH_SIZE
    learning_rate = tf.train.exponential_decay(
        learning_rate=0.01, global_step=global_step,
        decay_steps=100, decay_rate=0.001)
    #optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    optimizer = 
    ##gvs = optimizer.compute_gradients(loss)
    #capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    global_step = tf.train.get_global_step()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      #train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
      train_op = optimizer.minimize(loss, global_step=global_step)

    def host_call_fn(gs, loss, lr, ce):
      """Training host call. Creates scalar summaries for training metrics.
      This function is executed on the CPU and should not directly reference
      any Tensors in the rest of the `model_fn`. To pass Tensors from the
      model to the `metric_fn`, provide as part of the `host_call`. See
      https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
      for more information.
      Arguments should match the list of `Tensor` objects passed as the second
      element in the tuple passed to `host_call`.
      Args:
        gs: `Tensor with shape `[batch]` for the global_step
        loss: `Tensor` with shape `[batch]` for the training loss.
        lr: `Tensor` with shape `[batch]` for the learning_rate.
        ce: `Tensor` with shape `[batch]` for the current_epoch.
      Returns:
        List of summary ops to run on the CPU host.
      """
      gs = gs[0]
      with summary.create_file_writer(MODEL_DIR).as_default():
        with summary.always_record_summaries():
          summary.scalar('loss', loss[0], step=gs)
          summary.scalar('learning_rate', lr[0], step=gs)
          return summary.all_summary_ops()

    gs_t = tf.reshape(global_step, [1])
    loss_t = tf.reshape(loss, [1])
  else:
    train_op = None

  eval_metrics = None
  #if mode == tf.estimator.ModeKeys.EVAL or :
  def metric_fn(labels, logits):
    predictions = tf.argmax(logits, axis=1)

    return {
      'accuracy': tf.metrics.accuracy(
          labels=labels, predictions=predictions)
    }

  eval_metrics = metric_fn(labels, logits)

  return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            #host_call=host_call,
            eval_metric_ops=eval_metrics)

def build_estimator(model_dir, config=None):
  #strategy = tf.contrib.distribute.MirroredStrategy()
  #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
  #config.gpu_options.allow_growth = True
  config = tf.estimator.RunConfig(#train_distribute=strategy, eval_distribute=strategy,
    save_checkpoints_steps=100)
  #tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver()
  # config = tf.contrib.tpu.RunConfig(
  #   tpu_config=tpu_config.TPUConfig(num_shards=None, iterations_per_loop=100))
  #    eval_distribute=strategy)
  #return tf.estimator.Estimator(
  #    model_fn=_cnn_model_fn,
  #    model_dir=model_dir,
  #    config=config)
  return tf.contrib.tpu.TPUEstimator(
      model_fn=_cnn_model_fn,
      #model_dir=model_dir,
      use_tpu=True,
      config=config)

