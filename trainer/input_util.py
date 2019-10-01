"""Input functions."""

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
