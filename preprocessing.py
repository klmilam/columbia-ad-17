import tensorflow as tf
import nibabel as nib
import numpy as np
import os

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def serialize_example(image):
  #TODO: should data be ravelled?
  feature = {
    'train/image': _float_feature(image.ravel())
  }
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def create_tfrecord(filenames, tfrecord_filename):
  """Load each image and convert to add to TFRecord
  TFRecord is added to local directory
  """
  with tf.python_io.TFRecordWriter(tfrecord_filename) as writer:
    for filename in filenames:
      image = nib.load(filename)
      image_data = image.get_fdata()
      example = serialize_example(image_data)
      writer.write(example)


def run(input_dir, output_filename):
  """Creates TFRecords and adds them to local directory
  """
  filenames = os.listdir(input_dir)
  create_tfrecord(filenames, output_filename)


def main():
  #TODO: add CLI argument parsing
  #TODO: add logging
  input_dir = './input_files'
  output_filename = 'train.tfrecords'
  run(input_dir, output_filename)


if __name__ == '__main__':
  main()