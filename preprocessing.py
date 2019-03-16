import tensorflow as tf
import nibabel as nib
import numpy as np
import os
from google.oauth2 import service_account
from google.cloud import storage

INPUT_API_KEY="columbia-dl-storage-ab27543d7772.json"

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


def run(filename, bucket_name, output_filename):
  """Creates TFRecords and adds them to local directory
  """
  #filenames = os.listdir(input_dir)
  credentials = service_account.Credentials.from_service_account_file(INPUT_API_KEY)
  client = storage.Client(credentials=credentials,project="columbia-dl-storage")
  bucket = client.get_bucket(bucket_name)
  blob = bucket.blob(filename)
  blob.download_to_filename('example.nii')
  create_tfrecord('example.nii', output_filename)


def main():
  #TODO: add CLI argument parsing
  #TODO: add logging
  #input_dir = './input_files'
  filename = "data/002_S_0295_S21856_T1_brain_mni305.nii"
  bucket_name = "columbia-dl-storage-bucket"
  output_filename = 'train.tfrecords'
  run(filename, bucket_name, output_filename)


if __name__ == '__main__':
  main()