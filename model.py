import tensorflow as tf
import argparse
from google.oauth2 import service_account
from google.cloud import storage
import os

API_KEY="ieor-dl-group17-3493a54f706a.json"


def decode(serialized_example):
  features = {
    'train/image': tf.FixedLenFeature([], tf.float32),
    'train/label': tf.FixedLenFeature([], tf.string)
  }
  example = tf.parse_single_example(serialized_example, features)
  return example['train/image'], example['train/label']


def main():
  credentials = service_account.Credentials.from_service_account_file(API_KEY)
  output_client = storage.Client(credentials=credentials, project="ieor-dl-group17")
  bucket = output_client.get_bucket("ieor-dl-group17")

  filenames = [os.path.join('gs://ieor-dl-group17/', f.name) for f in
      bucket.list_blobs(prefix='input-data')]
  filenames = tf.data.Dataset.list_files(filenames, shuffle=True)
  dataset = tf.data.TFRecordDataset(filenames).map(decode)
  dataset = dataset.repeat(None)
  dataset = dataset.batch(100) #TODO: replace with batch_size
  dataset = dataset.prefetch(1)

  iterator = dataset.make_initializeable_iterator()
  features, labels = iterator.get_next()

  #TODO: replace with num_epochs
#  ds_train = tf.data.TFRecordDataset(train_records, num_parallel_reads=4).map(decode)


if __name__ == '__main__':
  main()