from tensorflow import gfile
import argparse
import sys
import os
import logging
import numpy as np
import apache_beam as beam
import nibabel as nib
import random
from apache_beam.transforms import PTransform
from apache_beam.io import tfrecordio
import tensorflow as tf
import pandas as pd
import io
from tensorflow.python.lib.io import file_io
from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions, StandardOptions, \
  SetupOptions, WorkerOptions
from apache_beam.options.pipeline_options import SetupOptions

  
def parse_arguments(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--output_dir',
      #type='str',
      default='gs://ieor-dl-group17/input-data',
  )
  parser.add_argument(
      '--log_level',
      help='Set logging level',
      default='INFO',
  )
  parser.add_argument(
      '--input_file',
      help='Input file (test)',
      default=filename,
      #default='gs://columbia-dl-storage-bucket/data/002_S_0295_S21856_T1_brain_mni305.nii',
  )
  parser.add_argument(
      '--input_dir',
      help='Input file (test)',
      #default=filename,
      default='gs://columbia-dl-storage-bucket/data/',
  )
  parser.add_argument(
      '--machine_type',
      help='Input file (test)',
      #default=filename,
      default='n1-highmem-2',
  )
  known_args, pipeline_args = parser.parse_known_args(argv)
  pipeline_args.extend([
      # CHANGE 2/5: (OPTIONAL) Change this to DataflowRunner to
      # run your pipeline on the Google Cloud Dataflow Service.
      #DirectRunner for local
      '--runner=DataflowRunner',
      # CHANGE 3/5: Your project ID is required in order to run your pipeline on
      # the Google Cloud Dataflow Service.
      '--project=ieor-dl-group17',
      # CHANGE 4/5: Your Google Cloud Storage path is required for staging local
      # files.
      '--staging_location=gs://ieor-dl-group17/staging/new',
      # CHANGE 5/5: Your Google Cloud Storage path is required for temporary
      # files.
      '--temp_location=gs://ieor-dl-group17/temp',
      '--job_name=generate-all-tfrecords-again6',
  ])
  return known_args, pipeline_args


class read_label(beam.DoFn):
  def process(self, element, labels):
    label_keys = {'CN':0, 'AD':1, 'MCI':2, 'EMCI':3, 'LMCI':4, 'SMC':5}
    try:
      label = labels[labels["Subject"] == element['subject']][labels["T1.SERIESID"] == int(element['series'])]
      return [{
        'label': label_keys[label["Group"].item()],
        'image': element['image']
      }]
    except:
      logging.warning("Series "+str(element['series']) + ", Subject " + 
          str(element['subject']) + " not found")
      return None


def get_keys(filename):
  logging.warning(filename)
  subject = "_".join(filename.split("/")[4].split("_")[0:3])
  logging.warning(subject)
  seriesID = filename.split("/")[4].split("_")[3:4][0][1:]
  #logging.warning(seriesID)
  seriesID = seriesID.split("-")[0]
  logging.warning(seriesID)
  return subject, seriesID


def _float_feature(value):
  import tensorflow as tf
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _create_tfrecord(element):
  import tensorflow as tf
  image = element['image']
  feature = {
    'label': _int64_feature(element['label']),
    'image': _float_feature(image.ravel()),
  }
  example = tf.train.Example(features=tf.train.Features(feature=feature))
  return example.SerializeToString()


def get_labels_array(filename):
  output = tf.gfile.Open(filename)
  csv_output = pd.read_csv(output)
  trimmed_output = csv_output[["Subject","T1.SERIESID","Group"]]
  return trimmed_output


class read_nii(beam.DoFn):
  def process(self, filename):
    import nibabel as nib
    with tf.gfile.GFile(filename, 'r') as f:
      bb = io.BytesIO(f.read())
      fh = nib.FileHolder(fileobj=bb)
      img = nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})
    subject, seriesID = get_keys(filename)
    return [{
      'image': img.get_fdata(), 
      'subject': subject, 
      'series': seriesID
    }]


def run(flags, pipeline_argv):
  import tensorflow as tf #handle NameError

  files = tf.gfile.Glob(flags.input_dir+"*")
  files = files[:2]
  labels = get_labels_array("gs://columbia-dl-storage-bucket/ADNI_t1_list_with_fsstatus_20190111.csv")
  #print(pipeline_argv)
  options = PipelineOptions(pipeline_argv)
  #options.project = 'ieor-dl-group17'
  #options.job_name = 'myjob'
  #options.staging_location = 'gs://ieor-dl-group17/staging'
  #options.temp_location = 'gs://ieor-dl-group17/temp'
  train_tfrecord = beam.io.WriteToTFRecord(
        'gs://ieor-dl-group17/input-data/0414191330/train/tfrecord',
        file_name_suffix='.tfrecord'
    )
  val_tfrecord = beam.io.WriteToTFRecord(
        'gs://ieor-dl-group17/input-data/0414191330/val/tfrecord',
        file_name_suffix='.tfrecord'
    )
  test_tfrecord = beam.io.WriteToTFRecord(
        'gs://ieor-dl-group17/input-data/0414191330/test/tfrecord',
        file_name_suffix='.tfrecord'
    )
  #pipeline_options = get_cloud_pipeline_options()
  options.view_as(SetupOptions).save_main_session = True
  options.view_as(WorkerOptions).machine_type = 'n1-highmem-4'


  with beam.Pipeline(options=options) as p:
    filenames = (p | 'Create filenames' >> beam.Create(files))
    nii = (filenames | 'Read NII' >> beam.ParDo(read_nii()))
    nii_with_labels = (nii | 'Get Label' >> beam.ParDo(read_label(), labels))
    tfrecord = (nii_with_labels | 'Create TFRecord' >> beam.Map(_create_tfrecord))
    train, val = (tfrecord |
      'Split dataset (train, val)' >> beam.Partition(
        lambda elem, _: int(random.uniform(0, 100) < 20), 2))
    val, test = (val |
      'Split dataset (val, test)' >> beam.Partition(
        lambda elem, _: int(random.uniform(0, 100) < 50), 2))
    train_tf = (train | 'Write Train TFRecord' >> beam.io.Write(train_tfrecord))
    val_tf = (val | 'Write Val TFRecord' >> beam.io.Write(val_tfrecord))
    test_tf = (test | 'Write Test TFRecord' >> beam.io.Write(test_tfrecord))


def main():
  flags, pipeline_argv = parse_arguments(sys.argv[1:])
  logging.basicConfig(level=getattr(logging, flags.log_level.upper()))
  run(flags, pipeline_argv)


if __name__ == '__main__':
  main()