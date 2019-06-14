"""Build preprocessing pipeline"""

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, WorkerOptions
from tensorflow_transform.beam import impl as tft_beam
from tensorflow_transform.beam import tft_beam_io
from tensorflow_transform import coders
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import dataset_schema

import tensorflow as tf
from tensorflow import gfile
import pandas as pd
import logging
import nibabel as nib
import random
import os
import io
import numpy as np


def get_keys(filename):
    subject = "_".join(filename.split("/")[4].split("_")[0:3])
    seriesID = filename.split("/")[4].split("_")[3:4][0][1:]
    seriesID = seriesID.split("-")[0]
    return subject, seriesID


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_labels_array(filename):
    output = tf.gfile.Open(filename)
    csv_output = pd.read_csv(output)
    trimmed_output = csv_output[["Subject","T1.SERIESID","Group"]]
    return trimmed_output


def _create_tfrecord(element):
    import tensorflow as tf
    image = element['image']
    feature = {
        'label': _int64_feature(element['label']),
        'image': _float_feature(image.ravel()),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


class read_label(beam.DoFn):
    def process(self, element, labels):
        label_keys = {'CN':0, 'AD':1, 'MCI':2, 'EMCI':3, 'LMCI':4, 'SMC':5}
        try:
            label = labels[labels["Subject"] == \
                element['subject']][labels["T1.SERIESID"] == \
                int(element['series'])]
            return [{
                'label': label_keys[label["Group"].item()],
                'image': element['image']
            }]
        except:
            logging.warning("Series "+str(element['series']) + ", Subject " +
                str(element['subject']) + " not found")
        return None


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


def run(flags, pipeline_args):
    """Run Apache Beam pipeline to generate TFRecords for Survival Analysis"""
    options = PipelineOptions(flags=[], **pipeline_args)
    options.view_as(WorkerOptions).machine_type = flags.machine_type
    temp_dir = os.path.join(flags.output_dir, 'tmp')
    runner = 'DataflowRunner' if flags.cloud else 'DirectRunner' 
    options.view_as(WorkerOptions).machine_type = flags.machine_type

    files = tf.gfile.Glob(flags.input_dir+"*")
    labels = get_labels_array(
            "gs://columbia-dl-storage-bucket/ADNI_t1_list_with_fsstatus_20190111.csv")
    train_tfrecord = beam.io.WriteToTFRecord(
        os.path.join(flags.output_dir,'train/tfrecord'),
        file_name_suffix='.tfrecord'
    )
    val_tfrecord = beam.io.WriteToTFRecord(
        os.path.join(flags.output_dir,'val/tfrecord'),
        file_name_suffix='.tfrecord'
    )
    test_tfrecord = beam.io.WriteToTFRecord(
        os.path.join(flags.output_dir,'test/tfrecord'),
        file_name_suffix='.tfrecord'
    )

    with beam.Pipeline(runner, options=options) as p:
        with tft_beam.Context(temp_dir=temp_dir):
            filenames = (p | 'Create filenames' >> beam.Create(files))
            nii = (filenames | 'Read NII' >> beam.ParDo(read_nii()))
            nii_with_labels = (nii | 'Get Label' >> beam.ParDo(read_label(), labels))
            tfrecord = (nii_with_labels | 'Create TFRecord' >> beam.Map(_create_tfrecord))
            train, val = (
                tfrecord | 'Split dataset (train, val)' >> beam.Partition(
                    lambda elem, _: int(random.uniform(0, 100) < 20), 2))
            val, test = (
                val | 'Split dataset (val, test)' >> beam.Partition(
                    lambda elem, _: int(random.uniform(0, 100) < 50), 2))
            train_tf = (train | 'Write Train TFRecord' >> beam.io.Write(train_tfrecord))
            val_tf = (val | 'Write Val TFRecord' >> beam.io.Write(val_tfrecord))
            test_tf = (test | 'Write Test TFRecord' >> beam.io.Write(test_tfrecord))
