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

from preprocessor import features


def get_keys(filename):
    subject = "_".join(filename.split("/")[4].split("_")[0:3])
    seriesID = filename.split("/")[4].split("_")[3:4][0][1:]
    seriesID = seriesID.split("-")[0]
    return subject, seriesID


def get_labels_array(filename):
    output = tf.gfile.Open(filename)
    csv_output = pd.read_csv(output)
    trimmed_output = csv_output[["Subject","T1.SERIESID","Group"]]
    return trimmed_output


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
            logging.warning("Label not found.")
        return None


class read_nii(beam.DoFn):
    def process(self, filename):
        import nibabel as nib
        with tf.gfile.GFile(filename, 'r') as f:
            bb = io.BytesIO(f.read())
            fh = nib.FileHolder(fileobj=bb)
            img = nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})
        subject, seriesID = get_keys(filename)
        return {
            'image': img.get_fdata().ravel(),
            'subject': subject,
            'series': seriesID
        }


@beam.ptransform_fn
def Shuffle(p):
    """Shuffles the given pCollection."""
    return (p
            | "PairWithRandom" >> beam.Map(lambda x: (random.random(), x))
            | "GroupByRandom" >> beam.GroupByKey()
            | "DropRandom" >> beam.FlatMap(lambda (k, vs): vs))


@beam.ptransform_fn
def WriteTFRecord(p, prefix, output_dir, metadata):
    """Shuffles and write the given pCollection as a TF-Record.
    Args:
        p: a pCollection.
        prefix: prefix for location tf-record will be written to.
        output_dir: the directory or bucket to write the json data.
        metadata
    """
    coder = coders.ExampleProtoCoder(metadata.schema)
    prefix = str(prefix).lower()
    _ = (p
        | "ShuffleData" >> Shuffle()  # pylint: disable=no-value-for-parameter
        | "WriteTFRecord" >> beam.io.tfrecordio.WriteToTFRecord(
            os.path.join(output_dir, 'data', prefix, prefix),
            coder=coder,
            file_name_suffix=".tfrecord"))


@beam.ptransform_fn
def randomly_split(p, train_size, validation_size, test_size):
    """Randomly splits input pipeline in three sets based on input ratio.
    Args:
        p: PCollection, input pipeline.
        train_size: float, ratio of data going to train set.
        validation_size: float, ratio of data going to validation set.
        test_size: float, ratio of data going to test set.
    Returns:
        Tuple of PCollection.
    Raises:
        ValueError: Train validation and test sizes don`t add up to 1.0.
    """
    if train_size + validation_size + test_size != 1.0:
        raise ValueError(
            'Train validation and test sizes don`t add up to 1.0.')

    class _SplitData(beam.DoFn):
        def process(self, element):
            r = random.random()
            if r < test_size:
                yield beam.pvalue.TaggedOutput('Test', element)
            elif r < 1 - train_size:
                yield beam.pvalue.TaggedOutput('Val', element)
            else:
                yield element

    split_data = (
        p | 'SplitData' >> beam.ParDo(_SplitData()).with_outputs(
            'Test',
            'Val',
            main='Train'))

    return split_data['Train'], split_data['Val'], split_data['Test']


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

    with beam.Pipeline(runner, options=options) as p:
        with tft_beam.Context(temp_dir=temp_dir):
            filenames = (p | 'Create filenames' >> beam.Create(files))
            nii = (filenames | 'Read NII' >> beam.ParDo(read_nii()))
            nii_with_labels = (nii | 'Get Label' >> beam.ParDo(
                read_label(), labels))
            raw_train, raw_eval, raw_test = (
                nii_with_labels | 'RandomlySplitData' >> randomly_split(
                    train_size=.7,
                    validation_size=.15,
                    test_size=.15))

            raw_metadata = features.get_raw_dataset_metadata()
            spec = features.get_raw_feature_spec()
            preprocess_fn = features.preprocess_fn
            transform_fn = ((raw_train, raw_metadata)
                           | 'AnalyzeTrain' >> tft_beam.AnalyzeDataset(
                                preprocess_fn))
            _ = (transform_fn
                | 'WriteTransformFn' >> tft_beam_io.WriteTransformFn(
                    flags.output_dir))

            for dataset_type, dataset in [('Train', raw_train), 
                                         ('Eval', raw_eval),
                                         ('Test', raw_test)]:
                transform_label = 'Transform{}'.format(dataset_type)
                t, metadata = (((dataset, raw_metadata), transform_fn)
                              | transform_label >> tft_beam.TransformDataset())
                if dataset_type == 'Train':
                    _ = (metadata
                        | 'WriteMetadata' >> tft_beam_io.WriteMetadata(
                            os.path.join(flags.output_dir, 
                                        'transformed_metadata'),
                            pipeline=p))
                write_label = 'Write{}TFRecord'.format(dataset_type)
                _ = t | write_label >> WriteTFRecord(
                    dataset_type, flags.output_dir, metadata)
