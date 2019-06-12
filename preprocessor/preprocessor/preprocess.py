"""Build preprocessing pipeline"""

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, WorkerOptions
from tensorflow_transform.beam import impl as tft_beam
from tensorflow_transform.beam import tft_beam_io
from tensorflow_transform import coders
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import dataset_schema

from tensorflow import gfile
import pandas as pd
import nibabel as nib
import random

def run(flags, pipeline_args):
    """Run Apache Beam pipeline to generate TFRecords for Survival Analysis"""
    options = PipelineOptions(flags=[], **pipeline_args)
    options.view_as(WorkerOptions).machine_type = flags.machine_type
    temp_dir = os.path.join(flags.output_dir, 'tmp')
    runner = 'DataflowRunner' if flags.cloud else 'DirectRunner' 

    files = tf.gfile.Glob(flags.input_dir+"*")

    with beam.Pipeline(runner, options=options) as p:
        with tft_beam.Context(temp_dir=temp_dir):
            filenames = (p | 'Create filenames' >> beam.Create(files))