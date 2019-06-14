"""Start preprocessing job for converting NII files to TFRecords"""

import argparse
import sys
import os
import logging
from datetime import datetime
import posixpath

from preprocessor import preprocess

def parse_arguments(argv):
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    parser.add_argument(
        '--job_name',
        default='{}-{}'.format('nii-to-tfrecords', timestamp)
    )
    parser.add_argument(
        '--output_dir',
        default=os.path.join('gs://ieor-dl-group17/mri', timestamp)
    )
    parser.add_argument(
        '--log_level',
        help='Set logging level',
        default='INFO'
    )
    parser.add_argument(
        '--machine_type',
        help="""Set machine type for Dataflow worker machines.""",
        default='n1-highmem-4'
    )
    parser.add_argument(
        '--cloud',
        help="""Run preprocessing on the cloud. Default False.""",
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--project_id',
        help="""Google Cloud project ID""",
        default='ieor-dl-group17'
    )
    parser.add_argument(
        '--input_dir',
        help="""GCS directory where NII files are stored.""",
        default='gs://columbia-dl-storage-bucket/data/'
    )
    known_args, _ = parser.parse_known_args(argv)
    return known_args


def get_pipeline_args(flags):
    """Create Apache Beam pipeline arguments"""
    options = {
        'project': flags.project_id,
        'staging_location': os.path.join(flags.output_dir, 'staging'),
        'temp_location': os.path.join(flags.output_dir, 'temp'),
        'job_name': flags.job_name,
        'save_main_session': True,
        'setup_file': posixpath.abspath(
            posixpath.join(posixpath.dirname(__file__), 'setup.py'))
    }
    return options


def main():
    flags = parse_arguments(sys.argv[1:])
    pipeline_args = get_pipeline_args(flags)
    logging.basicConfig(level=getattr(logging, flags.log_level.upper()))
    preprocess.run(flags, pipeline_args)


if __name__ == '__main__':
    main()
