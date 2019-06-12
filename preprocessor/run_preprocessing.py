import argparse
import sys
from datetime import datetime


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
        default=os.path.join('gs://internal-klm/mri', timestamp)
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
        default='internal-klm'
    )
    known_args, _ = parser.parse_known_args(argv)
    return known_args


def main():
    flags = parse_arguments(sys.argv[1:])
    #TODO(kmilam): add pipeline args


if __name__ == '__main__':
    main()
