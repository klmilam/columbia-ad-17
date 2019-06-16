## Setup

### Set up Python environment
Python 2 is currently required for Dataflow.
```
virtualenv venv --python=/usr/bin/python2.7
source ./venv/bin/activate
pip install -r requirements.txt
```
### Set up GCP credentials
```
gcloud auth login
gcloud auth application-default login
export GOOGLE_APPLICATION_CREDENTIALS=<PATH to GCS Key for gs://columbia-dl-storage-bucket>
```