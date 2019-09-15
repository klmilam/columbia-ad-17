## Setup

### Set up Python environment
```
virtualenv venv
source ./venv/bin/activate
pip install -r requirements.txt
```
### Set up GCP credentials
```
gcloud auth login
gcloud auth application-default login
export GOOGLE_APPLICATION_CREDENTIALS=<PATH to GCS Key for gs://columbia-dl-storage-bucket>
```

## Preprocessing
### Set Constants
```
BUCKET=gs://[GCS Bucket for TFRecord output]
NOW="$(date +%Y%m%d%H%M%S)"
OUTPUT_DIR="${BUCKET}/output_data/${NOW}"
```

### Run locally with Dataflow
```
cd preprocessor
python3 -m run_preprocessing --output_dir "${OUTPUT_DIR}"
cd ..
```
### Run on the Cloud with Dataflow
```
cd preprocessor
python3 -m run_preprocessing --cloud  --output_dir "${OUTPUT_DIR}"
cd ..
```
  

## Training
### Deploy a v3-8 TPU
The following commands will open port 22 (allowing you to SSH) and create a TPU and CPU with the given `name`. If the TPU and/or CPU of the given `name` already exist, you'll just SSH into the existing ones. 

```
gcloud compute firewall-rules create ctpu-ssh --allow=tcp:22 --source-ranges=0.0.0.0/0 \ --network=default
./ctpu up --tpu-size=v3-8 --preemptible --zone=us-central1-a --name=kmilam-tpu
```
### Clone the model code onto the VM
Since you're SSH'd into a VM, you need to clone your code onto the VM.
```
git clone https://github.com/klmilam/columbia-ad-17.git
cd columbia-ad-17
cd trainer
```

### Training
```
python3 -m task
```
