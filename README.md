## Setup

### Set up Python environment
```
virtualenv venv
source ./venv/bin/activate
pip3 install -r requirements.txt
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
When testing or debugging a Dataflow pipeline, it's easier to run the pipeline locally first. Due to the memory and computation requirements of the full dataset, the dataset is limited to just 100 files when running locally.
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
### Install cptu tool
```bash
curl -O https://dl.google.com/cloud_tpu/ctpu/latest/darwin/ctpu && chmod a+x ctpu
```

### Deploy a v3-8 TPU
You will use the ctpu tool to deploy a Google Compute Engine (GCE) TPU. 

The following commands will open port 22 (allowing you to SSH) and create a TPU and CPU with the given `name`. If the TPU and/or CPU of the given `name` already exist, you'll just SSH into the existing ones.

```
gcloud compute firewall-rules create ctpu-ssh --allow=tcp:22 --source-ranges=0.0.0.0/0 \
    --network=default
./ctpu up --tpu-size=v3-8 --preemptible --zone=us-central1-a --name=kmilam-tpu
```
### Clone the model code onto the VM
Since you're SSH'd into a VM, you need to clone your code onto the VM.

If you reuse the same `name` and do not delete your CPU between uses, your code will remain on the CPU. 
```
git clone https://github.com/klmilam/columbia-ad-17.git
cd columbia-ad-17
cd trainer
```

### Training
```
python3 -m task
```
