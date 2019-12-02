## Setup

### Set up Python environment
```
python3 -m venv venv
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
The model code be run using either the ctpu tool or Cloud AI Platform with minimal code changes. 
### Training using the ctpu tool
#### Install cptu tool
```bash
curl -O https://dl.google.com/cloud_tpu/ctpu/latest/darwin/ctpu && chmod a+x ctpu
```

#### Deploy a v3-8 TPU
You can use the ctpu tool to deploy a Google Compute Engine (GCE) TPU. 

The following commands will open port 22 (allowing you to SSH) and create a TPU and CPU with the given `name`. If the TPU and/or CPU of the given `name` already exist, you'll just SSH into the existing ones.

```bash
gcloud compute firewall-rules create ctpu-ssh --allow=tcp:22 --source-ranges=0.0.0.0/0 \
    --network=default
./ctpu up --tpu-size=v3-8 --preemptible --zone=us-central1-a --name=kmilam-tpu
```
#### Clone the model code onto the VM
Since you're SSH'd into a VM, you need to clone your code onto the VM.

If you reuse the same `name` and do not delete your CPU between uses, your code will remain on the CPU. 
```bash
git clone https://github.com/klmilam/columbia-ad-17.git
cd columbia-ad-17
```

#### Start training
```bash
python3 -m trainer.task
```

### Training using Cloud AI Platform
Cloud AI Platform is a managed service for training machine learning models. This means that we do not deploy TPU/CPU resources; this is managed by the service.
```bash
gcloud ai-platform jobs submit training "tpu_training_$(date +%Y%m%d%H%M%S)" \
        --staging-bucket "gs://internal-klm-tpu" \
        --config config.yaml \
        --module-name trainer.task \
        --package-path trainer/ \
        --region us-central1
```

#### Train on v2-8 TPU
If v3-8 TPU resources are insufficient, try running the model on a v2-8 TPU. This will have the same number of shards as the v3-8 TPU, so no code changes (i.e. changing hyperparameters) are necessary.
```bash
gcloud ai-platform jobs submit training "tpu_training_$(date +%Y%m%d%H%M%S)" \
        --staging-bucket "gs://internal-klm-tpu" \
        --runtime-version 1.14 \
        --python-version 3.5 \
        --scale-tier BASIC_TPU \
        --module-name trainer.task \
        --package-path trainer/ \
        --region us-central1
```

#### Hyperparameter Tuning
Cloud AI Platform offers built-in support for hyperparameter tuning.

We'll use a v2-8 TPU for hyperparameter tuning, since we'll need multiple TPUs for each hptuning trial. Ideally, we would run more than 2 trails in parallel. However, we only have quota for 16 TPU V2s, so we can only run 2 concurrent trials (each on a v2-8 TPU).

```bash
gcloud ai-platform jobs submit training "tpu_training_$(date +%Y%m%d%H%M%S)" \
        --staging-bucket "gs://internal-klm-tpu" \
        --config hptuning.yaml \
        --runtime-version 1.14 \
        --python-version 3.5 \
        --scale-tier BASIC_TPU \
        --module-name trainer.task \
        --package-path trainer/ \
        --region us-central1
```
