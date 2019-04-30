# columbia-ad-17

$ mkvirtualenv <environment name>
$ pip install -r requirements.txt
$ gcloud auth application-default login

export GOOGLE_APPLICATION_CREDENTIALS='path/to/columbia-dl-storage-bucket key'

python ./preprocessor/test.py --setup_file ./setup.py
