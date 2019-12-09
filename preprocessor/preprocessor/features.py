# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Feature management for data preprocessing."""

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema


LABEL_COLUMN = 'label'

KEY_COLUMNS = ['series', 'subject']

IMAGE_COLUMN = 'image'

CATEGORICAL_COLUMNS = []
STRING_COLUMNS = KEY_COLUMNS
NUMERIC_COLUMNS = [LABEL_COLUMN]
NUMERIC_LIST_COLUMNS =[IMAGE_COLUMN]
BOOLEAN_COLUMNS = []


def get_raw_feature_spec():
    """Returns TF feature spec for preprocessing"""
    features = dict(
        [(name, tf.FixedLenFeature([], tf.string))
            for name in CATEGORICAL_COLUMNS] +
        [(name, tf.FixedLenFeature([], tf.string))
            for name in STRING_COLUMNS] +
        [(name, tf.FixedLenFeature([], tf.float32))
            for name in NUMERIC_COLUMNS] +
        [(name, tf.FixedLenFeature([], tf.int64))
            for name in BOOLEAN_COLUMNS] +
        [(name, tf.FixedLenFeature([1, 160*160*32], tf.float32))
            for name in NUMERIC_LIST_COLUMNS]
    )
    return features


RAW_FEATURE_SPEC = get_raw_feature_spec()

def preprocess(inputs):
    return inputs.copy()
