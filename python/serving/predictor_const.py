# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Constants for the predictor module."""

from data_accessors import data_accessor_const


# Prediction Request Constants
INSTANCES = 'instances'
IMAGE_URL = 'image_url'  # generic dicom
DICOM_WEB_URI = 'dicomweb_uri'  # generic dicom
GCS_URI = 'gcs_uri'
INPUT_BYTES = 'input_bytes'
TEXT = 'text'
IMAGE = 'image'
EXTENSIONS = data_accessor_const.InstanceJsonKeys.EXTENSIONS
BEARER_TOKEN = data_accessor_const.InstanceJsonKeys.BEARER_TOKEN

TYPE = 'type'
ROLE = 'role'
CONTENT = 'content'
INPUT_TYPE = 'type'
TEXT_INPUT_TYPE = 'text'
IMAGE_TYPE_MEDGEMMA_INTERNAL = 'image'
IMAGE_TYPE_DICOM = 'image_dicom'
IMAGE_TYPE_GCS = 'image_gcs'
IMAGE_TYPE_BYTES = 'image_bytes'
IMAGE_TYPE_URL = 'image_url'
IMAGE_INPUT_TYPES = {
    IMAGE_TYPE_MEDGEMMA_INTERNAL,
    IMAGE_TYPE_DICOM,
    IMAGE_TYPE_GCS,
    IMAGE_TYPE_BYTES,
    IMAGE_TYPE_URL,
}

# embedding response type
EMBEDDING = 'embedding'
PATCH_COORDINATE = 'patch_coordinate'
PATCH_EMBEDDINGS = 'patch_embeddings'

PREDICTIONS = 'predictions'
ERROR_CODE = 'code'
ERROR_CODE_DESCRIPTION = 'description'
ERROR = 'error'
VERTEXAI_ERROR = 'error'
