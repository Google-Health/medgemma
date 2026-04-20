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

"""MedSigLIP flags."""

import json
import os
import sys
from typing import List, Optional, Union

from absl import flags

from serving.logging_lib.flags import flag_utils

# Endpoint configuration flags.

WORKER_DOWNLOAD_PARALLELISM = flags.DEFINE_enum(
    'worker_download_parallelism',
    'PROCESS',
    ['THREAD', 'PROCESS'],
    'Worker type to use when downloading instance data.',
)


MAX_PARALLEL_DOWNLOAD_WORKERS_FLAG = flags.DEFINE_integer(
    'max_parallel_download_workers',
    int(os.environ.get('MAX_PARALLEL_DOWNLOAD_WORKERS', 3)),
    'Max number of parallel workers to use when downloading instance data.',
)

THREAD_POOL_MAX_WORKERS_FLAG = flags.DEFINE_integer(
    'thread_pool_max_workers',
    int(os.environ.get('THREAD_POOL_MAX_WORKERS', 4)),
    'Max parallel workers async inference workers or async data loading.',
)

THREAD_POOL_TIMEOUT_FLAG = flags.DEFINE_integer(
    'thread_pool_timeout',
    int(os.environ.get('THREAD_POOL_TIMEOUT', 1800)),  # 30 minutes
    'Thread pool thread timeout in seconds.',
)


def _load_multi_string(val: Optional[str]) -> Optional[Union[List[str], str]]:
  if val is None:
    return None
  try:
    return json.loads(val)
  except json.decoder.JSONDecodeError:
    return val


# If true and Redis host is defined stores ICC Profile bytes in redis.
ICC_PROFILE_CACHE_GCS_BUCKET_FLAG = flags.DEFINE_string(
    'icc_profile_cache_gcs_bucket',
    os.environ.get('ICC_PROFILE_CACHE_GCS_BUCKET', ''),
    'Name of gcs bucket to cache icc profile to.',
)

ICC_PROFILE_CACHE_REDIS_IP_FLAG = flags.DEFINE_string(
    'icc_profile_cache_redis_ip',
    os.environ.get('ICC_PROFILE_CACHE_REDIS_IP', ''),
    'IP address of REDIS server to cache cache icc profile to.',
)

ICC_PROFILE_CACHE_REDIS_PORT_FLAG = flags.DEFINE_integer(
    'icc_profile_cache_redis_port',
    int(os.environ.get('ICC_PROFILE_CACHE_REDIS_PORT', '6379')),
    'Port of REDIS server to cache cache icc profile to.',
)

# If true and Redis host is defined stores ICC Profile bytes in redis.
STORE_ICC_PROFILE_BYTES_IN_REDIS_FLAG = flags.DEFINE_bool(
    'store_icc_profile_bytes_in_redis',
    flag_utils.env_value_to_bool('STORE_ICC_PROFILE_BYTES_IN_REDIS', False),
    'bool cache icc profile bytes in redis',
)

# If true and Redis host is defined stores ICC Profile bytes in redis.
IS_DEBUGGING_FLAG = flags.DEFINE_bool(
    'is_debugging',
    flag_utils.env_value_to_bool(
        'IS_DEBUGGING',
        'UNITTEST_ON_FORGE' in os.environ or 'unittest' in sys.modules,
    ),
    'internal flag for unit tests detects if running in debugger.',
)

APPROVED_GCS_SOURCE_LIST_FLAG = flags.DEFINE_multi_string(
    'approved_gcs_source_list',
    _load_multi_string(os.environ.get('APPROVED_GCS_SOURCE_LIST', None)),
    'List of GCS buckets endpoints can read from; all are allowed if'
    ' undefined.',
)


APPROVED_DICOM_STORE_SOURCE_LIST_FLAG = flags.DEFINE_multi_string(
    'approved_dicom_store_source_list',
    _load_multi_string(
        os.environ.get('APPROVED_DICOM_STORE_SOURCE_LIST', None)
    ),
    'List of DICOM stores endpoint can read from; all are allowed if'
    ' undefined.',
)

MODEL_INPUT_WIDTH_FLAG = flags.DEFINE_integer(
    'model_input_width',
    int(os.environ.get('MODEL_INPUT_WIDTH', 224)),
    'Width in pixels of input image to model.',
)

MODEL_INPUT_HEIGHT_FLAG = flags.DEFINE_integer(
    'model_input_height',
    int(os.environ.get('MODEL_INPUT_HEIGHT', 224)),
    'Height in pixels of input image to model.',
)

IMAGE_INPUT_COMPRESSION_FORMAT_FLAG = flags.DEFINE_string(
    'image_input_compression_format',
    os.environ.get('IMAGE_INPUT_COMPRESSION_FORMAT', 'jpeg'),
    'Compression format of input image.',
)

IMAGE_INPUT_JPEG_COMPRESSION_QUALITY_FLAG = flags.DEFINE_integer(
    'image_input_compression_quality',
    int(os.environ.get('IMAGE_INPUT_JPEG_COMPRESSION_QUALITY', 95)),
    'Compression quality of input image.',
)

IMAGE_SIZE_OPTIMIZEATION_FLAG = flags.DEFINE_bool(
    'image_size_optimization',
    flag_utils.env_value_to_bool('IMAGE_SIZE_OPTIMIZATION', False),
    'Optional optimization if image size does exceed model input size then '
    'shrink dim of the image to match the model encoder input size. The '
    'optimization reduces the size of the image sent to the model encoder. '
    'For imaging that exceeds the models input size, the optimization will '
    'reduce memory and compute costs associated with execution.',
)
