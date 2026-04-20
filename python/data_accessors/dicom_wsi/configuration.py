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

"""Configuration dataclasses for DICOM Digital Pathology data accessor."""

import dataclasses
from typing import Optional, Sequence


@dataclasses.dataclass(frozen=True)
class IccProfileCacheConfiguration:
  gcs_bucket: str = ''
  redis_ip: str = ''
  redis_port: int = 6379
  store_icc_profile_bytes_in_redis: bool = False
  testing: bool = False


@dataclasses.dataclass(frozen=True)
class ConfigurationSettings:
  endpoint_input_width: int
  endpoint_input_height: int
  approved_dicom_stores: Optional[Sequence[str]]
  icc_profile_cache_configuration: IccProfileCacheConfiguration
  require_patch_dim_match_default_dim: bool = False
  max_parallel_download_workers: int = 1

