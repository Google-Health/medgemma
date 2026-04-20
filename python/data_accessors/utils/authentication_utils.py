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
# ==============================================================================
"""Authentication utils for data sources."""

from ez_wsi_dicomweb import credential_factory

from data_accessors.utils import json_validation_utils

_APPLICATION_DEFAULT_BEARER_TOKEN = 'application_default'


def create_auth_from_instance(
    bearer_token: str,
) -> credential_factory.AbstractCredentialFactory:
  json_validation_utils.validate_str(bearer_token)
  if bearer_token == _APPLICATION_DEFAULT_BEARER_TOKEN:
    return credential_factory.DefaultCredentialFactory()
  elif bearer_token:
    return credential_factory.TokenPassthroughCredentialFactory(bearer_token)
  else:
    return credential_factory.NoAuthCredentialsFactory()
